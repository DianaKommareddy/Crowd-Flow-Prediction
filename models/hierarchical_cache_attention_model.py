import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from timm.models.layers import DropPath
except ImportError:
    DropPath = nn.Identity

# Activation function
def get_activation(name="mish"):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish":
        return nn.SiLU()
    elif name == "mish":
        return nn.Mish()
    else:
        raise ValueError("Unsupported activation")

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

# FeedForward with convolution and activation
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, activation="mish", dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# Inter Modulation module
class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()
        self.align = nn.AdaptiveAvgPool2d((1,1))
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2*in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)
    def forward(self, x1, x2):
        a = self.align(x2).squeeze(-1).squeeze(-1)  
        a = a.unsqueeze(-1)  
        x2_conv = self.conv_width(a)  
        scale, shift = x2_conv.chunk(2, dim=1)  
        if x1.dim() == 3:
            L = x1.shape[-1]
            scale = scale.expand(-1, -1, L)
            shift = shift.expand(-1, -1, L)
            gated = x1 * scale + shift
            gating = F.gelu(self.gatingConv(gated))
            out = gated * gating
        else:
            gated = x1 * scale.squeeze(-1) + shift.squeeze(-1)
            gating = F.gelu(self.gatingConv(gated.unsqueeze(-1))).squeeze(-1)
            out = gated * gating
        return out

# Intra Modulation module
class Intra_CacheModulation(nn.Module):
    def __init__(self, embed_dim=48):
        super(Intra_CacheModulation, self).__init__()
        self.down = nn.Conv1d(embed_dim, max(embed_dim//2, 1), kernel_size=1)
        self.up = nn.Conv1d(max(embed_dim//2, 1), embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
    def forward(self, x1, x2):
        gated = F.gelu(self.gatingConv(x2 + x1)) * (x2 + x1)
        out = self.up(self.down(gated))
        return out

class ReGroup(nn.Module):
    def __init__(self, groups=[1,1,2,4]):
        super(ReGroup, self).__init__()
        self.groups = groups
    def forward(self, query, key, value):
        b, C, L = query.shape
        channel_features = query.mean(dim=2) 
        channel_mean = channel_features.mean(dim=0)  
        _, sorted_indices = torch.sort(channel_mean, descending=True)
        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        total_ratio = sum(self.groups)
        base_sizes = [int(r / total_ratio * C) for r in self.groups]
        assigned = sum(base_sizes)
        if assigned < C:
            base_sizes[-1] += (C - assigned)

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        for group_size in base_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx
        return query_groups, key_groups, value_groups

def CalculateCurrentLayerCache(x_groups, dim=128, groups=[1,1,2,4]):
    """
    x_groups: list of attn tensors per group, each shaped (b, Cg, L)
    dim: target channel dimension (C) to upsample/compose to
    Returns: qv_cache sized (b, dim, S, S) with S chosen = ceil(sqrt(L)) or 1
    """
    b = x_groups[0].shape[0]
    device = x_groups[0].device
    L = x_groups[0].shape[-1]
    s = int((L ** 0.5) + 0.9999)
    accum = torch.zeros((b, dim, s, s), device=device)
    total = 0.0
    for i, g in enumerate(x_groups):
        gm = g.mean(dim=-1, keepdim=True) 
        gm2 = gm.unsqueeze(-1)  
        Cg = gm2.shape[1]
        rep = max(1, dim // Cg)
        gm_rep = gm2.repeat(1, rep, 1, 1)
        if gm_rep.shape[1] > dim:
            gm_rep = gm_rep[:, :dim, :, :]
        elif gm_rep.shape[1] < dim:
            pad = dim - gm_rep.shape[1]
            gm_rep = torch.cat([gm_rep, torch.zeros(b, pad, 1, 1, device=device)], dim=1)
        up = F.interpolate(gm_rep, size=(s, s), mode='bilinear', align_corners=False)
        accum = accum + up
        total += 1.0
    if total > 0:
        accum = accum / total
    return accum  

# Modulated Attention with inter- and intra-cache
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.group = [1, 2, 2, 3]
        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)
        self.inter_modulator1 = Inter_CacheModulation(in_c=max(1, dim//8))
        self.inter_modulator2 = Inter_CacheModulation(in_c=max(1, (2*dim)//8))
        self.inter_modulator3 = Inter_CacheModulation(in_c=max(1, (2*dim)//8))
        self.inter_modulator4 = Inter_CacheModulation(in_c=max(1, (3*dim)//8))
        self.inter_modulators = [self.inter_modulator1, self.inter_modulator2, self.inter_modulator3, self.inter_modulator4]
        self.regroup = ReGroup(self.group)
        self.dim = dim

    def forward(self, x, qv_cache=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        qu, ke, va = self.regroup(q, k, v)  
        attScore = []
        tmp_cache = []
        # for each group compute attention (scaled)
        for idx in range(len(self.group)):
            query_head = qu[idx]  
            key_head = ke[idx]
            val_head = va[idx]
            query_head = torch.nn.functional.normalize(query_head, dim=-1)
            key_head = torch.nn.functional.normalize(key_head, dim=-1)
            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[idx, :, :]
            attn = attn.softmax(dim=-1)
            attScore.append(attn)
            t_cache = query_head.clone().detach() + key_head.clone().detach()
            tmp_cache.append(t_cache)

        tmp_caches = torch.cat(tmp_cache, 1) 

        # Inter Modulation
        out = []
        if qv_cache is not None:
            if qv_cache.shape[1] != c:
                qv_cache = F.adaptive_avg_pool2d(qv_cache, 1) 
                qv_cache = qv_cache
        for i in range(len(self.group)):
            if qv_cache is not None:
                inter_modulator = self.inter_modulators[i]
                attScore[i] = inter_modulator(attScore[i], qv_cache) + attScore[i]
                out.append(attScore[i] @ va[i])
            else:
                out.append(attScore[i] @ va[i])

        update_factor = 0.9
        update_elements = CalculateCurrentLayerCache(tmp_cache, dim=c, groups=self.group)
        if qv_cache is not None:
            ue = update_elements
            if qv_cache.shape[-2:] != ue.shape[-2:]:
                qv_cache = F.interpolate(qv_cache, size=ue.shape[-2:], mode='bilinear', align_corners=False)
            qv_cache = qv_cache * update_factor + ue * (1 - update_factor)
        else:
            qv_cache = update_elements * (update_factor)

        out_all = torch.cat(out, dim=1)  
        # Intra Modulation
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all
        out_all = rearrange(out_all, 'b c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)
        return [out_all, qv_cache]

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias', isAtt=True):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, 'mish', 0.1)
    def forward(self, inputs):
        x = inputs[0]
        qv_cache = inputs[1]
        if self.isAtt:
            x_tmp = x
            x_att, qv_cache = self.attn(self.norm1(x), qv_cache=qv_cache)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))
        return [x, qv_cache]

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class HCAM(nn.Module):
    def __init__(self, 
                 inp_channels=9, 
                 out_channels=1, 
                 dim=48,
                 num_blocks=[4,6,6,8], 
                 num_refinement_blocks=4,
                 heads=[1,2,4,8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 qv_cache=None,
                 name="HCAM"):
        super(HCAM, self).__init__()
        self.qv_cache = qv_cache
        self.name = name

        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(dim*4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim*8, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True) for _ in range(num_blocks[1])
        ])

        self.up4_3 = Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim*2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True) for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim*2, 1, bias=bias)

        self.output = nn.Conv2d(dim*2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)  
        out_enc_level1, self.qv_cache = self.encoder_level1([inp_enc_level1, self.qv_cache])
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, self.qv_cache = self.encoder_level2([inp_enc_level2, self.qv_cache])
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, self.qv_cache = self.encoder_level3([inp_enc_level3, self.qv_cache])
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, self.qv_cache = self.latent([inp_enc_level4, self.qv_cache])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, self.qv_cache = self.decoder_level3([inp_dec_level3, self.qv_cache])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, self.qv_cache = self.decoder_level2([inp_dec_level2, self.qv_cache])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1, self.qv_cache = self.decoder_level1([inp_dec_level1, self.qv_cache])

        out_dec_level1, self.qv_cache = self.refinement([out_dec_level1, self.qv_cache])

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :1, :, :]

        return out_dec_level1
