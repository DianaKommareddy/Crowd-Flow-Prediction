import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.models.layers import DropPath
except ImportError:
    DropPath = nn.Identity


# -----------------------
# 🔹 Activation Function
# -----------------------
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


# -----------------------
# 🔹 Layer Normalization
# -----------------------
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


# -----------------------
# 🔹 FeedForward Network
# -----------------------
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


# -----------------------
# 🔹 Inter Cache Modulation
# -----------------------
class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2 * in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)

    def forward(self, x1, x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2) + x1)
        scale, shift = x2_pW.chunk(2, dim=1)
        x1_p = x1 * scale + shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p


# -----------------------
# 🔹 Intra Cache Modulation
# -----------------------
class Intra_CacheModulation(nn.Module):
    def __init__(self, embed_dim=48):
        super().__init__()
        self.down = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)

    def forward(self, x1, x2):
        x_gated = F.gelu(self.gatingConv(x2 + x1)) * (x2 + x1)
        x_p = self.up(self.down(x_gated))
        return x_p


# -----------------------
# 🔹 ReGroup Channels
# -----------------------
class ReGroup(nn.Module):
    def __init__(self, groups=[1, 1, 2, 4]):
        super().__init__()
        self.groups = groups

    def forward(self, query, key, value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)
        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True)

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups, key_groups, value_groups = [], [], []
        start_idx = 0
        total_ratio = sum(self.groups)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.groups]

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx

        return query_groups, key_groups, value_groups


# -----------------------
# 🔹 Cache Calculation
# -----------------------
def CalculateCurrentLayerCache(x, dim=128, groups=[1, 1, 2, 4]):
    lens = len(groups)
    ceil_dim = dim
    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f = torch.mean(qv_cache_f, dim=0, keepdim=True).detach()
        update_elements = F.interpolate(
            qv_cache_f.unsqueeze(1),
            size=(ceil_dim, ceil_dim),
            mode='bilinear',
            align_corners=False
        )
        c_i = qv_cache_f.shape[-1]

        if i == 0:
            qv_cache = update_elements * c_i // dim
        else:
            qv_cache = qv_cache + update_elements * c_i // dim

    return qv_cache.squeeze(1)


# -----------------------
# 🔹 Attention with Cache Modulation
# -----------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.group = [1, 2, 2, 3]
        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)
        self.inter_modulators = nn.ModuleList([
            Inter_CacheModulation(in_c=g * dim // 8) for g in self.group
        ])
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
        attScore, tmp_cache = [], []

        for index in range(len(self.group)):
            query_head = F.normalize(qu[index], dim=-1)
            key_head = F.normalize(ke[index], dim=-1)
            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index, :, :]
            attn = attn.softmax(dim=-1)
            attScore.append(attn)
            tmp_cache.append(query_head.detach() + key_head.detach())

        tmp_caches = torch.cat(tmp_cache, 1)

        # Inter Modulation
        out = []
        if qv_cache is not None and qv_cache.shape[-1] != c:
            qv_cache = F.adaptive_avg_pool2d(qv_cache, c)
        for i in range(len(self.group)):
            if qv_cache is not None:
                attScore[i] = self.inter_modulators[i](attScore[i], qv_cache) + attScore[i]
            out.append(attScore[i] @ va[i])

        update_factor = 0.9
        update_elements = CalculateCurrentLayerCache(attScore, c, self.group)
        qv_cache = (
            qv_cache * update_factor + update_elements * (1 - update_factor)
        ) if qv_cache is not None else update_elements * update_factor

        out_all = torch.cat(out, 1)
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all
        out_all = rearrange(out_all, 'b c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)

        return [out_all, qv_cache]


# -----------------------
# 🔹 Transformer Block
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias', isAtt=True):
        super().__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, 'mish', 0.1)

    def forward(self, inputs):
        x, qv_cache = inputs
        if self.isAtt:
            x_tmp = x
            x_att, qv_cache = self.attn(self.norm1(x), qv_cache=qv_cache)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))
        return [x, qv_cache]


# -----------------------
# 🔹 Downsample & Upsample
# -----------------------
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# -----------------------
# 🔹 Final HCAM Model
# -----------------------
class HCAM(nn.Module):
    def __init__(self,
                 inp_channels=9,          # ✅ changed from 3 → 9
                 out_channels=1,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 qv_cache=None):
        super().__init__()
        self.qv_cache = qv_cache

        # ✅ Now matches your 9-channel input dataset
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        # ---------------- Encoder ----------------
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[1])
        ])

        # ---------------- Decoder ----------------
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[0])
        ])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)

        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

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
            # ✅ added safer handling for grayscale output + skip
            out_dec_level1 = self.output(out_dec_level1)
            if inp_img.shape[1] >= 1:
                out_dec_level1 = out_dec_level1 + inp_img[:, :1, :, :]

        return out_dec_level1
