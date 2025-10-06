# hint_with_mcrpb.py
# Complete model code with Modal-Conditioned Relative Positional Bias (MCRPB)
# Requires: torch, einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Optional: for DropPath regularization
try:
    from timm.models.layers import DropPath
except ImportError:
    DropPath = nn.Identity

# ----------------------------
# Helpers & small convs
# ----------------------------
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

# LayerNorm updated to 4D input (keeps your behaviour)
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

# ----------------------------
# Modal-Conditioned Relative Positional Bias (MCRPB)
# ----------------------------
class ModalConditionedRPE(nn.Module):
    """
    Modal-Conditioned Relative Positional Bias (MCRPB).
    - Computes a learned bias for each pair of positions (i,j) using:
        [relative_coords, optional modality pair embedding, local structure values at i and j]
      passed through a small MLP to get a scalar bias.
    - Warning: computes an (H*W x H*W) bias matrix per forward call. This matches attention's complexity.
      If H*W is large, switch to windowed/local computation or caching.
    - Usage:
        bias = mcrpb(h, w, device, modality_pair=(q_mod,k_mod), structure_map=struct_map)
        # returns tensor of shape (1,1,N,N) where N = h*w
    """
    def __init__(self, mod_embed_dim=16, hidden_dim=64, num_modalities=3, radius=None):
        """
        mod_embed_dim : int - size of modality embedding vector
        hidden_dim : int - hidden layer size for MLP
        num_modalities : int - number of modalities (A,E,G) expected; if you don't pass modality ids, embeddings ignored
        radius : int or None - if set, only produce bias within +/- radius offsets (saves memory). Not implemented fully here.
        """
        super().__init__()
        self.mod_embed_dim = mod_embed_dim
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.radius = radius  # placeholder for future local-window optimization

        # modality embedding for each modality id (0..num_modalities-1)
        self.mod_embed = nn.Embedding(num_modalities, mod_embed_dim)

        # small MLP: input dims: rel_xy(2) + mod_concat(2*mod_embed_dim if pair given else 0) + structure_pair(2)
        # To keep it flexible, accept variable input length; we'll build MLP for max possible length and slice inputs.
        mlp_input_dim = 2 + 2 * mod_embed_dim + 2  # (dx,dy) + (q_mod_emb,k_mod_emb) + (s_i, s_j)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Cache for coordinate grids to avoid re-alloc every forward (keyed by (h,w,device))
        self._coord_cache = {}

    def _get_coords(self, h, w, device):
        key = (h, w, device)
        if key in self._coord_cache:
            return self._coord_cache[key]
        # normalized coordinates in [-0.5, 0.5] range
        ys = torch.linspace(-0.5, 0.5, steps=h, device=device)
        xs = torch.linspace(-0.5, 0.5, steps=w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # shape h,w
        coords = torch.stack([xx, yy], dim=-1)  # (h, w, 2)
        coords_flat = coords.view(-1, 2)  # (N,2)
        self._coord_cache[key] = coords_flat
        return coords_flat  # (N,2)

    def forward(self, h, w, device,
                modality_pair=None,
                structure_map=None):
        """
        Returns bias tensor shaped (1,1,N,N) where N = h*w.
        modality_pair: None or tuple (q_mod:int, k_mod:int) indicating modality ids (0..num_modalities-1)
        structure_map: optional tensor (b,1,H,W) or (1,1,H,W) with local structure (edge strength). If None, we use zeros.
        """
        coords = self._get_coords(h, w, device)          # (N,2)
        N = coords.shape[0]

        # Pairwise relative coords (N,N,2) -> (N*N,2)
        # WARNING: this allocation is O(N^2) memory.
        rel = coords.unsqueeze(1) - coords.unsqueeze(0)   # (N,N,2)
        rel_flat = rel.view(-1, 2)                        # (N*N,2)

        # Modality embedding pair
        if modality_pair is not None:
            q_mod, k_mod = modality_pair
            # embed both and concatenate (same for all pairs)
            q_emb = self.mod_embed(torch.tensor(q_mod, device=device)).unsqueeze(0)  # (1,mod_dim)
            k_emb = self.mod_embed(torch.tensor(k_mod, device=device)).unsqueeze(0)  # (1,mod_dim)
            # expand to (N*N, 2*mod_dim)
            mod_pair = torch.cat([q_emb.repeat(N*N,1), k_emb.repeat(N*N,1)], dim=1)
        else:
            # zeros
            mod_pair = torch.zeros((rel_flat.shape[0], 2*self.mod_embed_dim), device=device)

        # Structure pair values
        # structure_map expected shape (b,1,h,w) or (1,1,h,w); we collapse to per-position scalar by averaging channels if needed.
        if structure_map is None:
            s_i = torch.zeros((N,1), device=device)
            s_j = torch.zeros((N,1), device=device)
        else:
            # ensure shape (1,1,h,w)
            if structure_map.dim() == 4:
                s = structure_map
            elif structure_map.dim() == 3:
                s = structure_map.unsqueeze(0)
            else:
                raise ValueError("structure_map must be shape (b,1,h,w) or (1,1,h,w)")
            # resize to h,w
            s_resized = F.interpolate(s, size=(h,w), mode='bilinear', align_corners=False)  # (b,1,h,w)
            s_flat = s_resized.view(-1, 1, h*w)[:, 0, :]  # (b, N) for batch b
            # we will average across batch if b>1 to get a single structure per position
            s_mean = s_flat.mean(dim=0, keepdim=True).t()  # (N,1)
            s_i = s_mean
            s_j = s_mean  # we use same local value for both query and key positions; more complex variants can use pairwise values

        # Build MLP input: [rel_flat (2), mod_pair (2*mod_dim), s_i repeated & s_j repeated] -> (N*N, mlp_input_dim)
        # We need per-pair s_i and s_j: replicate s_i for each key repeated N times, and tile s_j across queries:
        s_i_rep = s_i.repeat_interleave(N, dim=0)  # (N*N,1)  query position repeated across all keys
        s_j_rep = s_j.repeat(N, 1)  # (N*N,1)  keys tiled for each query

        mlp_in = torch.cat([rel_flat, mod_pair, s_i_rep, s_j_rep], dim=1)  # (N*N, mlp_input_dim)

        bias_flat = self.mlp(mlp_in)  # (N*N,1)
        bias = bias_flat.view(1, 1, N, N)  # (1,1,N,N)

        return bias


# ----------------------------
# Inter / Intra Cache Modulators (unchanged)
# ----------------------------
class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()
        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2*in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)
    def forward(self, x1,x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2)+x1)
        scale,shift = x2_pW.chunk(2, dim=1)
        x1_p = x1*scale+shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p

class Intra_CacheModulation(nn.Module):
    def __init__(self,embed_dim=48):
        super(Intra_CacheModulation, self).__init__()
        self.down = nn.Conv1d(embed_dim, embed_dim//2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim//2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
    def forward(self, x1,x2):
        x_gated = F.gelu(self.gatingConv(x2+x1)) * (x2+x1)
        x_p = self.up(self.down(x_gated))
        return x_p

# ----------------------------
# ReGroup and cache utilities (unchanged)
# ----------------------------
class ReGroup(nn.Module):
    def __init__(self, groups=[1,1,2,4]):
        super(ReGroup, self).__init__()
        self.gourps = groups
    def forward(self, query,key,value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)
        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True)
        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]
        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.gourps)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.gourps]
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx
        return query_groups,key_groups,value_groups

def CalculateCurrentLayerCache(x,dim=128,groups=[1,1,2,4]):
    lens = len(groups)
    ceil_dim = dim
    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f = torch.mean(qv_cache_f,dim=0,keepdim=True).detach()
        update_elements = F.interpolate(qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim), mode='bilinear', align_corners=False)
        c_i = qv_cache_f.shape[-1]

        if i==0:
            qv_cache = update_elements * c_i // dim
        else:
            qv_cache = qv_cache + update_elements * c_i // dim

    return qv_cache.squeeze(1)

# ----------------------------
# Attention with integrated MCRPB
# ----------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_modalities=3):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.group = [1,2,2,3]
        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)
        self.inter_modulator1 = Inter_CacheModulation(in_c=1*dim//8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3*dim//8)
        self.inter_modulators = [self.inter_modulator1,self.inter_modulator2,self.inter_modulator3,self.inter_modulator4]
        self.regroup = ReGroup(self.group)
        self.dim = dim

        # Modal-Conditioned RPE
        self.mcrpb = ModalConditionedRPE(mod_embed_dim=16, hidden_dim=64, num_modalities=num_modalities)

    def forward(self, x, qv_cache=None, modality_pair=None, structure_map=None):
        """
        x: (b, c, h, w)
        modality_pair: optional tuple (q_mod, k_mod) ints indicating modality ids (0..num_modalities-1)
            If not provided, MCRPB will operate without explicit modality conditioning.
        structure_map: optional (b,1,h,w) structural cue tensor (e.g., edge magnitude). If None, internal zero used.
        """
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        qu, ke, va = self.regroup(q, k, v)

        attScore = []
        tmp_cache = []
        for index in range(len(self.group)):
            query_head = qu[index]
            key_head = ke[index]
            query_head = torch.nn.functional.normalize(query_head, dim=-1)
            key_head = torch.nn.functional.normalize(key_head, dim=-1)
            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index,:,:]
            # -----------------------
            # Add ModalConditionedRPE bias here
            # -----------------------
            # Compute bias shaped (1,1,N,N) for this spatial size
            # If structure_map provided, resize to (b,1,h,w) and pass (we'll take mean across batch inside MCRPB)
            bias = self.mcrpb(h, w, x.device, modality_pair=modality_pair, structure_map=structure_map)
            # attn shape: (b, group_c, N, N) but bias is (1,1,N,N) -> broadcast across batch & channel dims
            # We need attn to be float before adding
            # ensure shapes align: expand bias to (b, group_c, N, N) by broadcasting
            # Here attn shape: (b, head_c, N, N). We'll add bias to each channel equally.
            attn = attn + bias
            attn = attn.softmax(dim=-1)
            attScore.append(attn)
            t_cache = query_head.clone().detach() + key_head.clone().detach()
            tmp_cache.append(t_cache)

        tmp_caches = torch.cat(tmp_cache, 1)

        # Inter Modulation
        out = []
        if qv_cache is not None:
            if qv_cache.shape[-1] != c:
                qv_cache = F.adaptive_avg_pool2d(qv_cache, c)
        for i in range(4):
            if qv_cache is not None:
                inter_modulator = self.inter_modulators[i]
                # Note: attScore[i] shape (b, head_c, N, N) and va[i] shape (b, head_c, N)
                # Compute att @ v as in original code: (b, head_c, N, N) @ (b, head_c, N, d) but existing code used matrix multiply with broadcasting
                # Here we follow original pattern:
                out_i = attScore[i] @ va[i]
                out_i = inter_modulator(attScore[i], qv_cache) + out_i
                out.append(out_i)
            else:
                out.append(attScore[i] @ va[i])

        update_factor = 0.9
        if qv_cache is not None:
            update_elements = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor + update_elements * (1 - update_factor)
        else:
            qv_cache = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor

        out_all = torch.concat(out, 1)
        # Intra Modulation
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all
        out_all = rearrange(out_all, 'b  c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)
        return [out_all, qv_cache]

# ----------------------------
# Transformer Block with optional modality/structure pass-through
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias', isAtt=True):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, 'mish', 0.1)

    def forward(self, inputs, modality_pair=None, structure_map=None):
        """
        inputs: [x, qv_cache]
        modality_pair: optional tuple passed to Attention for MCRPB
        structure_map: optional structural map (b,1,h,w) to inform MCRPB
        """
        x = inputs[0]
        qv_cache = inputs[1]
        if self.isAtt:
            x_tmp = x
            [x_att, qv_cache] = self.attn(self.norm1(x), qv_cache=qv_cache,
                                          modality_pair=modality_pair,
                                          structure_map=structure_map)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))
        return [x, qv_cache]

# ----------------------------
# Downsample & Upsample (unchanged)
# ----------------------------
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

# ----------------------------
# Full HINT model with MCRPB-enabled Attention
# ----------------------------
class HINT(nn.Module):
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
                 num_modalities=3):
        """
        num_modalities: how many modality ids you'll use later (default 3 => A,E,G).
        """
        super(HINT, self).__init__()
        self.qv_cache = qv_cache

        # Patch Embedding
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        # Encoder Levels
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=False)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(dim*4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim*8, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[1])
        ])

        # Decoder Levels
        self.up4_3 = Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim*2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type, isAtt=True)
            for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim*2, 1, bias=bias)

        self.output = nn.Conv2d(dim*2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img, modality_pair=None, structure_map=None):
        """
        inp_img: (b, inp_channels, H, W)
        modality_pair: optional tuple or list passed into TransformerBlocks where attention exists.
            - If you later use separate A/E/G streams you can pass modality ids for the current fusion stage,
              e.g., modality_pair=(0,1) to indicate queries come from modality 0 (Agent) and keys from modality 1 (Env).
            - For now, model works without passing modality_pair (defaults to None).
        structure_map: optional (b,1,H,W) edge/structure map that MCRPB will use; if None MCRPB uses zeros.
        """
        inp_enc_level1 = self.patch_embed(inp_img)
        # encoder_level1 blocks do not use attention in your original config (isAtt=False for them)
        out_enc_level1, self.qv_cache = self.encoder_level1([inp_enc_level1, self.qv_cache])

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, self.qv_cache = self.encoder_level2([inp_enc_level2, self.qv_cache])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, self.qv_cache = self.encoder_level3([inp_enc_level3, self.qv_cache])

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, self.qv_cache = self.latent([inp_enc_level4, self.qv_cache])

        # For decoder attention blocks we forward optional modality_pair and structure_map
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # decoder blocks used with attention (isAtt=True)
        # Pass modality_pair & structure_map here if available (they default to None)
        out_dec_level3, self.qv_cache = self._run_seq_with_mod(self.decoder_level3, [inp_dec_level3, self.qv_cache], modality_pair, structure_map)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, self.qv_cache = self._run_seq_with_mod(self.decoder_level2, [inp_dec_level2, self.qv_cache], modality_pair, structure_map)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1, self.qv_cache = self._run_seq_with_mod(self.decoder_level1, [inp_dec_level1, self.qv_cache], modality_pair, structure_map)

        out_dec_level1, self.qv_cache = self._run_seq_with_mod(self.refinement, [out_dec_level1, self.qv_cache], modality_pair, structure_map)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :1, :, :]

        return out_dec_level1

    def _run_seq_with_mod(self, seq_module, inputs, modality_pair, structure_map):
        """
        Helper to run a Sequential of TransformerBlock elements while passing modality_pair and structure_map
        to blocks that accept them.
        inputs: [x, qv_cache]
        """
        x, qv_cache = inputs
        for block in seq_module:
            # call block.forward(inputs, modality_pair, structure_map)
            x, qv_cache = block([x, qv_cache], modality_pair=modality_pair, structure_map=structure_map)
        return [x, qv_cache]

# ----------------------------
# END of model
# ----------------------------
