# bio_crowd_flow.py
# Bio-Inspired Transformer for Crowd Flow Prediction
# Combines CNN feature extractors (for Agent, Environment, Goal images)
# with biologically inspired attention + latent mixing + transformer blocks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

# ============================================================
# Utilities
# ============================================================

class LayerNorm2d(nn.Module):
    """
    Applies LayerNorm across channels for 2D feature maps.
    Useful for stabilizing transformer training on images.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ============================================================
# Bio-inspired attention blocks
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    Multi-head self-attention with group-wise queries.
    - Splits heads into groups.
    - Each group attends locally, then outputs are merged.
    """
    def __init__(self, dim, heads=8, groups=4, dropout=0.0):
        super().__init__()
        assert heads % groups == 0, "heads must be divisible by groups"
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.groups = groups
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        hpg = self.heads // self.groups
        outs = []
        for g in range(self.groups):
            qs = q[:, g*hpg:(g+1)*hpg]
            ks = k[:, g*hpg:(g+1)*hpg]
            vs = v[:, g*hpg:(g+1)*hpg]
            attn = torch.matmul(qs, ks.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            outs.append(torch.matmul(attn, vs))

        out = torch.cat(outs, dim=1).reshape(B, C, H, W)
        return self.drop(self.proj(out))


class LatentMixer(nn.Module):
    """
    Cross-attention between learnable latent tokens and input features.
    - Latents "read" from spatial features (compression).
    - Latents "write" back into features (expansion).
    """
    def __init__(self, dim, num_latents=16, heads=8, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, dim, 1, num_latents) * 0.02)

        self.q_lat = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_x  = nn.Conv2d(dim, dim*2, 1, bias=False)
        self.q_x   = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_lat= nn.Conv2d(dim, dim*2, 1, bias=False)

        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return torch.matmul(attn.softmax(dim=-1), v)

    def forward(self, x):
        B, C, H, W = x.shape
        L = self.num_latents
        lat = self.latents.expand(B, -1, 1, -1)

        # Latents read from features
        ql = self.q_lat(lat).reshape(B, self.heads, C // self.heads, L)
        kx, vx = torch.chunk(self.kv_x(x), 2, dim=1)
        kx = kx.reshape(B, self.heads, C // self.heads, H * W)
        vx = vx.reshape(B, self.heads, C // self.heads, H * W)
        lat_read = self._attn(ql, kx, vx).reshape(B, C, 1, L)

        # Features read from updated latents
        qx = self.q_x(x).reshape(B, self.heads, C // self.heads, H * W)
        kl, vl = torch.chunk(self.kv_lat(lat_read), 2, dim=1)
        kl = kl.reshape(B, self.heads, C // self.heads, L)
        vl = vl.reshape(B, self.heads, C // self.heads, L)
        x_write = self._attn(qx, kl, vl).reshape(B, C, H, W)

        return self.drop(self.proj(x_write))


class BioAttentionFusion(nn.Module):
    """
    Combines:
      - Local attention
      - Downsampled global attention
    Then fuses both streams.
    """
    def __init__(self, dim, heads=8, groups=4, dropout=0.0, saccade_stride=4):
        super().__init__()
        self.local = GroupedQueryAttention(dim, heads=heads, groups=groups, dropout=dropout)
        self.saccade_stride = saccade_stride
        self.global_attn = GroupedQueryAttention(dim, heads=heads, groups=groups, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        local = self.local(x)
        if self.saccade_stride > 1:
            xc = F.avg_pool2d(x, kernel_size=self.saccade_stride, stride=self.saccade_stride)
            g = self.global_attn(xc)
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            g = self.global_attn(x)
        return self.fuse(torch.cat([local, g], dim=1))


class FeedForward(nn.Module):
    """ Standard MLP block with expansion and GELU activation. """
    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class BioTransformerBlock(nn.Module):
    """
    Transformer block with:
      - BioAttentionFusion
      - LatentMixer
      - FeedForward MLP
    Uses residual connections + DropPath.
    """
    def __init__(self, dim, heads=8, groups=4, num_latents=16, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.n1 = LayerNorm2d(dim)
        self.attn = BioAttentionFusion(dim, heads=heads, groups=groups, dropout=dropout)
        self.n_lat = LayerNorm2d(dim)
        self.latent = LatentMixer(dim, num_latents=num_latents, heads=heads, dropout=dropout)
        self.n2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dp3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.dp1(self.attn(self.n1(x)))
        x = x + self.dp2(self.latent(self.n_lat(x)))
        x = x + self.dp3(self.ffn(self.n2(x)))
        return x


# ============================================================
# Encoders, Fusion, Decoder, Model
# ============================================================

class CNNEncoder(nn.Module):
    """ Light CNN feature extractor for each input (A, E, G). """
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class LateFusion(nn.Module):
    """
    Late fusion of (A, E, G):
      - Concatenate features
      - Project to common dim
      - LatentMixer for interaction
      - Residual normalization
    """
    def __init__(self, dim, heads=8, num_latents=8, dropout=0.0):
        super().__init__()
        self.project_in = nn.Conv2d(dim * 3, dim, 1)
        self.latent = LatentMixer(dim, num_latents=num_latents, heads=heads, dropout=dropout)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.gn = nn.GroupNorm(8, dim) if dim % 8 == 0 else nn.BatchNorm2d(dim)

    def forward(self, a, e, g):
        x = torch.cat([a, e, g], dim=1)
        x = self.project_in(x)
        lat = self.latent(x)
        x = x + lat
        x = self.gn(x)
        return self.proj_out(x)


class Decoder(nn.Module):
    """
    Stacked BioTransformerBlocks for decoding fused representation.
    - Optionally integrates light skip connections from agent/env encoders.
    """
    def __init__(self, dim, heads=8, groups=4, num_latents=8, depth=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            BioTransformerBlock(dim, heads=heads, groups=groups,
                                num_latents=num_latents, dropout=dropout,
                                drop_path=drop_path)
            for _ in range(depth)
        ])
        self.skip_e = nn.Conv2d(dim, dim, 1)
        self.skip_a = nn.Conv2d(dim, dim, 1)

    def forward(self, fused, e=None, a=None):
        x = fused
        for i, blk in enumerate(self.blocks):
            if e is not None and i % 2 == 0:
                x = x + self.skip_e(e)
            if a is not None and i % 3 == 0:
                x = x + self.skip_a(a)
            x = blk(x)
        return x


class BioCrowdFlowModel(nn.Module):
    """
    Full model:
      - CNN encoders for Agent (A), Environment (E), Goal (G)
      - LateFusion
      - BioTransformer Decoder
      - Output head for predicting target (Y)
    """
    def __init__(self, dim=64, heads=8, groups=4, num_latents=8, decoder_depth=4):
        super().__init__()
        self.AgentEncoder = CNNEncoder(in_ch=1, out_ch=dim)
        self.EnvEncoder   = CNNEncoder(in_ch=1, out_ch=dim)
        self.GoalEncoder  = CNNEncoder(in_ch=1, out_ch=dim)

        self.fuse = LateFusion(dim, heads=heads, num_latents=num_latents)
        self.decoder = Decoder(dim, heads=heads, groups=groups,
                               num_latents=num_latents, depth=decoder_depth)
        self.out_head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, A, E, G):
        a = self.AgentEncoder(A)
        e = self.EnvEncoder(E)
        g = self.GoalEncoder(G)
        fused = self.fuse(a, e, g)
        x = self.decoder(fused, e=e, a=a)
        out = self.out_head(x)
        return out


# ============================================================
# Quick sanity test
# ============================================================
# if __name__ == "__main__":
#     B, H, W = 2, 64, 64
#     A = torch.randn(B, 1, H, W)
#     E = torch.randn(B, 1, H, W)
#     G = torch.randn(B, 1, H, W)
#     model = BioCrowdFlowModel(dim=64, heads=8, groups=4, num_latents=8, decoder_depth=3)
#     out = model(A, E, G)
#     print("output shape:", out.shape)   # expected (B, 1, H, W)
