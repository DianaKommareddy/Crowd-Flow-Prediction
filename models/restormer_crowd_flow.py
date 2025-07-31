import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=3, stride=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        x = self.norm(x)
        return x, H, W


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.0):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x_res = x
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.attn(x)
        x = x + x_res

        x_res = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn(x)
        x = x + x_res
        return x


class RestormerCrowdFlow(nn.Module):
    def __init__(self, input_channels=9, embed_dim=48, num_blocks=4, num_heads=4):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(input_channels, embed_dim)
        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)  # output: 1-channel heatmap

    def forward(self, x):
        # x shape: [B, 9, H, W]
        x_patch, H, W = self.patch_embed(x)  # [B, H*W, C]
        x_patch = x_patch.transpose(1, 2).reshape(x.shape[0], -1, H, W)  # [B, C, H, W]
        x_enc = self.encoder(x_patch)  # Transformer encoding
        out = self.head(x_enc)  # [B, 1, H, W] heatmap
        return out
