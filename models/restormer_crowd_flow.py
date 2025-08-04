import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Choose your activation function here
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
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, activation="mish"):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            get_activation(activation),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).reshape(B, C, H, W)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, activation="mish"):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, activation=activation)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, activation="mish"):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, 3, 2, 1),
            get_activation(activation)
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim=None, activation="mish"):
        super().__init__()
        out_dim = out_dim or in_dim // 2
        self.up = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            get_activation(activation)
        )

    def forward(self, x):
        return self.up(x)


class SharpRestormer(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, dim=64, heads=8, activation="mish"):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.encoder1 = nn.Sequential(*[TransformerBlock(dim, heads, activation) for _ in range(2)])
        self.down1 = Downsample(dim, activation)

        self.encoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads, activation) for _ in range(2)])
        self.down2 = Downsample(dim * 2, activation)

        self.latent = nn.Sequential(*[TransformerBlock(dim * 4, heads, activation) for _ in range(4)])

        self.up2 = Upsample(dim * 4, dim * 2, activation)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads, activation) for _ in range(2)])

        self.up1 = Upsample(dim * 2, dim, activation)
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim, heads, activation) for _ in range(2)])

        self.out_head = nn.Sequential(
            nn.Conv2d(dim, out_channels, 1),
            nn.Sigmoid()  # Clamp between 0 and 1
        )

    def forward(self, x):
        # Encoding
        x1 = self.embedding(x)
        x1_res = self.encoder1(x1)

        x2 = self.down1(x1_res)
        x2_res = self.encoder2(x2)

        x3 = self.down2(x2_res)
        x_latent = self.latent(x3)

        # Decoding with skip connections
        x2_up = self.up2(x_latent)
        x2_dec = self.decoder2(x2_up + x2_res)

        x1_up = self.up1(x2_dec)
        x1_dec = self.decoder1(x1_up + x1_res)

        out = self.out_head(x1_dec)
        return out
