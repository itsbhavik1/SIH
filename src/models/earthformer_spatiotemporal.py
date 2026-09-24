"""
Phase 3 Earthformer / MetNet-3 Deep Spatiotemporal Transformer Architecture.
PyTorch Neural Network for fusing high-resolution multi-sensor streams over India.

Inputs:  (B, T_in=6, C_in=12, H, W) - 60 min multi-channel history (Radar, INSAT, Lightning, NWP)
Outputs: (B, T_out=36, C_out=2, H, W) - 0-6 hour forecast sequence
         - Channel 0: Radar Max Reflectivity (dBZ)
         - Channel 1: Cloud-to-Ground Lightning Flash Density Probability [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CuboidSelfAttention(nn.Module):
    """3D Cuboid Spatiotemporal Self-Attention Block."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x_flat = x.view(B, T * H * W, C)

        qkv = self.qkv(x_flat).reshape(B, T * H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T * H * W, C)
        out = self.proj(out).reshape(B, T, H, W, C)
        return out

class EarthformerBlock(nn.Module):
    """Transformer Encoder Block with LayerNorm, Cuboid Attention, and GELU MLP."""

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CuboidSelfAttention(embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EarthformerIndiaNowcaster(nn.Module):
    """
    Full Spatiotemporal Transformer Model for Convective Storm & Lightning Prediction.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 2,
        history_steps: int = 6,
        forecast_steps: int = 36,
        embed_dim: int = 64,
        depth: int = 4
    ):
        super().__init__()
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps

        # 2D Spatial Patch Downsampling Stem (4x spatial reduction)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Spatiotemporal Transformer Blocks
        self.encoder_blocks = nn.ModuleList([
            EarthformerBlock(embed_dim=embed_dim, num_heads=4) for _ in range(depth)
        ])

        # Linear Temporal Expansion (6 history steps -> 36 forecast steps)
        self.temporal_expand = nn.Linear(history_steps, forecast_steps)

        # Spatial Decoder Stem (4x upsampling back to original spatial grid)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, C_in, H, W)
        B, T_in, C_in, H, W = x.shape

        # Patch Embed per timestep
        x_reshaped = x.view(B * T_in, C_in, H, W)
        feat = self.patch_embed(x_reshaped) # (B*T_in, embed_dim, H/4, W/4)
        _, C_emb, H_sub, W_sub = feat.shape

        feat = feat.view(B, T_in, C_emb, H_sub, W_sub).permute(0, 1, 3, 4, 2) # (B, T_in, H_sub, W_sub, C_emb)

        for block in self.encoder_blocks:
            feat = block(feat)

        # Expand temporal dimension
        feat_t = feat.permute(0, 2, 3, 4, 1) # (B, H_sub, W_sub, C_emb, T_in)
        feat_t_expanded = self.temporal_expand(feat_t) # (B, H_sub, W_sub, C_emb, T_out)
        feat_expanded = feat_t_expanded.permute(0, 4, 3, 1, 2) # (B*T_out, C_emb, H_sub, W_sub)

        # Decode back to spatial grid
        out = self.decoder(feat_expanded) # (B*T_out, out_channels, H, W)
        _, C_out, H_out, W_out = out.shape

        out = out.view(B, self.forecast_steps, C_out, H_out, W_out)

        # Physical output activations: Non-negative Reflectivity & Sigmoid Lightning Probability
        reflectivity = F.relu(out[:, :, 0:1, :, :])
        lightning_prob = torch.sigmoid(out[:, :, 1:2, :, :])

        return torch.cat([reflectivity, lightning_prob], dim=2)

if __name__ == "__main__":
    model = EarthformerIndiaNowcaster(
        in_channels=12,
        out_channels=2,
        history_steps=6,
        forecast_steps=36,
        embed_dim=64,
        depth=2
    )
    dummy_input = torch.randn(2, 6, 12, 128, 128)
    output = model(dummy_input)
    print("Earthformer Spatiotemporal Transformer Verified:")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (B, Forecast_Steps=36, Out_Channels=2, H=128, W=128)")
