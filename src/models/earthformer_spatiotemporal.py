"""
Phase 3 Earthformer / MetNet-3 Deep Spatiotemporal Transformer Model.
PyTorch Implementation of multi-sensor attention-based spatiotemporal neural network.
Fuses high-resolution radar, geostationary satellite, lightning, and NWP covariates to predict 0-6h convective evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CuboidSelfAttention(nn.Module):
    """Spatio-Temporal Cuboid Attention Block for learning convective dynamics."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
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
    """Transformer Encoder Block with Spatiotemporal Cuboid Attention and FeedForward Network."""

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
    Spatiotemporal Transformer Architecture for India Convective Storm & Lightning Nowcasting.
    Input:  (B, T_in=6, C_in=12, H=128, W=128) - Multi-Sensor History (Radar, INSAT, Lightning, NWP)
    Output: (B, T_out=36, C_out=2, H=128, W=128) - Forecast Reflectivity (dBZ) & Lightning Flash Density Probability
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

        # Spatial patch embedding stem (Downsamples spatial dimensions by 4x for efficiency)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Spatiotemporal Transformer Encoder
        self.encoder_blocks = nn.ModuleList([
            EarthformerBlock(embed_dim=embed_dim, num_heads=4) for _ in range(depth)
        ])

        # Temporal expansion projection (6 steps -> 36 forecast steps)
        self.temporal_expand = nn.Linear(history_steps, forecast_steps)

        # Spatial Decoder / Upsampler stem (Upsamples 4x back to full spatial resolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T_in, C_in, H, W)
        B, T_in, C_in, H, W = x.shape

        # Patch Embed per timestep
        x_reshaped = x.view(B * T_in, C_in, H, W)
        feat = self.patch_embed(x_reshaped) # (B*T_in, embed_dim, H/4, W/4)
        _, C_emb, H_sub, W_sub = feat.shape

        feat = feat.view(B, T_in, C_emb, H_sub, W_sub).permute(0, 1, 3, 4, 2) # (B, T_in, H_sub, W_sub, C_emb)

        # Pass through Transformer blocks
        for block in self.encoder_blocks:
            feat = block(feat)

        # Expand Temporal Dimension to forecast horizon: (B, T_in, H_sub, W_sub, C_emb) -> (B, T_out, H_sub, W_sub, C_emb)
        feat_t = feat.permute(0, 2, 3, 4, 1) # (B, H_sub, W_sub, C_emb, T_in)
        feat_t_expanded = self.temporal_expand(feat_t) # (B, H_sub, W_sub, C_emb, T_out)
        feat_expanded = feat_t_expanded.permute(0, 4, 3, 1, 2) # (B*T_out, C_emb, H_sub, W_sub)

        # Decode back to spatial grid
        out = self.decoder(feat_expanded) # (B*T_out, out_channels, H, W)
        _, C_out, H_out, W_out = out.shape

        out = out.view(B, self.forecast_steps, C_out, H_out, W_out)
        
        # Apply physical activations: Non-negative Reflectivity & Sigmoid Lightning Probability
        reflectivity = F.relu(out[:, :, 0:1, :, :])
        lightning_prob = torch.sigmoid(out[:, :, 1:2, :, :])

        return torch.cat([reflectivity, lightning_prob], dim=2)

if __name__ == "__main__":
    # Sanity test model forward pass
    model = EarthformerIndiaNowcaster(
        in_channels=12,
        out_channels=2,
        history_steps=6,
        forecast_steps=36,
        embed_dim=64,
        depth=2
    )
    dummy_input = torch.randn(2, 6, 12, 128, 128) # Batch=2, History=60min, Channels=12, 128x128 crop
    output = model(dummy_input)
    print("Earthformer Spatiotemporal Model Verification:")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch, Forecast_Steps=36, Out_Channels=2, H=128, W=128)")
