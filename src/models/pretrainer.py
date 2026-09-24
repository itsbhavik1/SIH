"""
Phase 3 Self-Supervised Pretraining Module on BharatBench Benchmark Dataset.
Applies Masked Spatiotemporal Autoencoding (MAE) to pretrain Earthformer Transformer backbones.
Reduces data scarcity bottleneck before fine-tuning on high-resolution IMD radar and ILLN feeds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple

class MaskedSpatiotemporalPretrainer:
    """Pretrainer using Masked Autoencoder (MAE) strategy on BharatBench reanalysis fields."""

    def __init__(self, model: nn.Module, lr: float = 1e-4, mask_ratio: float = 0.5):
        self.model = model
        self.mask_ratio = mask_ratio
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.criterion = nn.MSELoss()

    def generate_random_patch_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generates 3D spatiotemporal binary mask with specified masking ratio."""
        B, T, C, H, W = x.shape
        random_matrix = torch.rand(B, T, 1, H, W, device=x.device)
        mask = random_matrix < self.mask_ratio
        return mask

    def pretrain_epoch(self, dataloader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
        """Executes one full epoch of self-supervised MAE pretraining."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (batch_x,) in enumerate(dataloader):
            batch_x = batch_x.to(device) # (B, T=42, C=12, H=128, W=128)
            mask = self.generate_random_patch_mask(batch_x)

            masked_input = batch_x.clone()
            masked_input[mask.expand_as(batch_x)] = 0.0

            self.optimizer.zero_grad()
            # Predict 36 steps using first 6 history steps
            pred = self.model(masked_input[:, :6, :, :, :])

            # Loss computed on reconstruction of first channel (reflectivity proxy)
            target = batch_x[:, :36, 0:1, :, :]
            loss = self.criterion(pred[:, :, 0:1, :, :], target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.earthformer_spatiotemporal import EarthformerIndiaNowcaster

    dummy_bharatbench = torch.randn(8, 42, 12, 128, 128)
    loader = DataLoader(TensorDataset(dummy_bharatbench), batch_size=2, shuffle=True)

    model = EarthformerIndiaNowcaster(in_channels=12, out_channels=2, history_steps=6, forecast_steps=36, embed_dim=32, depth=2)
    pretrainer = MaskedSpatiotemporalPretrainer(model, lr=1e-4)

    print("Executing Self-Supervised Pretraining Epoch on BharatBench...")
    loss = pretrainer.pretrain_epoch(loader, device="cpu")
    print(f"Pretraining Epoch Completed — Reconstruction Loss: {loss:.6f}")
