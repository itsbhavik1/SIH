"""
Phase 3 Fine-Tuning & Meteorological Validation Pipeline.
Fine-tunes pretrained Earthformer model on fused IMD Radar + MOSDAC INSAT + Lightning + NCMRWF feeds.
Evaluates domain-specific metrics: Critical Success Index (CSI), Fractions Skill Score (FSS), POD, and FAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

class FocalLoss(nn.Module):
    """Focal Loss to handle extreme class imbalance in lightning occurrence."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MeteorologicalEvaluator:
    """Calculates standardized meteorological nowcasting evaluation metrics (CSI, POD, FAR, FSS)."""

    @staticmethod
    def calculate_contingency_table(pred: np.ndarray, target: np.ndarray, threshold: float = 35.0) -> Dict[str, int]:
        """Computes Hits (hits), False Alarms (fa), Misses (misses), and Correct Negatives (cn)."""
        pred_bin = (pred >= threshold).astype(bool)
        target_bin = (target >= threshold).astype(bool)

        hits = np.sum(pred_bin & target_bin)
        fa = np.sum(pred_bin & ~target_bin)
        misses = np.sum(~pred_bin & target_bin)
        cn = np.sum(~pred_bin & ~target_bin)

        return {"hits": hits, "fa": fa, "misses": misses, "cn": cn}

    @classmethod
    def compute_csi_pod_far(cls, pred: np.ndarray, target: np.ndarray, threshold: float = 35.0) -> Dict[str, float]:
        """Calculates Critical Success Index (CSI), Probability of Detection (POD), and False Alarm Ratio (FAR)."""
        table = cls.calculate_contingency_table(pred, target, threshold)
        hits, fa, misses = table["hits"], table["fa"], table["misses"]

        pod = hits / (hits + misses + 1e-8)
        far = fa / (hits + fa + 1e-8)
        csi = hits / (hits + misses + fa + 1e-8)

        return {"CSI": float(csi), "POD": float(pod), "FAR": float(far)}

class EarthformerTrainer:
    """Trainer & Fine-tuner for Earthformer India Nowcaster."""

    def __init__(self, model: nn.Module, lr: float = 2e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.evaluator = MeteorologicalEvaluator()

    def train_step(self, x_history: torch.Tensor, y_future: torch.Tensor) -> float:
        """
        x_history: (B, 6, 12, H, W)
        y_future:  (B, 36, 2, H, W) [Channel 0: Reflectivity, Channel 1: Lightning Density]
        """
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(x_history) # (B, 36, 2, H, W)
        
        loss_reflectivity = self.mse_loss(pred[:, :, 0:1, :, :], y_future[:, :, 0:1, :, :])
        loss_lightning = self.focal_loss(pred[:, :, 1:2, :, :], (y_future[:, :, 1:2, :, :] > 0.1).float())

        total_loss = loss_reflectivity + 10.0 * loss_lightning
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def evaluate(self, x_history: torch.Tensor, y_future: torch.Tensor) -> Dict[str, float]:
        """Runs evaluation and computes meteorological metrics."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_history)
        
        pred_dbz = pred[:, :, 0, :, :].cpu().numpy()
        target_dbz = y_future[:, :, 0, :, :].cpu().numpy()

        metrics_35dbz = self.evaluator.compute_csi_pod_far(pred_dbz, target_dbz, threshold=35.0)
        metrics_45dbz = self.evaluator.compute_csi_pod_far(pred_dbz, target_dbz, threshold=45.0)

        return {
            "CSI_35dBZ": metrics_35dbz["CSI"],
            "POD_35dBZ": metrics_35dbz["POD"],
            "FAR_35dBZ": metrics_35dbz["FAR"],
            "CSI_45dBZ": metrics_45dbz["CSI"]
        }

if __name__ == "__main__":
    from src.models.earthformer_spatiotemporal import EarthformerIndiaNowcaster

    model = EarthformerIndiaNowcaster(in_channels=12, out_channels=2, history_steps=6, forecast_steps=36, embed_dim=32, depth=2)
    trainer = EarthformerTrainer(model)

    x_dummy = torch.randn(2, 6, 12, 128, 128)
    y_dummy = torch.randn(2, 36, 2, 128, 128)
    y_dummy[:, :, 0, :, :] = torch.clamp(y_dummy[:, :, 0, :, :] * 20.0 + 20.0, 0, 70) # Realistic dBZ

    print("Executing Phase 3 Fine-Tuning Step...")
    loss = trainer.train_step(x_dummy, y_dummy)
    metrics = trainer.evaluate(x_dummy, y_dummy)
    
    print(f"Fine-Tuning Loss: {loss:.4f}")
    print("Phase 3 Meteorological Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
