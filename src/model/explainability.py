"""
Feature Attribution & Root-Cause Analysis for Refinery PM.
Calculates sensor-level contribution to failure probability using gradient-based attribution
and feature deviation from the calibrated baseline.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

from model.lstm_pm import RefineryPredictiveMaintenanceLSTM

class RefineryExplainabilityEngine:
    def __init__(self, ckpt_path: Path = Path("checkpoints/best_refinery_pm.pt"),
                 scaler_path: Path = Path("checkpoints/feature_scaler.json")):
        self.device = torch.device("cpu")
        self.model = None
        self.scaler = None
        self.feature_names = []

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            n_features = checkpoint.get("n_features", 48)
            self.model = RefineryPredictiveMaintenanceLSTM(
                n_features=n_features, hidden_dim=128, num_layers=2, dropout=0.0
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

        if scaler_path.exists():
            with open(scaler_path, "r") as f:
                self.scaler = json.load(f)
                self.feature_names = self.scaler.get("columns", [])

    def explain(self, window_data: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Calculates attribution scores for an input window of shape [seq_len, n_features].
        Uses Input * Gradient attribution to identify the top driving sensors.
        """
        if self.model is None or len(self.feature_names) == 0:
            return []

        x_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
        x_tensor.requires_grad_(True)
        prob_logits, rul_pred = self.model(x_tensor)
        
        # Backward pass on logits
        prob_logits.backward()
        
        # Attribution = (grad * input) magnitude averaged across time sequence
        grads = x_tensor.grad.detach().squeeze(0).numpy() # [seq_len, n_features]
        inputs = x_tensor.detach().squeeze(0).numpy()
        attributions = np.mean(np.abs(grads * inputs), axis=0) # [n_features]

        # Group attribution by base sensor tag
        base_sensor_scores = {}
        for feat_name, score in zip(self.feature_names, attributions):
            base_tag = feat_name.split("_")[0] + "_" + feat_name.split("_")[1] if "_" in feat_name else feat_name
            base_sensor_scores[base_tag] = base_sensor_scores.get(base_tag, 0.0) + float(score)

        # Normalize to percentages
        total_score = sum(base_sensor_scores.values()) or 1.0
        sorted_sensors = sorted(
            [{"sensor": tag, "contribution_pct": round((score / total_score) * 100, 1)}
             for tag, score in base_sensor_scores.items()],
            key=lambda item: item["contribution_pct"],
            reverse=True
        )

        return sorted_sensors[:top_k]
