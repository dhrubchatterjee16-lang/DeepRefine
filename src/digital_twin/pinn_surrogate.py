"""
Physics-Informed Neural Network (PINN) Surrogate Model for Digital Twin.
A high-speed neural surrogate trained on the first-principles thermodynamic engine.
Predicts product yields, hydraulic flooding risk, and energy consumption in microseconds,
enabling real-time what-if simulations and plant-wide optimization.

Inputs: [feed_flow_bpd, feed_temp_c, reflux_ratio, furnace_duty_mw, crude_naphtha_pct, crude_kero_pct, crude_diesel_pct]
Outputs: [top_temp_c, bot_temp_c, delta_p_bar, flooding_pct, naphtha_bpd, kero_bpd, diesel_bpd, residue_bpd]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class DistillationSurrogatePINN(nn.Module):
    def __init__(self, in_features: int = 7, out_features: int = 8):
        super().__init__()
        
        # Deep Residual MLP Architecture
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(), # Smooth non-linear activation for physics surrogates
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)

    def compute_physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Physics-Informed Regularization (True Mass Conservation per VDI 2048 / AIChE standards):
        Enforces Mass_in (metric tons/hr) == Mass_out (naphtha + kero + diesel + residue).
        
        Refinery cuts exhibit volume non-additivity and varying specific gravities (SG):
        - Feed crude: SG ~ 0.850 (135.1 kg/bbl)
        - Naphtha:    SG ~ 0.730 (116.1 kg/bbl)
        - Kerosene:   SG ~ 0.805 (128.0 kg/bbl)
        - Diesel:     SG ~ 0.845 (134.3 kg/bbl)
        - Residue:    SG ~ 0.945 (150.2 kg/bbl)
        1 bbl = 0.158987 m3. Conversion to metric tons/hour (tph): flow_bpd * SG * (0.158987 / 24)
        """
        bbl_to_tph = 0.158987 / 24.0
        feed_sg = 0.850
        cut_sgs = torch.tensor([0.730, 0.805, 0.845, 0.945], device=predictions.device, dtype=predictions.dtype)

        feed_mass_tph = inputs[:, 0] * feed_sg * bbl_to_tph
        cut_flows_bpd = predictions[:, 4:8]
        cut_masses_tph = cut_flows_bpd * cut_sgs * bbl_to_tph
        total_mass_out_tph = torch.sum(cut_masses_tph, dim=-1)

        # Normalized mass balance residual: ((Mass_in - Mass_out) / Mass_in)^2
        normalized_residual = (feed_mass_tph - total_mass_out_tph) / (feed_mass_tph + 1e-5)
        return torch.mean(normalized_residual ** 2)

