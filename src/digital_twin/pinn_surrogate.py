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
        Physics-informed regularization:
        Enforces Overall Mass Conservation:
        Mass_in (feed_flow_bpd) == Mass_out (naphtha + kero + diesel + residue)
        """
        feed_flow = inputs[:, 0] # feed_flow_bpd
        predicted_total_yield = torch.sum(predictions[:, 4:8], dim=-1) # Sum of 4 cuts
        
        # Penalize violation of mass conservation: (feed_in - sum_yields)^2
        mass_balance_residual = torch.mean((feed_flow - predicted_total_yield) ** 2)
        return mass_balance_residual
