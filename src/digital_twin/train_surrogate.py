"""
Synthetic Calibration & Training Script for the Digital Twin PINN Surrogate.
Generates thousands of steady-state thermodynamic states using the first-principles engine,
trains the PINN with mass-conservation loss, and exports the calibrated surrogate.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from digital_twin.physics_engine import CrudeDistillationDigitalTwin
from digital_twin.pinn_surrogate import DistillationSurrogatePINN

def generate_training_data(n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    twin = CrudeDistillationDigitalTwin()
    print(f"[*] Simulating {n_samples} first-principles thermodynamic operating points...")

    X = []
    Y = []

    np.random.seed(42)
    for _ in range(n_samples):
        feed_flow = float(np.random.uniform(70000.0, 130000.0))    # 70k to 130k bpd
        feed_temp = float(np.random.uniform(330.0, 365.0))         # 330 to 365 °C
        reflux_ratio = float(np.random.uniform(1.8, 3.2))          # 1.8 to 3.2
        duty_mw = float(np.random.uniform(35.0, 60.0))             # 35 to 60 MW
        
        # Crude slate fractions [Naphtha, Kero, Diesel, Residue]
        c1 = float(np.random.uniform(0.18, 0.26))
        c2 = float(np.random.uniform(0.12, 0.18))
        c3 = float(np.random.uniform(0.24, 0.32))
        c4 = 1.0 - (c1 + c2 + c3)
        crude_comp = np.array([c1, c2, c3, c4], dtype=np.float32)

        res = twin.step(
            feed_flow_bpd=feed_flow,
            feed_temp_c=feed_temp,
            reflux_ratio=reflux_ratio,
            furnace_duty_mw=duty_mw,
            crude_composition=crude_comp
        )

        in_vec = [feed_flow, feed_temp, reflux_ratio, duty_mw, c1, c2, c3]
        out_vec = [
            res["top_temperature_c"],
            res["bottom_temperature_c"],
            res["delta_pressure_bar"],
            res["hydraulic_flooding_pct"],
            res["product_yields_bpd"]["naphtha_gasoline"],
            res["product_yields_bpd"]["kerosene_jet_fuel"],
            res["product_yields_bpd"]["diesel_gas_oil"],
            res["product_yields_bpd"]["heavy_atmospheric_residue"]
        ]

        X.append(in_vec)
        Y.append(out_vec)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_surrogate():
    X_raw, Y_raw = generate_training_data(1500)

    # Scaling normalization
    x_mean = np.mean(X_raw, axis=0)
    x_std = np.std(X_raw, axis=0) + 1e-6
    y_mean = np.mean(Y_raw, axis=0)
    y_std = np.std(Y_raw, axis=0) + 1e-6

    X_norm = (X_raw - x_mean) / x_std
    Y_norm = (Y_raw - y_mean) / y_std

    dataset = TensorDataset(torch.tensor(X_norm), torch.tensor(Y_norm), torch.tensor(X_raw), torch.tensor(Y_raw))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DistillationSurrogatePINN(in_features=7, out_features=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse_loss = nn.MSELoss()

    print("[*] Training Physics-Informed Neural Network (PINN)...")
    for epoch in range(1, 16):
        total_loss = 0.0
        for bx_norm, by_norm, bx_raw, by_raw in loader:
            optimizer.zero_grad()
            pred_norm = model(bx_norm)
            loss_data = mse_loss(pred_norm, by_norm)

            # Unnormalize predictions to evaluate mass conservation loss
            pred_unnorm = pred_norm * torch.tensor(y_std) + torch.tensor(y_mean)
            loss_physics = model.compute_physics_loss(bx_raw, pred_unnorm)

            loss = loss_data + 1e-6 * loss_physics
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/15] - Total Loss: {total_loss/len(loader):.5f}")

    # Save trained surrogate checkpoint
    ckpt_dir = Path("checkpoints/digital_twin")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist()
    }, ckpt_dir / "pinn_surrogate.pt")
    print(f"[+] Successfully saved Digital Twin PINN Surrogate to: {ckpt_dir / 'pinn_surrogate.pt'}")

if __name__ == "__main__":
    train_surrogate()
