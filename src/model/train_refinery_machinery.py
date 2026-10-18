"""
Continuous Training Script for Refinery Rotating Machinery Multi-Task Bi-LSTM.

Replaces NASA aviation model with genuine API 610/617 rotating machinery telemetry:
- Input features: Proximity vibration (X/Y), 1X/2X harmonics, bearing temp, compressor surge %,
  pump NPSH margin, seal Plan 53A dP, heat exchanger fouling Rf.
- Multi-Task Target 1: 5-day binary mechanical failure classification.
- Multi-Task Target 2: Continuous Remaining Useful Life (RUL in hours).
- Export: Serialized PyTorch and TorchScript edge binary for SCADA runtime.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.lstm_pm import RefineryPredictiveMaintenanceLSTM

class RotatingMachineryDataset(Dataset):
    def __init__(self, X: np.ndarray, y_label: np.ndarray, y_rul: np.ndarray, seq_len: int = 30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_label = torch.tensor(y_label, dtype=torch.float32)
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        label = self.y_label[idx + self.seq_len]
        rul = self.y_rul[idx + self.seq_len]
        return x_seq, label, rul

def train_refinery_machinery_lstm(
    data_path: Path = Path("data/raw/machinery_api670/refinery_rotating_machinery.parquet"),
    ckpt_dir: Path = Path("checkpoints/machinery"),
    epochs: int = 6,
    batch_size: int = 64,
    seq_len: int = 30
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path)

    feature_cols = [
        "probe_x_vibe_um", "probe_y_vibe_um", "harmonic_1x_pct", "harmonic_2x_pct",
        "subsync_pct", "bearing_babbitt_temp_c", "compressor_surge_margin_pct",
        "pump_npsh_margin_m", "seal_plan53a_dp_bar", "exchanger_fouling_rf"
    ]

    X_raw = df[feature_cols].values
    y_label = df["failure_label"].values
    y_rul = df["rul_hours"].values

    # Normalization statistics
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0) + 1e-6
    X_norm = (X_raw - means) / stds

    # Save normalization scaler
    scaler_payload = {
        "feature_names": feature_cols,
        "means": means.tolist(),
        "stds": stds.tolist()
    }
    with open(ckpt_dir / "machinery_scaler.json", "w") as f:
        json.dump(scaler_payload, f, indent=2)

    # 80/20 train/val chronological split
    split_idx = int(len(X_norm) * 0.8)
    train_ds = RotatingMachineryDataset(X_norm[:split_idx], y_label[:split_idx], y_rul[:split_idx], seq_len=seq_len)
    val_ds = RotatingMachineryDataset(X_norm[split_idx:], y_label[split_idx:], y_rul[split_idx:], seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RefineryPredictiveMaintenanceLSTM(
        n_features=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    print(f"[*] Training Multi-Task Bi-LSTM on continuous refinery rotating machinery data...")
    print(f"    Features: {len(feature_cols)} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by_label, by_rul in train_loader:
            optimizer.zero_grad()
            logits, pred_rul = model(bx)
            loss_cls = bce_loss_fn(logits.squeeze(-1), by_label)
            loss_reg = mse_loss_fn(pred_rul.squeeze(-1) / 500.0, by_rul / 500.0) # scaled RUL
            loss = loss_cls + 1.5 * loss_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_cls_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by_label, by_rul in val_loader:
                logits, pred_rul = model(bx)
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                correct += (preds == by_label).sum().item()
                total += len(by_label)

        val_acc = (correct / max(1, total)) * 100.0
        print(f"    Epoch [{ep}/{epochs}] - Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.1f}%")

    # Save trained checkpoint
    model_path = ckpt_dir / "refinery_rotating_lstm.pt"
    torch.save(model.state_dict(), model_path)

    # Export TorchScript Edge Binary for SCADA runtime
    model.eval()
    example_input = torch.randn(1, seq_len, len(feature_cols), dtype=torch.float32)
    scripted_model = torch.jit.trace(model, example_input)
    edge_path = ckpt_dir / "refinery_rotating_edge.pt"
    scripted_model.save(str(edge_path))

    print(f"[+] Model checkpoint saved: {model_path}")
    print(f"[+] TorchScript SCADA binary exported: {edge_path}")

    return {
        "status": "TRAINING_COMPLETE",
        "final_val_accuracy_pct": round(val_acc, 2),
        "model_path": str(model_path),
        "edge_path": str(edge_path)
    }

if __name__ == "__main__":
    train_refinery_machinery_lstm()
