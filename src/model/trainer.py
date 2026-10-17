"""
Pure PyTorch Training Script for Refinery Predictive Maintenance LSTM.
Does not require PyTorch Lightning. Trains multi-task heads (Failure risk + RUL),
evaluates validation AUC/loss, and saves the best model checkpoint and scaling stats.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_pm import RefineryPredictiveMaintenanceLSTM

class RefinerySequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, rul: np.ndarray, seq_len: int = 30):
        self.features = features
        self.labels = labels
        self.rul = rul
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y_label = self.labels[idx + self.seq_len]
        y_rul = self.rul[idx + self.seq_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_label, dtype=torch.float32),
            torch.tensor(y_rul, dtype=torch.float32)
        )

def train_model():
    parser = argparse.ArgumentParser(description="Train Refinery PM LSTM Model")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--labels", type=str, default="data/processed/failure_label.parquet")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    feat_path = Path(args.features)
    label_path = Path(args.labels)

    if not feat_path.exists() or not label_path.exists():
        print(f"Error: Processed data not found at {feat_path} and {label_path}")
        return

    print(f"Loading engineered features from {feat_path}...")
    df_feat = pd.read_parquet(feat_path)
    df_labels = pd.read_parquet(label_path)

    # Calculate normalization parameters and save for production inference
    mean = df_feat.mean()
    std = df_feat.std().replace(0, 1.0)
    
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler_info = {
        "columns": list(df_feat.columns),
        "mean": mean.to_dict(),
        "std": std.to_dict()
    }
    with open(ckpt_dir / "feature_scaler.json", "w") as f:
        json.dump(scaler_info, f, indent=2)

    df_feat_norm = (df_feat - mean) / std

    X = df_feat_norm.values
    y = df_labels["failure_label"].values
    rul = df_labels["rul_hours"].values

    # Chronological 80/20 train/val split
    split_idx = int(len(X) * 0.8)
    train_ds = RefinerySequenceDataset(X[:split_idx], y[:split_idx], rul[:split_idx], seq_len=args.seq_len)
    val_ds = RefinerySequenceDataset(X[split_idx:], y[split_idx:], rul[split_idx:], seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_features = X.shape[1]
    print(f"Total features: {n_features}")
    print(f"Train samples: {len(train_ds):,}, Validation samples: {len(val_ds):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RefineryPredictiveMaintenanceLSTM(
        n_features=n_features,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ).to(device)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best_refinery_pm.pt"

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for batch_x, batch_y, batch_rul in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_rul = batch_rul.to(device)

            optimizer.zero_grad()
            logits, pred_rul = model(batch_x)

            loss_cls = bce_loss_fn(logits, batch_y)
            loss_reg = mse_loss_fn(pred_rul, batch_rul)
            loss = loss_cls + 0.005 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_total / max(1, train_batches)

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        correct_preds = 0
        total_eval = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_rul in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_rul = batch_rul.to(device)

                logits, pred_rul = model(batch_x)
                loss_cls = bce_loss_fn(logits, batch_y)
                loss_reg = mse_loss_fn(pred_rul, batch_rul)
                loss = loss_cls + 0.005 * loss_reg

                val_loss_total += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct_preds += (preds == batch_y).sum().item()
                total_eval += batch_y.size(0)

        avg_val_loss = val_loss_total / max(1, val_batches)
        val_acc = (correct_preds / max(1, total_eval)) * 100.0
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "n_features": n_features,
                "val_loss": best_val_loss,
                "val_acc": val_acc
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint: {best_ckpt_path}")

    print("Training process finished.")

if __name__ == "__main__":
    train_model()
