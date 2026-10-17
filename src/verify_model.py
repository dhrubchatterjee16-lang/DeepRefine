import torch
from pathlib import Path
from model.lstm_pm import RefineryPredictiveMaintenanceLSTM

def verify():
    ckpt_path = Path("checkpoints/best_refinery_pm.pt")
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint verified: Epoch {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.4f}, Val Accuracy: {ckpt['val_acc']:.2f}%")
    
    model = RefineryPredictiveMaintenanceLSTM(
        n_features=ckpt["n_features"],
        hidden_dim=128,
        num_layers=2,
        dropout=0.0
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_x = torch.randn(1, 30, ckpt["n_features"])
    with torch.no_grad():
        logits, rul = model(sample_x)
        prob = torch.sigmoid(logits).item()
        rul_val = max(0.0, rul.item())

    print(f"Sample test inference: Failure Risk: {prob:.4f} | Predicted RUL: {rul_val:.1f} hours")
    print("Verification SUCCESSFUL!")

if __name__ == "__main__":
    verify()
