"""
End-to-End Refinery Predictive Maintenance Pipeline Test Runner.
Executes and validates all PM stages in sequence:
1. Synthetic Data Verification
2. Feature Engine Check
3. Bi-LSTM Model Checkpoint Validation
4. Explainability & Root-Cause Analysis Attribution
5. Streaming Consumer Window Buffer & Alert Dispatch
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from model.lstm_pm import RefineryPredictiveMaintenanceLSTM
from model.explainability import RefineryExplainabilityEngine
from serve.stream_consumer import RefineryStreamConsumer

def run_test():
    print("======================================================================")
    print(" [*] REFINERY PREDICTIVE MAINTENANCE ENGINE - END-TO-END VALIDATION")
    print("======================================================================")

    # 1. Check datasets
    sensors_path = Path("data/raw/sensors/refinery_sensors.parquet")
    features_path = Path("data/processed/features.parquet")
    ckpt_path = Path("checkpoints/best_refinery_pm.pt")

    assert sensors_path.exists(), f"Missing {sensors_path}"
    assert features_path.exists(), f"Missing {features_path}"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"
    print("[+] Step 1: Data lakes & checkpoints verified.")

    # 2. Check Feature Store
    df_feat = pd.read_parquet(features_path)
    print(f"[+] Step 2: Feature store loaded successfully ({len(df_feat):,} samples x {df_feat.shape[1]} features).")

    # 3. Model Forward Pass
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = RefineryPredictiveMaintenanceLSTM(n_features=ckpt["n_features"], hidden_dim=128, num_layers=2, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_window = torch.randn(1, 30, ckpt["n_features"])
    with torch.no_grad():
        logits, rul = model(sample_window)
        prob = torch.sigmoid(logits).item()
        rul_h = rul.item()
    print(f"[+] Step 3: PyTorch Model inference OK -> Prob: {prob:.4f}, RUL: {rul_h:.1f} hrs.")

    # 4. Explainability & Root-Cause Attribution
    explainer = RefineryExplainabilityEngine(ckpt_path=ckpt_path)
    sample_window_np = df_feat.iloc[-30:].values
    attributions = explainer.explain(sample_window_np, top_k=3)
    print(f"[+] Step 4: AI Explainability Attribution calculated:")
    for attr in attributions:
        print(f"   - {attr['sensor']}: {attr['contribution_pct']}% attribution")

    # 5. Streaming Buffer & Real-time Alerting
    consumer = RefineryStreamConsumer(window_size=30, ckpt_path=ckpt_path)
    print("[+] Step 5: Testing Real-time Stream Consumer & Alert Triggering...")
    
    # Push 30 ticks of degrading telemetry
    for i in range(30):
        tick = list(df_feat.iloc[i].values)
        res = consumer.process_sensor_tick("WGC-101-COMPRESSOR", tick)

    print(f"   -> Result on full window: Risk: {res['failure_probability']*100:.1f}%, Alert: {res['alert_level']}, RUL: {res['predicted_rul_hours']} hrs")

    print("\n" + "======================================================================")
    print(" [SUCCESS] ALL PREDICTIVE MAINTENANCE ENGINE COMPONENTS PASSED!")
    print("======================================================================")

if __name__ == "__main__":
    run_test()
