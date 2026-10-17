import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from model.lstm_pm import RefineryPredictiveMaintenanceLSTM
from model.explainability import RefineryExplainabilityEngine

app = FastAPI(title="Refinery PM Real-Time Inference & Explainability API", version="1.1.0")

# Prometheus Metrics
INFERENCE_COUNT = Counter("pm_inference_requests_total", "Total inference requests scored")
INFERENCE_LATENCY = Histogram("pm_inference_latency_seconds", "Inference duration in seconds")
FAILURE_RISK_GAUGE = Gauge("pm_latest_failure_risk_score", "Latest failure risk score (0 to 1)", ["equipment"])
RUL_HOURS_GAUGE = Gauge("pm_latest_predicted_rul_hours", "Latest predicted RUL in hours", ["equipment"])

class SensorWindowPayload(BaseModel):
    equipment_id: str = Field(..., example="WGC-101-COMPRESSOR")
    features: List[List[float]] = Field(..., description="Window of shape [seq_len, n_features]")

MODEL = None
SCALER = None
EXPLAINER = None

def get_resources():
    global MODEL, SCALER, EXPLAINER
    if MODEL is None:
        ckpt_path = Path("checkpoints/best_refinery_pm.pt")
        scaler_path = Path("checkpoints/feature_scaler.json")
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            n_features = checkpoint.get("n_features", 48)
            MODEL = RefineryPredictiveMaintenanceLSTM(
                n_features=n_features, hidden_dim=128, num_layers=2, dropout=0.0
            )
            MODEL.load_state_dict(checkpoint["model_state_dict"])
            MODEL.eval()
        if scaler_path.exists():
            with open(scaler_path, "r") as f:
                SCALER = json.load(f)
        EXPLAINER = RefineryExplainabilityEngine(ckpt_path=ckpt_path, scaler_path=scaler_path)
    return MODEL, SCALER, EXPLAINER

@app.get("/health")
def health():
    return {"status": "healthy", "service": "refinery-pm-inference"}

@app.post("/predict")
def predict(payload: SensorWindowPayload):
    start_time = time.time()
    INFERENCE_COUNT.inc()

    arr = np.array(payload.features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window shape {arr.shape}. Expected [sequence_length >= 5, n_features]"
        )

    model, scaler, explainer = get_resources()
    tensor_in = torch.tensor(arr).unsqueeze(0)

    if model is not None:
        with torch.no_grad():
            prob_logits, rul_pred = model(tensor_in)
            failure_prob = float(torch.sigmoid(prob_logits).item())
            rul_hours = max(0.0, float(rul_pred.item()))
    else:
        vib_magnitude = float(np.mean(np.abs(arr)))
        failure_prob = float(1.0 / (1.0 + np.exp(- (vib_magnitude - 5.0) / 2.0)))
        rul_hours = float(max(12.0, 168.0 * (1.0 - failure_prob)))

    latency = time.time() - start_time
    INFERENCE_LATENCY.observe(latency)

    # Prometheus gauges
    FAILURE_RISK_GAUGE.labels(equipment=payload.equipment_id).set(failure_prob)
    RUL_HOURS_GAUGE.labels(equipment=payload.equipment_id).set(rul_hours)

    severity = "NORMAL"
    if failure_prob > 0.8:
        severity = "CRITICAL_ALERT"
    elif failure_prob > 0.5:
        severity = "WARNING"

    # Explainability root-cause analysis if elevated risk
    root_causes = []
    if failure_prob > 0.4 and explainer is not None:
        root_causes = explainer.explain(arr, top_k=3)

    return {
        "equipment_id": payload.equipment_id,
        "failure_probability": round(failure_prob, 4),
        "predicted_rul_hours": round(rul_hours, 1),
        "severity": severity,
        "root_cause_attributions": root_causes,
        "latency_ms": round(latency * 1000, 2)
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
