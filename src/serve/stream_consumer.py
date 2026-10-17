"""
Refinery Streaming Buffer & Real-Time Alert Engine (Enhanced v2).
Integrates:
- Sensor Sentinel (Drift & Data Quality checks)
- Physical Integrity Engine (ISO 10816 vibration bands & operational regime filtering)
- AI Explainability (Gradient attribution)
- CMMS Work Order Dispatcher (Auto-generation of SAP/Maximo work notifications)
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from model.lstm_pm import RefineryPredictiveMaintenanceLSTM
from model.explainability import RefineryExplainabilityEngine
from model.integrity import RefineryIntegrityEngine, ISO10816Zone
from model.sentinel import SensorSentinel
from model.cmms import RefineryCMMSBridge

class RefineryStreamConsumer:
    def __init__(
        self,
        window_size: int = 30,
        ckpt_path: Path = Path("checkpoints/best_refinery_pm.pt"),
        scaler_path: Path = Path("checkpoints/feature_scaler.json"),
        alerts_log_path: Path = Path("data/alerts.jsonl")
    ):
        self.window_size = window_size
        self.buffers: Dict[str, deque] = {}
        self.alerts_log_path = alerts_log_path
        self.alerts_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.n_features = checkpoint.get("n_features", 48)

        self.model = RefineryPredictiveMaintenanceLSTM(
            n_features=self.n_features, hidden_dim=128, num_layers=2, dropout=0.0
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with open(scaler_path, "r") as f:
            self.scaler = json.load(f)
            self.feature_names = self.scaler.get("columns", [])

        # Domain engines
        self.explainer = RefineryExplainabilityEngine(ckpt_path, scaler_path)
        self.integrity_engine = RefineryIntegrityEngine()
        self.sentinel = SensorSentinel(scaler_path)
        self.cmms_bridge = RefineryCMMSBridge()

    def process_sensor_tick(self, equipment_id: str, feature_vector: list) -> Dict:
        """
        Receives a single time-step feature vector, appends to the rolling buffer,
        and scores the window once full.
        """
        if equipment_id not in self.buffers:
            self.buffers[equipment_id] = deque(maxlen=self.window_size)

        self.buffers[equipment_id].append(feature_vector)

        if len(self.buffers[equipment_id]) < self.window_size:
            return {
                "status": "BUFFERING",
                "equipment_id": equipment_id,
                "buffered_points": len(self.buffers[equipment_id]),
                "required": self.window_size
            }

        window_arr = np.array(self.buffers[equipment_id], dtype=np.float32)

        # 1. Data Quality & Drift Validation
        quality_check = self.sentinel.validate_window(window_arr, self.feature_names)
        if not quality_check["can_infer"]:
            return {
                "status": "REJECTED_BAD_DATA",
                "equipment_id": equipment_id,
                "quality_check": quality_check
            }

        # 2. Operational Transient Detection (Crude switch filter)
        regime_info = self.integrity_engine.detect_operational_regime(window_arr)

        # 3. Model Inference
        x_tensor = torch.tensor(window_arr).unsqueeze(0)
        with torch.no_grad():
            logits, rul_pred = self.model(x_tensor)
            prob = float(torch.sigmoid(logits).item())
            rul_hours = max(0.0, float(rul_pred.item()))

        # 4. Physical ISO-10816 Vibration Severity Validation
        # Find vibration magnitude feature if available
        vib_features = [window_arr[-1, idx] for idx, col in enumerate(self.feature_names) if "Vibration" in col]
        est_vib_rms = float(np.mean(np.abs(vib_features))) if vib_features else 2.5
        iso_zone, iso_desc = self.integrity_engine.evaluate_vibration_iso10816(est_vib_rms)

        alert_level = "NORMAL"
        if not regime_info["suppress_alerts"]:
            if prob >= 0.8 or iso_zone == ISO10816Zone.ZONE_D_UNACCEPTABLE:
                alert_level = "CRITICAL"
            elif prob >= 0.5 or iso_zone == ISO10816Zone.ZONE_C_UNSATISFACTORY:
                alert_level = "WARNING"
        else:
            alert_level = "TRANSIENT_SUPPRESSED"

        result = {
            "timestamp": time.time(),
            "equipment_id": equipment_id,
            "failure_probability": round(prob, 4),
            "predicted_rul_hours": round(rul_hours, 1),
            "alert_level": alert_level,
            "iso10816_zone": iso_zone.value,
            "iso10816_assessment": iso_desc,
            "operational_regime": regime_info["regime"],
            "data_quality_score": quality_check["data_quality_score"],
            "root_causes": [],
            "cmms_work_order": None
        }

        # Trigger explainability and CMMS dispatch if Warning/Critical
        if alert_level in ["WARNING", "CRITICAL"]:
            result["root_causes"] = self.explainer.explain(window_arr, top_k=3)
            # Auto-dispatch work notification
            wo = self.cmms_bridge.dispatch_work_order(result)
            result["cmms_work_order"] = wo["order_id"]

            with open(self.alerts_log_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        return result
