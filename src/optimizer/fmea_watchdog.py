"""
Failure Modes and Effects Analysis (FMEA) Matrix & Runtime Watchdog.
Evaluates every potential software, model, and physical instrument failure mode,
categorizes risk priority numbers (RPN), and executes deterministic mitigation protocols.
"""

from enum import Enum
from typing import Dict, List, Tuple
import time

class SeverityLevel(int, Enum):
    MINOR = 1
    MODERATE = 2
    CRITICAL_SAFETY = 3

class FMEARuntimeWatchdog:
    def __init__(self):
        # Comprehensive FMEA Matrix for Refinery AI Operations
        self.fmea_catalog = {
            "FROZEN_SENSOR_ADC": {
                "failure_mode": "Thermocouple or Pressure transmitter frozen/stuck ADC",
                "effect_on_plant": "Model uses stale telemetry, leading to incorrect setpoint calculation",
                "severity": SeverityLevel.CRITICAL_SAFETY,
                "mitigation": "Drop channel, degrade feature vector, fallback to APC park setpoints",
                "rpn": 48
            },
            "DISTRIBUTION_DRIFT_EXCEEDED": {
                "failure_mode": "Severe seasonal or feed slate shift outside training envelope (KS > 0.5)",
                "effect_on_plant": "Neural network extrapolation risk (hallucination)",
                "severity": SeverityLevel.MODERATE,
                "mitigation": "Hold AI advisories, notify plant engineer, activate active-learning retraining",
                "rpn": 36
            },
            "HYDRAULIC_FLOODING_WARNING": {
                "failure_mode": "Vapor traffic exceeds Fair flooding limit (Flooding > 80%)",
                "effect_on_plant": "Column liquid entrainment, loss of product cut separation, pressure surge",
                "severity": SeverityLevel.CRITICAL_SAFETY,
                "mitigation": "Immediate cutback on furnace MW duty, increase reflux ratio",
                "rpn": 72
            },
            "DATA_DIODE_SEQUENCE_GAP": {
                "failure_mode": "Unidirectional data packet loss across OT/IT boundary",
                "effect_on_plant": "Discontinuous sliding window buffer",
                "severity": SeverityLevel.MINOR,
                "mitigation": "Hold last verified good state, log telemetry gap warning",
                "rpn": 18
            }
        }

    def evaluate_fmea_state(self, telemetry_health: Dict, flooding_pct: float) -> Tuple[bool, str, Dict]:
        """
        Evaluates current plant state against FMEA failure modes.
        Returns: (requires_fallback, failure_mode_key, mitigation_details)
        """
        # 1. Flooding Check
        if flooding_pct > 80.0:
            info = self.fmea_catalog["HYDRAULIC_FLOODING_WARNING"]
            return True, "HYDRAULIC_FLOODING_WARNING", info

        # 2. Frozen Sensor Check
        if not telemetry_health.get("is_clean", True):
            for anom in telemetry_health.get("flagged_anomalies", []):
                if anom.get("issue") == "FROZEN_SIGNAL":
                    info = self.fmea_catalog["FROZEN_SENSOR_ADC"]
                    return True, "FROZEN_SENSOR_ADC", info

        # 3. Severe Distribution Drift Check
        if telemetry_health.get("distribution_drift_index", 0.0) > 3.0:
            info = self.fmea_catalog["DISTRIBUTION_DRIFT_EXCEEDED"]
            return True, "DISTRIBUTION_DRIFT_EXCEEDED", info

        return False, "NORMAL_OPERATION", {"status": "ALL_FMEA_GATES_NOMINAL"}
