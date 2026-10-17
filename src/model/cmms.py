"""
CMMS (SAP PM / IBM Maximo) Work Order Dispatcher & Operator Feedback Journal.
Translates critical AI failure predictions into formatted industrial work notifications
and captures operator feedback for active learning.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import uuid

class RefineryCMMSBridge:
    def __init__(
        self,
        orders_file: Path = Path("data/cmms_work_orders.jsonl"),
        feedback_file: Path = Path("data/operator_feedback.jsonl")
    ):
        self.orders_file = orders_file
        self.feedback_file = feedback_file
        self.orders_file.parent.mkdir(parents=True, exist_ok=True)

        # Asset parts and isolation protocol catalog
        self.equipment_catalog = {
            "WGC-101-COMPRESSOR": {
                "unit_name": "Wet Gas Compressor Train A",
                "criticality": "HIGH_P1",
                "spare_parts": ["SKF-23134-2CS Thrust Bearing", "John Crane Dry Gas Mechanical Seal"],
                "isolation_procedure": "SOP-MECH-ISO-402 (LOTO on 6.6kV Feed & Nitrogen Purge)",
                "downtime_cost_per_hour_usd": 12500.0
            },
            "CDC-201-COLUMN": {
                "unit_name": "Crude Distillation Column Main Fractionator",
                "criticality": "CRITICAL_P0",
                "spare_parts": ["Sulzer Structured Packing MellapakPlus", "Demister Pad 316L SS"],
                "isolation_procedure": "SOP-PROC-ISO-101 (Hot Blind Isolation & Steaming Out)",
                "downtime_cost_per_hour_usd": 38000.0
            },
            "P-301-PUMP": {
                "unit_name": "Atmospheric Residue Booster Pump",
                "criticality": "MEDIUM_P2",
                "spare_parts": ["Flowserve Cartridge Seal", "Impeller Key 410 SS"],
                "isolation_procedure": "SOP-PUMP-ISO-205 (Suction/Discharge Blinding & Tagout)",
                "downtime_cost_per_hour_usd": 4500.0
            }
        }

    def dispatch_work_order(self, alert_payload: Dict) -> Dict:
        """
        Creates an industrial SAP PM Notification (Type M2 - Malfunction Report)
        or IBM Maximo Work Order based on AI degradation predictions.
        """
        equipment_id = alert_payload.get("equipment_id", "UNKNOWN_ASSET")
        info = self.equipment_catalog.get(equipment_id, {
            "unit_name": equipment_id,
            "criticality": "P2_STANDARD",
            "spare_parts": ["General Maintenance Kit"],
            "isolation_procedure": "Standard LOTO Protocol",
            "downtime_cost_per_hour_usd": 5000.0
        })

        order_id = f"WO-{int(time.time())}-{str(uuid.uuid4())[:4].upper()}"
        rul_h = alert_payload.get("predicted_rul_hours", 24.0)
        failure_prob = alert_payload.get("failure_probability", 0.85)

        # Economic risk calculation
        est_downtime_hours = 8.0
        financial_risk_usd = round(failure_prob * (est_downtime_hours * info["downtime_cost_per_hour_usd"]), 2)

        work_order = {
            "order_id": order_id,
            "sap_notification_type": "M2_MALFUNCTION_REPORT",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "equipment_id": equipment_id,
            "unit_name": info["unit_name"],
            "criticality": info["criticality"],
            "urgency": "SCHEDULE_BEFORE_RUL_EXPIRATION" if rul_h > 24 else "EMERGENCY_INTERVENTION",
            "predicted_rul_hours": rul_h,
            "failure_probability": failure_prob,
            "economic_risk_usd": financial_risk_usd,
            "diagnosed_root_causes": alert_payload.get("root_causes", []),
            "required_spare_parts": info["spare_parts"],
            "isolation_sop": info["isolation_procedure"],
            "status": "DISPATCHED_TO_PLANNER"
        }

        with open(self.orders_file, "a") as f:
            f.write(json.dumps(work_order) + "\n")

        return work_order

    def record_operator_feedback(self, order_id: str, operator_action: str, notes: str) -> Dict:
        """
        Records human-in-the-loop operator validation (e.g. Confirmed True Alarm, Dismissed Calibration Error).
        """
        feedback_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "order_id": order_id,
            "operator_action": operator_action, # "ACKNOWLEDGED_TRUE_ALERT" or "DISMISSED_FALSE_ALARM"
            "notes": notes
        }
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        return feedback_entry
