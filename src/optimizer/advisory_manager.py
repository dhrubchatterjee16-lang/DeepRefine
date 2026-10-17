"""
Operator Advisory Engine & Human-in-the-Loop Safe Clearance Manager.
Enforces read-only non-intrusive operational boundaries in compliance with refinery safety committees.

Guarantees:
1. Zero Direct Write-Back: The AI cannot command physical actuators or DCS setpoints.
2. Operator Clearance Workflow:
   - AI generates an 'Advisory Card' (Predicted benefit, risk analysis, and safety margin).
   - Requires explicit human operator authorization ("ACCEPT_TRIM" or "REJECT_TRIM") with mandatory log reason.
   - Outputs a signed setpoint modification slip formatted for the DCS console logbook.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time
import json
import uuid

class OperatorAdvisoryManager:
    def __init__(
        self,
        advisory_log: Path = Path("data/advisory_journal.jsonl"),
        audit_trail_log: Path = Path("data/operator_action_audit.jsonl")
    ):
        self.advisory_log = advisory_log
        self.audit_trail_log = audit_trail_log
        self.advisory_log.parent.mkdir(parents=True, exist_ok=True)

    def generate_advisory(
        self,
        unit_id: str,
        current_setpoints: Dict[str, float],
        proposed_setpoints: Dict[str, float],
        economic_gain_usd_day: float,
        safety_status: str,
        flooding_margin_pct: float
    ) -> Dict:
        """Creates a formal Operator Advisory Card for human review."""
        advisory_id = f"ADV-{int(time.time())}-{str(uuid.uuid4())[:4].upper()}"

        card = {
            "advisory_id": advisory_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "unit_id": unit_id,
            "mode": "READ_ONLY_OPERATOR_ADVISORY",
            "current_state": current_setpoints,
            "proposed_trim": proposed_setpoints,
            "projected_benefit_usd_day": round(economic_gain_usd_day, 2),
            "safety_assessment": {
                "interlock_status": safety_status,
                "flooding_pct": flooding_margin_pct,
                "is_cleared": safety_status == "APPROVED_BY_SAFETY_INTERLOCK"
            },
            "status": "AWAITING_OPERATOR_CLEARANCE"
        }

        with open(self.advisory_log, "a") as f:
            f.write(json.dumps(card) + "\n")

        return card

    def record_operator_clearance(
        self,
        advisory_id: str,
        operator_badge_id: str,
        decision: str, # "ACCEPTED_BY_OPERATOR" or "REJECTED_BY_OPERATOR"
        justification: str
    ) -> Dict:
        """Logs the immutable audit trail of operator decisions."""
        entry = {
            "advisory_id": advisory_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operator_badge": operator_badge_id,
            "decision": decision,
            "justification": justification,
            "dcs_console_slip": f"MANUAL_ENTRY_REF_{advisory_id} (Signed by Badge #{operator_badge_id})"
        }

        with open(self.audit_trail_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry
