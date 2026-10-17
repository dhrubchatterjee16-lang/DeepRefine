"""
Role-Based Access Control (RBAC) & Security Policy Gatekeeper for DeepRefine.
Complies with IEC 62443 Security Levels (SL-1 to SL-3).

Enforces Role Permissions:
- READ_ONLY_VIEWER: View telemetry and model dashboards.
- RELIABILITY_ENGINEER: View diagnostics, model explainability, and configure asset thresholds.
- BOARD_OPERATOR: Authorize/sign setpoint advisory cards and submit DCS console slips.
- PLANT_ADMIN: Retrain models, configure historian endpoints, and manage user keys.
"""

from enum import Enum
from typing import Dict, List, Optional
import time

class IndustrialRole(str, Enum):
    VIEWER = "READ_ONLY_VIEWER"
    RELIABILITY = "RELIABILITY_ENGINEER"
    BOARD_OPERATOR = "BOARD_OPERATOR"
    ADMIN = "PLANT_ADMIN"

class RBACGatekeeper:
    def __init__(self):
        # Industrial permission matrix
        self.permissions = {
            IndustrialRole.VIEWER: ["VIEW_DASHBOARD", "VIEW_METRICS"],
            IndustrialRole.RELIABILITY: ["VIEW_DASHBOARD", "VIEW_METRICS", "VIEW_EXPLAINABILITY", "DISMISS_FALSE_ALARM", "EXPORT_MODELS"],
            IndustrialRole.BOARD_OPERATOR: ["VIEW_DASHBOARD", "VIEW_METRICS", "AUTHORIZE_ADVISORY", "SIGN_DCS_SLIP"],
            IndustrialRole.ADMIN: ["VIEW_DASHBOARD", "VIEW_METRICS", "VIEW_EXPLAINABILITY", "DISMISS_FALSE_ALARM", "AUTHORIZE_ADVISORY", "SIGN_DCS_SLIP", "TRIGGER_RETRAINING", "CONFIGURE_SECURITY"]
        }

    def verify_action(self, role: IndustrialRole, action: str) -> bool:
        """Verifies if the assigned role has clearance to perform the target action."""
        allowed_actions = self.permissions.get(role, [])
        return action in allowed_actions
