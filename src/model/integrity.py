"""
Refinery Physical Integrity & Vibration Standards Module.
1. Implements ISO 10816-3 machinery vibration classification:
   - Zone A: Newly commissioned / Good
   - Zone B: Acceptable for unrestricted continuous operation
   - Zone C: Unsatisfactory / Limited operation (maintenance required)
   - Zone D: Unacceptable / Damage imminent (trip threshold)
2. Implements Operational Regime & Transient Detector:
   Detects plant startup, shutdown, and crude blend changeover transients to prevent false alarms.
"""

from enum import Enum
from typing import Dict, Tuple
import numpy as np

class ISO10816Zone(str, Enum):
    ZONE_A_GOOD = "ZONE_A_GOOD"
    ZONE_B_ACCEPTABLE = "ZONE_B_ACCEPTABLE"
    ZONE_C_UNSATISFACTORY = "ZONE_C_UNSATISFACTORY"
    ZONE_D_UNACCEPTABLE = "ZONE_D_UNACCEPTABLE"

class RefineryIntegrityEngine:
    """
    Evaluates rotating machinery against ISO 10816-3 Group 1 (large machines >300kW, flexible foundations)
    and verifies operational steady-state stability.
    """
    def __init__(self):
        # Thresholds in mm/s RMS velocity for ISO 10816-3 Group 1
        self.iso_limits = {
            "A_to_B": 2.8,   # Border between Good and Acceptable
            "B_to_C": 4.5,   # Border between Acceptable and Unsatisfactory (Alert)
            "C_to_D": 7.1    # Border between Unsatisfactory and Unacceptable (Trip/Danger)
        }

    def evaluate_vibration_iso10816(self, rms_velocity_mms: float) -> Tuple[ISO10816Zone, str]:
        """Classifies vibration levels according to ISO 10816-3 standards."""
        if rms_velocity_mms < self.iso_limits["A_to_B"]:
            return ISO10816Zone.ZONE_A_GOOD, "Vibration within newly-commissioned specifications."
        elif rms_velocity_mms < self.iso_limits["B_to_C"]:
            return ISO10816Zone.ZONE_B_ACCEPTABLE, "Vibration acceptable for continuous unrestricted run."
        elif rms_velocity_mms < self.iso_limits["C_to_D"]:
            return ISO10816Zone.ZONE_C_UNSATISFACTORY, "Warning: Significant degradation. Schedule inspection."
        else:
            return ISO10816Zone.ZONE_D_UNACCEPTABLE, "CRITICAL: Imminent bearing/shaft mechanical damage. Trip required."

    def detect_operational_regime(self, recent_history: np.ndarray, rate_threshold: float = 0.15) -> Dict:
        """
        Determines if the unit is in a dynamic transient (e.g. crude slate switch, throughput ramp)
        versus steady-state equilibrium.
        recent_history: shape [window_len, n_tags]
        """
        if len(recent_history) < 5:
            return {"regime": "STEADY_STATE", "suppress_alerts": False, "transient_confidence": 0.0}

        # Measure rate of change (gradient) across temperature and pressure channels
        gradients = np.gradient(recent_history, axis=0)
        norm_gradient = np.mean(np.abs(gradients), axis=0)
        max_rate = float(np.max(norm_gradient))

        is_transient = max_rate > rate_threshold
        regime_label = "DYNAMIC_TRANSIENT_CRUDE_SWITCH" if is_transient else "STEADY_STATE"

        return {
            "regime": regime_label,
            "suppress_alerts": is_transient,
            "max_rate_of_change": round(max_rate, 4),
            "recommendation": "Holding model alerts during process ramp" if is_transient else "Model active"
        }
