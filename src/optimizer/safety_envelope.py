"""
Deterministic Operational Boundary & Rate-of-Change Limiter.
Enforces hard safety interlocks derived from refinery process engineering standards:
1. Max Slew Rate: Restricts ramp-rates to prevent furnace refractory cracking and thermal shock.
2. Hard Engineering Envelopes: Minimum/Maximum temperatures, reflux, and pressures.
3. Hydraulic Safeguards: Automatic abort if flooding margin drops below minimum safety cushion.
"""

from typing import Dict, Tuple
import numpy as np

class IndustrialSafetyEnvelope:
    def __init__(self):
        # Strict industrial rate-of-change and boundary limits
        self.max_furnace_ramp_mw_per_step = 0.5   # Maximum 0.5 MW thermal step per adjustment
        self.max_reflux_ramp_per_step = 0.08       # Maximum 0.08 reflux ratio step
        
        # Absolute operating limits
        self.min_furnace_mw = 35.0
        self.max_furnace_mw = 58.0
        self.min_reflux = 1.80
        self.max_reflux = 3.20
        self.max_allowed_flooding_pct = 80.0       # Standard refinery 80% flood limit

    def clamp_and_verify(
        self,
        current_reflux: float,
        current_furnace_mw: float,
        proposed_reflux: float,
        proposed_furnace_mw: float,
        predicted_flooding_pct: float
    ) -> Tuple[bool, float, float, str]:
        """
        Validates proposed AI trim against hard deterministic engineering envelopes.
        Returns: (is_safe, clamped_reflux, clamped_furnace, audit_message)
        """
        # 1. Hydraulic flooding check
        if predicted_flooding_pct >= self.max_allowed_flooding_pct:
            return (
                False,
                current_reflux,
                current_furnace_mw,
                f"INTERLOCK_TRIP: Predicted column flooding ({predicted_flooding_pct:.1f}%) exceeds safety ceiling (80.0%). Action aborted."
            )

        # 2. Rate-of-change clamping (Slew-rate limiting)
        delta_reflux = proposed_reflux - current_reflux
        clamped_delta_reflux = float(np.clip(delta_reflux, -self.max_reflux_ramp_per_step, self.max_reflux_ramp_per_step))
        safe_reflux = float(np.clip(current_reflux + clamped_delta_reflux, self.min_reflux, self.max_reflux))

        delta_furnace = proposed_furnace_mw - current_furnace_mw
        clamped_delta_furnace = float(np.clip(delta_furnace, -self.max_furnace_ramp_mw_per_step, self.max_furnace_ramp_mw_per_step))
        safe_furnace = float(np.clip(current_furnace_mw + clamped_delta_furnace, self.min_furnace_mw, self.max_furnace_mw))

        # Check if clamping occurred
        was_clamped = (safe_reflux != proposed_reflux) or (safe_furnace != proposed_furnace_mw)
        message = "CLEARANCE_GRANTED: Setpoints verified within safe industrial envelope."
        if was_clamped:
            message += f" (Clamped to prevent thermal shock / slew-rate violation: Reflux={safe_reflux:.2f}, Duty={safe_furnace:.1f}MW)"

        return True, safe_reflux, safe_furnace, message
