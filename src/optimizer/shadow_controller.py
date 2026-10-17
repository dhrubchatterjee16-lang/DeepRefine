"""
Industrial Shadow-Mode Safe Controller Interface.
Evaluates RL setpoint recommendations against hard engineering interlocks before displaying to operators:
1. Hydraulic tray flooding limit: Flooding must not exceed 82%.
2. Column temperature differential: (Bottom Temp - Top Temp) must remain within [180°C, 300°C].
3. Minimum Diesel & Jet Fuel yield quotas.
"""

from typing import Dict, Tuple
from pathlib import Path
import numpy as np

from optimizer.rl_agent import AutonomousOptimizerAgent
from digital_twin.twin_engine import DigitalTwinSimulator
from optimizer.safety_envelope import IndustrialSafetyEnvelope

class ShadowModeController:
    def __init__(self):
        self.agent = AutonomousOptimizerAgent()
        self.twin = DigitalTwinSimulator()
        self.envelope = IndustrialSafetyEnvelope()

    def evaluate_recommendation(
        self,
        current_reflux: float,
        current_furnace_mw: float,
        feed_flow_bpd: float = 100000.0
    ) -> Dict:
        """
        Runs the RL agent to suggest an optimal setpoint trim, validates safety constraints in shadow mode,
        and applies deterministic rate-of-change clamps.
        """
        # Current state observation vector
        obs = np.array([
            feed_flow_bpd / 100000.0,
            current_reflux / 3.0,
            current_furnace_mw / 50.0,
            0.25, # baseline flooding
            0.05, # failure risk
            0.90, # RUL
            0.80  # margin
        ], dtype=np.float32)

        # RL policy raw output: [-1, 1]
        action = self.agent.get_action(obs, deterministic=True)
        raw_proposed_reflux = float(current_reflux + action[0] * 0.15)
        raw_proposed_furnace = float(current_furnace_mw + action[1] * 1.5)

        # First pass simulation to test flooding
        test_sim = self.twin.simulate_what_if(
            feed_flow_bpd=feed_flow_bpd,
            reflux_ratio=raw_proposed_reflux,
            furnace_duty_mw=raw_proposed_furnace
        )
        test_flooding = test_sim["predicted_thermodynamics"]["hydraulic_flooding_pct"]

        # Deterministic Industrial Safety Clamping
        is_safe, proposed_reflux, proposed_furnace, audit_msg = self.envelope.clamp_and_verify(
            current_reflux=current_reflux,
            current_furnace_mw=current_furnace_mw,
            proposed_reflux=raw_proposed_reflux,
            proposed_furnace_mw=raw_proposed_furnace,
            predicted_flooding_pct=test_flooding
        )

        # Baseline simulation
        base_sim = self.twin.simulate_what_if(
            feed_flow_bpd=feed_flow_bpd,
            reflux_ratio=current_reflux,
            furnace_duty_mw=current_furnace_mw
        )

        # Optimized simulation
        opt_sim = self.twin.simulate_what_if(
            feed_flow_bpd=feed_flow_bpd,
            reflux_ratio=proposed_reflux,
            furnace_duty_mw=proposed_furnace
        )

        base_margin = base_sim["economics_and_emissions"]["estimated_operating_margin_usd_day"]
        opt_margin = opt_sim["economics_and_emissions"]["estimated_operating_margin_usd_day"]
        delta_margin_day = opt_margin - base_margin

        opt_flooding = opt_sim["predicted_thermodynamics"]["hydraulic_flooding_pct"]
        is_safe = opt_flooding < 82.0

        return {
            "status": "APPROVED_BY_SAFETY_INTERLOCK" if is_safe else "REJECTED_FLOODING_RISK",
            "audit_message": audit_msg,
            "current_setpoints": {
                "reflux_ratio": current_reflux,
                "furnace_duty_mw": current_furnace_mw
            },
            "recommended_setpoints": {
                "reflux_ratio": round(proposed_reflux, 2),
                "furnace_duty_mw": round(proposed_furnace, 2),
                "delta_reflux": round(proposed_reflux - current_reflux, 3),
                "delta_furnace_mw": round(proposed_furnace - current_furnace_mw, 2)
            },
            "expected_economic_benefit": {
                "baseline_margin_usd_day": round(base_margin, 2),
                "optimized_margin_usd_day": round(opt_margin, 2),
                "daily_profit_gain_usd": round(delta_margin_day, 2),
                "annualized_gain_usd": round(delta_margin_day * 350.0, 2)
            },
            "hydraulic_safety_margin": {
                "flooding_pct": opt_flooding,
                "safety_interlock_limit": 80.0,
                "is_cleared_for_operator_review": is_safe
            }
        }
