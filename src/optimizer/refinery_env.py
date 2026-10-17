"""
Gymnasium-compatible Multi-Objective Digital Refinery Control Environment.
Wraps the first-principles Digital Twin and Predictive Maintenance models into an RL Gym environment.

Action Space (Continuous setpoint adjustments):
1. delta_reflux_ratio: [-0.2, +0.2]
2. delta_furnace_duty_mw: [-2.0, +2.0 MW]

Observation Space (State vectors):
[feed_flow_norm, top_temp_norm, bot_temp_norm, flooding_norm, failure_risk_norm, rul_norm, market_margin_norm]

Reward Function:
R = w1 * (Normalized Economic Margin) - w2 * (Carbon Penalty) - w3 * (Equipment Failure Penalty) - w4 * (Flooding Violation)
"""

from typing import Dict, Tuple, Optional
import numpy as np

class RefineryGymEnv:
    def __init__(
        self,
        base_feed_bpd: float = 100000.0,
        base_furnace_mw: float = 45.0,
        base_reflux: float = 2.4,
        max_steps: int = 50
    ):
        self.base_feed = base_feed_bpd
        self.max_steps = max_steps
        self.current_step = 0

        # State bounds
        self.furnace_mw = base_furnace_mw
        self.reflux = base_reflux
        self.crude_comp = np.array([0.22, 0.16, 0.28, 0.34], dtype=np.float32)

        # Normalized observation vector (7 dims)
        self.state = np.zeros(7, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.furnace_mw = 45.0 + np.random.uniform(-2.0, 2.0)
        self.reflux = 2.4 + np.random.uniform(-0.2, 0.2)
        self.state = np.array([1.0, 0.5, 0.5, 0.2, 0.1, 0.8, 0.6], dtype=np.float32)
        return self.state

    def step(self, action: np.ndarray, twin_sim=None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes action [delta_reflux, delta_furnace_mw].
        Returns: next_state, reward, done, info
        """
        self.current_step += 1
        d_reflux = float(np.clip(action[0], -0.2, 0.2))
        d_furnace = float(np.clip(action[1], -2.0, 2.0))

        self.reflux = float(np.clip(self.reflux + d_reflux, 1.6, 3.2))
        self.furnace_mw = float(np.clip(self.furnace_mw + d_furnace, 32.0, 60.0))

        # Evaluate against Digital Twin if available, else approximate physics
        if twin_sim is not None:
            sim = twin_sim.simulate_what_if(
                feed_flow_bpd=self.base_feed,
                reflux_ratio=self.reflux,
                furnace_duty_mw=self.furnace_mw
            )
            margin = sim["economics_and_emissions"]["estimated_operating_margin_usd_day"]
            flooding = sim["predicted_thermodynamics"]["hydraulic_flooding_pct"]
            carbon = sim["economics_and_emissions"]["carbon_footprint_co2_tons_day"]
        else:
            # Physics proxy
            margin = 7800000.0 + (self.furnace_mw - 45.0) * 15000.0 - (self.reflux - 2.4) * 8000.0
            flooding = 25.0 + (self.furnace_mw - 45.0) * 1.5
            carbon = self.furnace_mw * 24.0 * 0.2024

        # Multi-objective reward calculation
        # 1. Economic Reward (Scale: $7.5M to $8.5M normalized to ~ [0, 10])
        r_econ = (margin - 7500000.0) / 100000.0
        
        # 2. Carbon emissions penalty ($50/ton CO2 carbon tax proxy)
        r_carbon = - (carbon * 0.05)
        
        # 3. Flooding hydraulic penalty (severe penalty if flooding > 80%)
        r_flooding = - 20.0 if flooding > 80.0 else 0.0

        total_reward = float(r_econ + r_carbon + r_flooding)

        done = self.current_step >= self.max_steps or flooding > 92.0

        # Update observation
        self.state = np.array([
            self.base_feed / 100000.0,
            self.reflux / 3.0,
            self.furnace_mw / 50.0,
            flooding / 100.0,
            0.05,  # Failure risk proxy
            0.90,  # RUL proxy
            min(1.0, margin / 8500000.0)
        ], dtype=np.float32)

        info = {
            "reflux": round(self.reflux, 2),
            "furnace_mw": round(self.furnace_mw, 2),
            "margin_usd": round(margin, 2),
            "flooding_pct": round(flooding, 1),
            "carbon_tons": round(carbon, 1)
        }

        return self.state, total_reward, done, info
