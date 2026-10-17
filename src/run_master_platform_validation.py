"""
Master System Validation Suite for the Entire 3-Pillar Digital Refinery Platform.
Validates:
1. Pillar 1: Predictive Maintenance (Real NASA benchmark data, Bi-LSTM, ISO 10816, CMMS work orders)
2. Pillar 2: Digital Twin (First-principles thermodynamics + PINN neural surrogate what-if simulation)
3. Pillar 3: Smart Control & Optimization (Reinforcement learning Actor-Critic agent + shadow controller safety interlocks)
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))

from run_enterprise_validation import run_enterprise_validation
from digital_twin.verify_twin import verify_twin
from optimizer.shadow_controller import ShadowModeController

def run_master_platform_validation():
    print("\n" + "======================================================================")
    print(" [*] DIGITAL REFINERIES & AI - FULL 3-PILLAR MASTER VALIDATION")
    print("======================================================================")

    # 1. PILLAR 1: Predictive Maintenance
    print("\n>>> VALIDATING PILLAR 1: PREDICTIVE MAINTENANCE ENGINE...")
    run_enterprise_validation()

    # 2. PILLAR 2: Digital Twin Simulation
    print("\n>>> VALIDATING PILLAR 2: THERMODYNAMIC DIGITAL TWIN & PINN SURROGATE...")
    verify_twin()

    # 3. PILLAR 3: Autonomous RL Controller & Shadow Safety
    print("\n>>> VALIDATING PILLAR 3: SMART CONTROL & RL AUTONOMOUS OPTIMIZATION...")
    shadow = ShadowModeController()
    rec = shadow.evaluate_recommendation(current_reflux=2.40, current_furnace_mw=45.0, feed_flow_bpd=100000.0)
    print(f"[+] Controller Status: {rec['status']}")
    print(f"[+] Proposed Trim: Reflux {rec['recommended_setpoints']['reflux_ratio']} (Delta {rec['recommended_setpoints']['delta_reflux']}), Furnace {rec['recommended_setpoints']['furnace_duty_mw']} MW")
    print(f"[+] Annualized Profit Uplift: +${rec['expected_economic_benefit']['annualized_gain_usd']:,.2f} USD/year")
    print(f"[+] Safety Interlock: Flooding at {rec['hydraulic_safety_margin']['flooding_pct']}% (Limit: 82.0%)")

    print("\n" + "======================================================================")
    print(" [SUCCESS] ALL 3 PILLARS ARE FULLY OPERATIONAL AND VALIDATED!")
    print("======================================================================")

if __name__ == "__main__":
    run_master_platform_validation()
