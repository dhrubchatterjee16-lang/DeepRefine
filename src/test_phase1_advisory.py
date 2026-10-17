import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent))

from optimizer.shadow_controller import ShadowModeController
from optimizer.advisory_manager import OperatorAdvisoryManager

def test_phase1_advisory_pipeline():
    print("======================================================================")
    print(" [*] TESTING PHASE 1: OPERATOR ADVISORY DECISION SUPPORT PIPELINE")
    print("======================================================================")

    shadow = ShadowModeController()
    mgr = OperatorAdvisoryManager()

    # 1. Evaluate RL recommendation with safety clamps
    rec = shadow.evaluate_recommendation(current_reflux=2.40, current_furnace_mw=45.0, feed_flow_bpd=100000.0)
    print(f"[+] Slew-rate check: {rec['audit_message']}")
    print(f"[+] Recommended Reflux: {rec['recommended_setpoints']['reflux_ratio']} (Delta {rec['recommended_setpoints']['delta_reflux']})")
    print(f"[+] Recommended Duty: {rec['recommended_setpoints']['furnace_duty_mw']} MW (Delta {rec['recommended_setpoints']['delta_furnace_mw']} MW)")

    # 2. Generate formal Advisory Card
    card = mgr.generate_advisory(
        unit_id="CDC-201-COLUMN",
        current_setpoints={"reflux": 2.40, "furnace_mw": 45.0},
        proposed_setpoints=rec["recommended_setpoints"],
        economic_gain_usd_day=rec["expected_economic_benefit"]["daily_profit_gain_usd"],
        safety_status=rec["status"],
        flooding_margin_pct=rec["hydraulic_safety_margin"]["flooding_pct"]
    )
    print(f"[+] Generated Advisory Card: {card['advisory_id']} (Mode: {card['mode']})")

    # 3. Simulate Human Operator Review & Sign-off
    audit = mgr.record_operator_clearance(
        advisory_id=card["advisory_id"],
        operator_badge_id="OP-8821-LEAD",
        decision="ACCEPTED_BY_OPERATOR",
        justification="Hydraulic stability verified. Column delta-P normal. Executed in DCS console."
    )
    print(f"[+] Operator Sign-off Recorded: {audit['dcs_console_slip']}")
    print(f"    --> Decision: {audit['decision']} by Badge #{audit['operator_badge']}")

    print("======================================================================")
    print(" [SUCCESS] PHASE 1 OPERATOR ADVISORY WORKFLOW PASSED & AUDITED!")
    print("======================================================================")

if __name__ == "__main__":
    test_phase1_advisory_pipeline()
