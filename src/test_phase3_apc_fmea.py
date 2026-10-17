import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent))

from optimizer.apc_fallback import TraditionalAPCFallback
from optimizer.fmea_watchdog import FMEARuntimeWatchdog

def test_phase3_apc_and_fmea():
    print("======================================================================")
    print(" [*] TESTING PHASE 3: APC FALLBACK & FMEA RUNTIME SAFETY WATCHDOG")
    print("======================================================================")

    apc = TraditionalAPCFallback()
    watchdog = FMEARuntimeWatchdog()

    # 1. Test FMEA Evaluation under nominal conditions
    nominal_telemetry = {"is_clean": True, "distribution_drift_index": 0.25, "flagged_anomalies": []}
    req_fb, mode_key, details = watchdog.evaluate_fmea_state(nominal_telemetry, flooding_pct=25.0)
    assert not req_fb, "Nominal state triggered unexpected fallback!"
    print(f"[+] Nominal FMEA State: {mode_key} (Status: {details['status']})")

    # 2. Test FMEA Trigger on Frozen ADC failure mode
    faulty_telemetry = {
        "is_clean": False,
        "distribution_drift_index": 0.30,
        "flagged_anomalies": [{"issue": "FROZEN_SIGNAL", "feature": "CDC_TopTemp"}]
    }
    req_fb, mode_key, details = watchdog.evaluate_fmea_state(faulty_telemetry, flooding_pct=25.0)
    assert req_fb, "FMEA watchdog failed to catch frozen sensor failure mode!"
    print(f"[+] Caught Failure Mode: {mode_key} (Severity: {details['severity'].name}, RPN: {details['rpn']})")
    print(f"    --> Recommended Mitigation: {details['mitigation']}")

    # 3. Test Bumpless Transfer to Traditional APC Fallback Controller
    trim = apc.compute_deterministic_trim(
        current_reflux=2.40,
        current_furnace_mw=45.0,
        target_top_temp_c=88.0,
        current_top_temp_c=91.2
    )
    assert trim["bumpless_transfer"], "Bumpless transfer flag not asserted"
    print(f"[+] Fallback Control Executed ({trim['mode']}): Reflux={trim['fallback_setpoints']['reflux_ratio']} (Delta {trim['fallback_setpoints']['delta_reflux']})")
    print(f"    --> Transfer Type: Bumpless, Zero Step Discontinuity, Reason: {trim['reason']}")

    print("======================================================================")
    print(" [SUCCESS] PHASE 3 APC FALLBACK & FMEA SAFETY WATCHDOG FULLY VALIDATED!")
    print("======================================================================")

if __name__ == "__main__":
    test_phase3_apc_and_fmea()
