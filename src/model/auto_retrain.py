"""
Autonomous Active Learning & Retraining Pipeline.
Monitors operator feedback (ground-truth confirmations and false alarms)
and automatically schedules incremental model fine-tuning when sufficient feedback accumulates.
"""

import json
import subprocess
import sys
from pathlib import Path

FEEDBACK_FILE = Path("data/operator_feedback.jsonl")
RETRAIN_THRESHOLD = 5  # Retrain after 5 new verified operator tickets

def evaluate_retraining_trigger():
    if not FEEDBACK_FILE.exists():
        print("[*] No operator feedback recorded yet. Model is up to date.")
        return {"trigger_retrain": False, "verified_samples": 0}

    with open(FEEDBACK_FILE, "r") as f:
        feedback_lines = [json.loads(line) for line in f if line.strip()]

    confirmed_alarms = [entry for entry in feedback_lines if entry.get("operator_action") == "ACKNOWLEDGED_TRUE_ALERT"]
    dismissed_alarms = [entry for entry in feedback_lines if entry.get("operator_action") == "DISMISSED_FALSE_ALARM"]

    total_feedback = len(feedback_lines)
    print(f"[*] Operator Feedback Summary: {total_feedback} entries ({len(confirmed_alarms)} True Alarms, {len(dismissed_alarms)} False Alarms)")

    if total_feedback >= RETRAIN_THRESHOLD:
        print(f"[!] Retraining threshold ({RETRAIN_THRESHOLD} entries) reached! Triggering automated fine-tuning...")
        # Run incremental training loop
        cmd = [sys.executable, "src/model/trainer.py", "--epochs", "3", "--lr", "0.0005"]
        subprocess.run(cmd, check=True)
        return {"trigger_retrain": True, "status": "RETRAINED"}
    else:
        remaining = RETRAIN_THRESHOLD - total_feedback
        print(f"[*] Need {remaining} more operator reviews before next scheduled continuous retraining run.")
        return {"trigger_retrain": False, "remaining_reviews": remaining}

if __name__ == "__main__":
    evaluate_retraining_trigger()
