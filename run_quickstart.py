"""
DeepRefine: Master Quickstart Launcher.
Runs pre-flight validation and launches the ISA-101 High Performance HMI Console.
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 75)
    print("  DEEPREFINE: FULL-STACK REFINERY DIGITAL TWIN & ADVISORY PLATFORM")
    print("  Purdue Model Level 3 Supervisory RTO | Dynamic DMC | VDI 2048 DVR")
    print("=" * 75)
    
    root_dir = Path(__file__).resolve().parent
    sys.path.append(str(root_dir / "src"))

    print("\n[1/3] Running pre-flight system integrity checks...")
    try:
        from security.idmz_gateway import IndustrialDemilitarizedZone
        from optimizer.dynamic_mpc import DynamicDMCEngine
        from digital_twin.dvr_engine import RefineryDVREngine
        from digital_twin.inferential_analyzers import InferentialAnalyzerSuite
        from model.api670_pdm import API670MachineryDiagnosticEngine

        idmz = IndustrialDemilitarizedZone()
        dmc = DynamicDMCEngine()
        dvr = RefineryDVREngine()
        soft_sensors = InferentialAnalyzerSuite()
        api670 = API670MachineryDiagnosticEngine()
        print("  [✓] All 5 Core Industrial Pillars Initialized Successfully.")
    except Exception as e:
        print(f"  [!] Pre-flight check warning: {e}")

    print("\n[2/3] Verifying trained TorchScript models and scaler checkpoints...")
    ckpt_dir = root_dir / "checkpoints"
    expected = ["best_refinery_pm.pt", "edge/refinery_pm_edge.pt", "digital_twin/pinn_surrogate.pt", "optimizer/refinery_policy.pt"]
    for f in expected:
        p = ckpt_dir / f
        if p.exists():
            print(f"  [✓] Found {f} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [-] Note: {f} will be generated on first run if missing.")

    print("\n[3/3] Launching High-Performance ISA-101 Streamlit Dashboard...")
    print("  URL: http://localhost:8501")
    print("  Press Ctrl+C to terminate the console.\n")
    
    dashboard_path = root_dir / "src" / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main()
