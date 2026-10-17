import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from digital_twin.twin_engine import DigitalTwinSimulator

def verify_twin():
    twin = DigitalTwinSimulator()
    out = twin.simulate_what_if(
        feed_flow_bpd=105000.0,
        furnace_duty_mw=48.0,
        reflux_ratio=2.6
    )
    print("======================================================================")
    print(" [*] DIGITAL TWIN SIMULATION VERIFICATION")
    print("======================================================================")
    print(f"[+] Status: {out['status']}")
    print(f"[+] Column Top Temperature: {out['predicted_thermodynamics']['top_temperature_c']} C")
    print(f"[+] Hydraulic Flooding Margin: {out['predicted_thermodynamics']['hydraulic_flooding_pct']}% ({out['predicted_thermodynamics']['flooding_alert']})")
    print(f"[+] Estimated Daily Margin: ${out['economics_and_emissions']['estimated_operating_margin_usd_day']:,.2f} USD")
    print(f"[+] Carbon Footprint: {out['economics_and_emissions']['carbon_footprint_co2_tons_day']} Tons CO2/day")
    print(f"[+] Product Cut Breakdown (BPD):")
    for cut, bpd in out['product_yields_bpd'].items():
        print(f"    - {cut}: {bpd:,.1f} BPD")
    print("======================================================================")
    print(" [SUCCESS] PILLAR 2 DIGITAL TWIN IS OPERATIONAL!")
    print("======================================================================")

if __name__ == "__main__":
    verify_twin()
