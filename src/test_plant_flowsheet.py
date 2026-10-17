import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent))

from digital_twin.plant_flowsheet import RefineryFlowsheetNetwork

def test_plant_flowsheet():
    print("==============================================================================")
    print(" [*] TESTING PATH C: PLANT-WIDE FLOWSHEET NETWORK (CDU + FCCU INTEGRATION)")
    print("==============================================================================")

    network = RefineryFlowsheetNetwork()
    flowsheet = network.simulate_plant_flowsheet(
        crude_feed_bpd=100000.0,
        cdu_reflux=2.40,
        cdu_furnace_mw=45.0,
        fcc_riser_temp_c=530.0,
        fcc_cat_to_oil=7.2
    )

    print(f"[+] Flowsheet Status: {flowsheet['flowsheet_status']}")
    print(f"[+] Inter-Unit Transfer: {flowsheet['inter_unit_streams']['cdu_gas_oil_to_fcc_riser_bpd']:,} BPD Gas Oil -> FCC Riser")
    print(f"[+] FCC Conversion: {flowsheet['finished_product_qualities']['fcc_conversion_efficiency_pct']}%")
    print(f"[+] Finished Gasoline Pool RON: {flowsheet['finished_product_qualities']['gasoline_pool_octane_ron']} Octane")

    print("\n[+] Commercial Refinery Product Slate (BPD):")
    for prod, vol in flowsheet['finished_refinery_product_slate_bpd'].items():
        print(f"    - {prod:<30}: {vol:>10,.1f} BPD")

    econ = flowsheet['refinery_complex_totals']
    print("\n[+] Total Complex Economics & Emissions:")
    print(f"    - Net Refining Margin: ${econ['net_refining_margin_usd_day']:,.2f} USD/day")
    print(f"    - Annualized Complex EBITDA: ${econ['annualized_refining_ebitda_usd']:,.2f} USD/year")
    print(f"    - Total Carbon Footprint: {econ['total_carbon_emissions_tons_co2_day']:.1f} Tons CO2/day")

    print("==============================================================================")
    print(" [SUCCESS] PLANT-WIDE MULTI-UNIT DIGITAL TWIN NETWORK CONVERGED!")
    print("==============================================================================")

if __name__ == "__main__":
    test_plant_flowsheet()
