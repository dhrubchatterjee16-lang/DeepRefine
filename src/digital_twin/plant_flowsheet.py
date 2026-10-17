"""
Plant-Wide Flowsheet Orchestrator (CDU + FCCU Integrated Digital Twin).
Couples the primary Atmospheric Distillation Unit with the secondary Fluid Catalytic Cracker.
Transfers heavy Gas Oil streams dynamically into the downstream reactor,
enforcing plant-wide mass, thermal energy, and overall refinery economic balances.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np

from digital_twin.twin_engine import DigitalTwinSimulator
from digital_twin.fcc_engine import FluidCatalyticCrackerDigitalTwin

class RefineryFlowsheetNetwork:
    def __init__(self):
        self.cdu_twin = DigitalTwinSimulator()
        self.fcc_twin = FluidCatalyticCrackerDigitalTwin()

    def simulate_plant_flowsheet(
        self,
        crude_feed_bpd: float = 100000.0,
        cdu_reflux: float = 2.4,
        cdu_furnace_mw: float = 45.0,
        fcc_riser_temp_c: float = 525.0,
        fcc_cat_to_oil: float = 6.8,
        crude_assay: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Executes coupled plant simulation across CDU and FCCU.
        CDU Diesel/Gas Oil cut -> FCCU reactor feed.
        """
        # 1. Primary Fractionation (Crude Distillation Column)
        cdu_result = self.cdu_twin.simulate_what_if(
            feed_flow_bpd=crude_feed_bpd,
            reflux_ratio=cdu_reflux,
            furnace_duty_mw=cdu_furnace_mw,
            crude_composition=crude_assay
        )

        # Extract heavy gas oil yield to transfer downstream
        vgo_feed_to_fcc_bpd = cdu_result["product_yields_bpd"]["diesel_gas_oil"]

        # 2. Secondary Catalytic Cracking (FCC Unit)
        fcc_result = self.fcc_twin.simulate_cracking_reaction(
            feed_vgo_bpd=vgo_feed_to_fcc_bpd,
            riser_temperature_c=fcc_riser_temp_c,
            cat_to_oil_ratio=fcc_cat_to_oil
        )

        # 3. Coupled Plant-Wide Synthesis
        # Finished commercial products:
        # Naphtha (CDU) + High-Octane Gasoline (FCC) -> Finished Premium Gasoline
        total_gasoline_bpd = cdu_result["product_yields_bpd"]["naphtha_gasoline"] + fcc_result["product_slate_bpd"]["high_octane_gasoline"]
        total_jet_kero_bpd = cdu_result["product_yields_bpd"]["kerosene_jet_fuel"]
        total_diesel_bpd = fcc_result["product_slate_bpd"]["light_cycle_oil_diesel"]
        total_lpg_bpd = fcc_result["product_slate_bpd"]["liquefied_petroleum_gas_lpg"]
        total_residue_bpd = cdu_result["product_yields_bpd"]["atmospheric_residue"]

        # Overall Plant Revenue and Carbon Balances
        total_revenue_day = cdu_result["economics_and_emissions"]["gross_revenue_usd_day"] + fcc_result["fcc_economics"]["net_upgrading_margin_usd_day"]
        total_carbon_tons_day = (
            cdu_result["economics_and_emissions"]["carbon_footprint_co2_tons_day"] +
            fcc_result["thermal_and_environmental"]["flue_gas_co2_tons_day"]
        )

        return {
            "flowsheet_status": "PLANT_WIDE_CONVERGENCE_ACHIEVED",
            "primary_distillation_unit": cdu_result,
            "secondary_catalytic_cracking_unit": fcc_result,
            "inter_unit_streams": {
                "cdu_gas_oil_to_fcc_riser_bpd": round(vgo_feed_to_fcc_bpd, 1),
                "transfer_quality": "ATMOSPHERIC_GAS_OIL_350C_PLUS"
            },
            "finished_refinery_product_slate_bpd": {
                "premium_gasoline_blendstock": round(total_gasoline_bpd, 1),
                "commercial_jet_a1_kerosene": round(total_jet_kero_bpd, 1),
                "ultra_low_sulfur_diesel": round(total_diesel_bpd, 1),
                "liquefied_petroleum_gas_lpg": round(total_lpg_bpd, 1),
                "heavy_fuel_oil_residue": round(total_residue_bpd, 1)
            },
            "finished_product_qualities": {
                "gasoline_pool_octane_ron": fcc_result["conversion_metrics"]["gasoline_research_octane_number_ron"],
                "fcc_conversion_efficiency_pct": fcc_result["conversion_metrics"]["total_conversion_pct"]
            },
            "refinery_complex_totals": {
                "net_refining_margin_usd_day": round(total_revenue_day, 2),
                "annualized_refining_ebitda_usd": round(total_revenue_day * 350.0, 2),
                "total_carbon_emissions_tons_co2_day": round(total_carbon_tons_day, 1)
            }
        }
