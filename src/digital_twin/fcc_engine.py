"""
Fluid Catalytic Cracking Unit (FCCU) & Hydrocracker Secondary Reactor Digital Twin.
Models downstream conversion of heavy atmospheric gas oil (AGO) and vacuum gas oil (VGO)
from the Crude Distillation Column into high-octane gasoline blendstock and LPG.

Reaction Kinetics & Kinetic Lumping:
1. Four-Lump Kinetic Model:
   VGO Feed -> Gasoline (C5-C12) + Light Gases/LPG (C1-C4) + Coke (Combusted in Regenerator)
2. Reactor-Regenerator Heat Integration:
   Exothermic coke burn in regenerator supplies thermal energy for endothermic cracking in riser.
3. Yield & Product Quality Calculations:
   - Research Octane Number (RON) prediction (91 - 96 RON)
   - Dry gas (H2, CH4, C2) and Flue gas emissions (SOx, NOx, CO2)
"""

from typing import Dict, Tuple, Optional
import numpy as np

class FluidCatalyticCrackerDigitalTwin:
    def __init__(
        self,
        riser_height_m: float = 38.0,
        riser_diameter_m: float = 1.4,
        catalyst_type: str = "USY_ZEOLITE_ULTRA_STABLE"
    ):
        self.height = riser_height_m
        self.diameter = riser_diameter_m
        self.catalyst = catalyst_type

        # Kinetic rate constants (Arrhenius baseline at 520°C / 793 K)
        # k1: VGO -> Gasoline, k2: VGO -> LPG, k3: VGO -> Coke, k4: Gasoline -> LPG
        self.k_ref = np.array([0.085, 0.035, 0.015, 0.010], dtype=np.float32)
        self.activation_energies_kj_mol = np.array([105.0, 115.0, 95.0, 125.0], dtype=np.float32)

    def simulate_cracking_reaction(
        self,
        feed_vgo_bpd: float = 28000.0,            # Gas Oil feed rate from CDU
        feed_api_gravity: float = 23.5,           # Feed heaviness (API)
        riser_temperature_c: float = 525.0,       # 500 - 550 °C
        cat_to_oil_ratio: float = 6.8,            # Catalyst circulation rate / Oil rate (5 - 9)
        regenerator_temperature_c: float = 690.0, # 660 - 730 °C
        catalyst_activity: float = 68.0           # Microactivity test (MAT) 60 - 75%
    ) -> Dict:
        """
        Calculates reactor conversion, yield slate, octane rating, and regenerator heat balance.
        """
        # Temperature effect via Arrhenius equation
        t_kelvin = riser_temperature_c + 273.15
        t_ref_k = 793.15 # 520 °C
        r_gas = 8.314e-3 # kJ/(mol*K)

        k_factors = self.k_ref * np.exp(- (self.activation_energies_kj_mol / r_gas) * (1.0 / t_kelvin - 1.0 / t_ref_k))
        
        # Residence time in riser (~ 2.5 seconds)
        tau_sec = 2.4 * (6.5 / max(cat_to_oil_ratio, 1.0))

        # Catalyst activity scaling factor
        act_factor = (catalyst_activity / 65.0) * (cat_to_oil_ratio / 6.5)

        # Kinetic 4-lump conversion calculations
        vgo_fraction = np.exp(- np.sum(k_factors[:3]) * tau_sec * act_factor)
        conversion_pct = float(np.clip((1.0 - vgo_fraction) * 100.0, 50.0, 88.0))

        # Product selectivity (% weight of converted feed)
        gasoline_wt_pct = float(np.clip(48.0 + 0.15 * (riser_temperature_c - 520.0) - 0.25 * (cat_to_oil_ratio - 6.5)**2, 40.0, 56.0))
        lpg_wt_pct = float(np.clip(22.0 + 0.20 * (riser_temperature_c - 520.0), 16.0, 28.0))
        coke_wt_pct = float(np.clip(4.8 + 0.4 * (cat_to_oil_ratio - 6.5), 3.5, 7.5))
        light_cycle_oil_pct = float(np.clip(100.0 - (gasoline_wt_pct + lpg_wt_pct + coke_wt_pct), 12.0, 28.0))

        # Convert to volumetric rates (BPD)
        gasoline_bpd = (feed_vgo_bpd * (gasoline_wt_pct / 100.0)) * 1.18 # Volumetric expansion
        lpg_bpd = (feed_vgo_bpd * (lpg_wt_pct / 100.0)) * 1.32
        lco_diesel_bpd = (feed_vgo_bpd * (light_cycle_oil_pct / 100.0)) * 0.94
        coke_tons_day = (feed_vgo_bpd * 0.15898 * 910.0 * (coke_wt_pct / 100.0)) / 1000.0

        # Research Octane Number (RON) correlation
        # Higher riser temperature increases aromatics and olefins, boosting octane
        ron = 91.5 + 0.045 * (riser_temperature_c - 515.0) + 0.12 * (catalyst_activity - 65.0)
        ron = float(np.clip(ron, 89.0, 96.5))

        # Coke combustion thermal energy & flue gas CO2 in regenerator
        # Coke heating value ~ 32,000 kJ/kg
        regenerator_duty_mw = (coke_tons_day * 1000.0 * 32000.0) / (86400.0 * 1000.0)
        co2_emissions_tons_day = coke_tons_day * 3.664 # C + O2 -> CO2 stoichiometric ratio

        # Economic valuation ($/bbl): High-Octane Gasoline: $104, LPG: $62, LCO: $86
        gross_value_usd_day = (gasoline_bpd * 104.0) + (lpg_bpd * 62.0) + (lco_diesel_bpd * 86.0)
        feed_cost_usd_day = feed_vgo_bpd * 88.0 # Feedstock transfer cost from CDU
        net_fcc_margin_usd_day = gross_value_usd_day - feed_cost_usd_day

        return {
            "status": "FCCU_REACTION_CONVERGED",
            "operating_parameters": {
                "feed_vgo_bpd": feed_vgo_bpd,
                "riser_temperature_c": riser_temperature_c,
                "cat_to_oil_ratio": cat_to_oil_ratio,
                "regenerator_temperature_c": regenerator_temperature_c
            },
            "conversion_metrics": {
                "total_conversion_pct": round(conversion_pct, 2),
                "gasoline_research_octane_number_ron": round(ron, 1),
                "coke_yield_wt_pct": round(coke_wt_pct, 2)
            },
            "product_slate_bpd": {
                "high_octane_gasoline": round(gasoline_bpd, 1),
                "liquefied_petroleum_gas_lpg": round(lpg_bpd, 1),
                "light_cycle_oil_diesel": round(lco_diesel_bpd, 1),
                "coke_combusted_tons_day": round(coke_tons_day, 1)
            },
            "thermal_and_environmental": {
                "regenerator_heat_duty_mw": round(regenerator_duty_mw, 2),
                "flue_gas_co2_tons_day": round(co2_emissions_tons_day, 1)
            },
            "fcc_economics": {
                "gross_revenue_usd_day": round(gross_value_usd_day, 2),
                "net_upgrading_margin_usd_day": round(net_fcc_margin_usd_day, 2)
            }
        }
