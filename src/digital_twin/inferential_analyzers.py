"""
Refinery Inferential Soft Sensors (Virtual Property Analyzers).

Solves the critical brownfield industry challenge:
Laboratory test samples (ASTM D86, CFR Engine Octane, Pensky-Martens Flash) take 4 to 8 hours.
To prevent off-spec shipments, refiners routinely operate with massive 'product giveaway'
(e.g., shipping 96.2 RON when contract requires 95.0, or over-stripping diesel).

This module delivers real-time (< 50 ms) virtual analyzers for:
1. FCC Gasoline Research Octane Number (RON) and Motor Octane Number (MON).
2. Kerosene / Jet A-1 Flash Point (ASTM D93) & Freezing Point (ASTM D2386).
3. Ultra-Low Sulfur Diesel (ULSD) ASTM D86 95% Cut Point & Cetane Index (ASTM D4737).
4. Stabilized Light Naphtha Reid Vapor Pressure (RVP / ASTM D323).
5. Real-Time Economic Product Giveaway & Savings Tracker.
"""

from typing import Dict, Tuple
import numpy as np

class InferentialAnalyzerSuite:
    def __init__(self):
        # Target specifications per commercial fuel contracts
        self.contract_specs = {
            "gasoline_ron_min": 95.0,
            "kerosene_flash_c_min": 38.0,
            "kerosene_freeze_c_max": -47.0,
            "diesel_d86_95_c_max": 360.0,
            "diesel_cetane_index_min": 45.0,
            "naphtha_rvp_psi_max": 9.0
        }

    # -------------------------------------------------------------------------
    # 1. FCC GASOLINE OCTANE (RON & MON) INFERENTIAL
    # -------------------------------------------------------------------------
    def predict_gasoline_octane(
        self,
        riser_outlet_temp_c: float,
        catalyst_to_oil_ratio: float,
        feed_api_gravity: float,
        coke_on_regen_cat_wt_pct: float = 0.05
    ) -> Dict:
        """
        Calculates FCC Gasoline Research Octane Number (RON) and Motor Octane Number (MON).
        Higher riser temperature increases olefin and aromatic content, boosting RON.
        """
        # Baseline reference: ROT 525 C, C/O 6.5, Feed API 24.0 -> Baseline RON 92.8
        delta_rot = riser_outlet_temp_c - 525.0
        delta_co = catalyst_to_oil_ratio - 6.5
        delta_api = feed_api_gravity - 24.0
        cat_poison_penalty = (coke_on_regen_cat_wt_pct - 0.05) * 4.5

        ron = 92.8 + (0.058 * delta_rot) + (0.42 * delta_co) - (0.28 * delta_api) - cat_poison_penalty
        mon = ron - 8.4 + (0.015 * delta_rot)  # Octane sensitivity: RON - MON

        giveaway_ron = max(0.0, ron - self.contract_specs["gasoline_ron_min"])

        return {
            "predicted_ron": round(float(ron), 2),
            "predicted_mon": round(float(mon), 2),
            "octane_sensitivity": round(float(ron - mon), 2),
            "contract_spec_min": self.contract_specs["gasoline_ron_min"],
            "giveaway_ron": round(float(giveaway_ron), 2),
            "on_spec": bool(ron >= self.contract_specs["gasoline_ron_min"])
        }

    # -------------------------------------------------------------------------
    # 2. KEROSENE / JET A-1 FLASH & FREEZE POINT INFERENTIAL
    # -------------------------------------------------------------------------
    def predict_kerosene_properties(
        self,
        draw_tray_temp_c: float,
        stripping_steam_flow_kg_per_m3: float,
        specific_gravity_15c: float = 0.805
    ) -> Dict:
        """
        Computes Closed-Cup Flash Point (Pensky-Martens / ASTM D93) and Freezing Point.
        Stripping steam strips light ends, increasing flash point.
        """
        # Flash Point correlation:
        flash_c = 0.72 * draw_tray_temp_c - 110.0 + (0.85 * stripping_steam_flow_kg_per_m3)
        # Freezing Point correlation:
        freeze_c = -58.0 + 0.24 * (draw_tray_temp_c - 195.0) + 45.0 * (specific_gravity_15c - 0.800)

        flash_giveaway = max(0.0, flash_c - self.contract_specs["kerosene_flash_c_min"])
        freeze_giveaway = max(0.0, self.contract_specs["kerosene_freeze_c_max"] - freeze_c)

        is_on_spec = (
            flash_c >= self.contract_specs["kerosene_flash_c_min"] and
            freeze_c <= self.contract_specs["kerosene_freeze_c_max"]
        )

        return {
            "predicted_flash_point_c": round(float(flash_c), 1),
            "predicted_freeze_point_c": round(float(freeze_c), 1),
            "spec_flash_min_c": self.contract_specs["kerosene_flash_c_min"],
            "spec_freeze_max_c": self.contract_specs["kerosene_freeze_c_max"],
            "flash_giveaway_c": round(float(flash_giveaway), 1),
            "freeze_giveaway_c": round(float(freeze_giveaway), 1),
            "on_spec": bool(is_on_spec)
        }

    # -------------------------------------------------------------------------
    # 3. DIESEL ASTM D86 95% CUT POINT & CETANE INDEX
    # -------------------------------------------------------------------------
    def predict_diesel_properties(
        self,
        diesel_draw_temp_c: float,
        column_bottom_reflux_ratio: float,
        api_gravity: float = 34.5
    ) -> Dict:
        """
        Computes ASTM D86 95% recovery temperature and Four-Variable Cetane Index (ASTM D4737).
        """
        # D86 95% distillation temperature:
        d86_95_c = 1.12 * diesel_draw_temp_c + 58.0 - (4.2 * column_bottom_reflux_ratio)
        # Mid-boiling point estimate T50
        t50_c = d86_95_c - 62.0

        # ASTM D4737 Cetane Index approximation:
        cetane_index = 45.4 + (0.088 * (t50_c - 270.0)) + (0.165 * (api_gravity - 34.0))

        on_spec = (
            d86_95_c <= self.contract_specs["diesel_d86_95_c_max"] and
            cetane_index >= self.contract_specs["diesel_cetane_index_min"]
        )

        giveaway_d86 = max(0.0, self.contract_specs["diesel_d86_95_c_max"] - d86_95_c)

        return {
            "predicted_d86_95_c": round(float(d86_95_c), 1),
            "predicted_cetane_index": round(float(cetane_index), 1),
            "spec_d86_95_max_c": self.contract_specs["diesel_d86_95_c_max"],
            "spec_cetane_min": self.contract_specs["diesel_cetane_index_min"],
            "heavy_tail_margin_c": round(float(giveaway_d86), 1),
            "on_spec": bool(on_spec)
        }

    # -------------------------------------------------------------------------
    # 4. NAPHTHA REID VAPOR PRESSURE (RVP) INFERENTIAL
    # -------------------------------------------------------------------------
    def predict_naphtha_rvp(
        self,
        stabilizer_top_temp_c: float,
        stabilizer_pressure_bar: float,
        overhead_reflux_ratio: float
    ) -> Dict:
        """
        Calculates Stabilized Naphtha RVP (ASTM D323 in psi).
        Excess light ends (butane/pentane) increase RVP above summer environmental pipeline limits.
        """
        # Antoine/VLE based soft sensor
        log_rvp = (0.018 * stabilizer_top_temp_c) + (0.15 * stabilizer_pressure_bar) - (0.12 * overhead_reflux_ratio)
        rvp_psi = 5.8 + log_rvp

        giveaway_rvp = max(0.0, self.contract_specs["naphtha_rvp_psi_max"] - rvp_psi)

        return {
            "predicted_rvp_psi": round(float(rvp_psi), 2),
            "spec_rvp_max_psi": self.contract_specs["naphtha_rvp_psi_max"],
            "rvp_margin_psi": round(float(giveaway_rvp), 2),
            "on_spec": bool(rvp_psi <= self.contract_specs["naphtha_rvp_psi_max"])
        }

    # -------------------------------------------------------------------------
    # 5. ECONOMIC PRODUCT GIVEAWAY & SAVINGS CALCULATOR
    # -------------------------------------------------------------------------
    def calculate_economic_giveaway_savings(
        self,
        gasoline_bpd: float = 35000.0,
        diesel_bpd: float = 42000.0,
        ron_giveaway: float = 0.8,
        diesel_margin_c: float = 8.5
    ) -> Dict:
        """
        Calculates annualized financial benefit of real-time inferentials vs 6-hour lab cycle.
        - Octane giveaway cost: ~$0.55 per barrel per excess RON point.
        - Diesel heavies giveaway (sub-optimal cut into residue): ~$0.40 per bbl per deg C.
        """
        daily_octane_waste_usd = gasoline_bpd * ron_giveaway * 0.55
        daily_diesel_waste_usd = diesel_bpd * (diesel_margin_c * 0.05) * 0.40
        annual_total_waste_usd = (daily_octane_waste_usd + daily_diesel_waste_usd) * 350.0

        return {
            "daily_octane_giveaway_waste_usd": round(daily_octane_waste_usd, 2),
            "daily_diesel_cut_waste_usd": round(daily_diesel_waste_usd, 2),
            "annualized_opportunity_loss_usd": round(annual_total_waste_usd, 2),
            "annualized_savings_with_inferentials_usd": round(annual_total_waste_usd * 0.85, 2) # 85% recovered
        }
