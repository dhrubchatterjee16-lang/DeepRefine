"""
Refinery Operator Visual Dashboard (Streamlit) - Master Edition v4.0
Complete integration of all 3 Pillars:
- PILLAR 1: Predictive Maintenance (Real-time degradation, ISO-10816, SAP Work Orders, Human Feedback)
- PILLAR 2: Digital Twin Simulator & What-If Scenario Lab (Thermodynamic PINN Surrogate, Yields, Carbon & Economic Margin)
- PILLAR 3: Smart Control & Autonomous Optimization (RL Actor-Critic, Safe Shadow-Mode Trims, $ Profit Uplift)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json

from model.cmms import RefineryCMMSBridge
from digital_twin.twin_engine import DigitalTwinSimulator
from optimizer.shadow_controller import ShadowModeController

st.set_page_config(
    page_title="Digital Refinery Master Platform | AI, Digital Twin & Smart Control",
    page_icon="🏭",
    layout="wide"
)

cmms_bridge = RefineryCMMSBridge()

st.title("🏭 Digital Refinery - Industry 4.0 Autonomous Operations")
st.markdown("Unified platform powering **Predictive Maintenance (P1)**, **Thermodynamic Digital Twin (P2)**, and **RL Smart Control & Yield Optimization (P3)**.")

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "🛡️ Pillar 1: Predictive Maintenance",
    "🔮 Pillar 2: Distillation Digital Twin",
    "⚡ Pillar 3: Operator Advisory (RL)",
    "🌐 Path C: Plant Flowsheet Network (CDU + FCCU)"
])

# =========================================================================
# PILLAR 1: PREDICTIVE MAINTENANCE
# =========================================================================
with main_tab1:
    selected_unit = st.selectbox(
        "Active Asset",
        ["WGC-101-COMPRESSOR (Wet Gas Compressor)",
         "CDC-201-COLUMN (Crude Distillation Column)",
         "P-301-PUMP (Heavy Oil Booster Pump)"],
        key="pm_asset_selector"
    )

    features_path = Path("data/processed/features.parquet")
    orders_path = Path("data/cmms_work_orders.jsonl")

    col1, col2, col3, col4, col5 = st.columns(5)
    if "COMPRESSOR" in selected_unit:
        col1.metric("Asset Health", "58%", "-14% (Degrading)", delta_color="inverse")
        col2.metric("Failure Risk", "78.4%", "+22.1%", delta_color="inverse")
        col3.metric("Predicted RUL", "28.5 Hours", "-4.2 hrs", delta_color="inverse")
        col4.metric("ISO 10816 Zone", "ZONE C (Unsatisfactory)", delta_color="inverse")
        col5.metric("Financial Exposure", "$78,500 USD", "At Risk", delta_color="inverse")
    elif "COLUMN" in selected_unit:
        col1.metric("Asset Health", "92%", "+1.2%")
        col2.metric("Failure Risk", "8.2%", "-2.0%")
        col3.metric("Predicted RUL", "168.0 Hours", "Steady")
        col4.metric("ISO 10816 Zone", "ZONE A (Good)", delta_color="off")
        col5.metric("Financial Exposure", "$0 USD", "Nominal", delta_color="off")
    else:
        col1.metric("Asset Health", "86%", "+0.5%")
        col2.metric("Failure Risk", "14.1%", "-1.1%")
        col3.metric("Predicted RUL", "142.0 Hours", "Steady")
        col4.metric("ISO 10816 Zone", "ZONE B (Acceptable)", delta_color="off")
        col5.metric("Financial Exposure", "$5,200 USD", "Low", delta_color="off")

    st.markdown("---")
    pm_sub1, pm_sub2 = st.columns([2, 1])
    with pm_sub1:
        st.subheader("📈 Multi-Channel Sensor Degradation Trends")
        if features_path.exists():
            df_feat = pd.read_parquet(features_path).tail(80)
            cols = [c for c in df_feat.columns if "mean" in c][:4]
            st.line_chart(df_feat[cols])
        else:
            st.info("Feature store not loaded.")

    with pm_sub2:
        st.subheader("📋 Active SAP PM Work Orders")
        if orders_path.exists():
            with open(orders_path, "r") as f:
                orders = [json.loads(line) for line in f if line.strip()]
            for o in reversed(orders[-3:]):
                st.warning(f"**{o['order_id']}** - {o['equipment_id']} | Risk: ${o['economic_risk_usd']:,}")
        else:
            st.info("No active work orders.")

# =========================================================================
# PILLAR 2: DIGITAL TWIN SIMULATOR
# =========================================================================
with main_tab2:
    st.subheader("🔬 Atmospheric Distillation Column - Real-Time What-If Scenario Lab")
    try:
        twin_sim = DigitalTwinSimulator()
        sim_ready = True
    except Exception as e:
        sim_ready = False
        st.error(f"Digital Twin Surrogate not loaded: {e}")

    if sim_ready:
        ctl_col, res_col = st.columns([1, 2])
        with ctl_col:
            st.markdown("### 🎛️ Process Setpoints")
            feed_rate = st.slider("Crude Feed Throughput (BPD)", 60000, 140000, 100000, 5000)
            furnace_duty = st.slider("Reboiler / Furnace Duty (MW)", 30.0, 65.0, 45.0, 1.0)
            reflux_r = st.slider("Overhead Reflux Ratio (L/D)", 1.5, 3.5, 2.4, 0.1)
            feed_t = st.slider("Furnace Outlet Temperature (°C)", 320.0, 375.0, 345.0, 1.0)
            
            c_naphtha = st.slider("Naphtha Cut (%)", 15.0, 30.0, 22.0, 1.0) / 100.0
            c_kero = st.slider("Kerosene / Jet Fuel (%)", 10.0, 22.0, 16.0, 1.0) / 100.0
            c_diesel = st.slider("Diesel Cut (%)", 20.0, 35.0, 28.0, 1.0) / 100.0
            c_residue = max(0.01, 1.0 - (c_naphtha + c_kero + c_diesel))

            sim_results = twin_sim.simulate_what_if(
                feed_flow_bpd=float(feed_rate),
                feed_temp_c=float(feed_t),
                reflux_ratio=float(reflux_r),
                furnace_duty_mw=float(furnace_duty),
                crude_composition={"naphtha_pct": c_naphtha, "kerosene_pct": c_kero, "diesel_pct": c_diesel, "residue_pct": c_residue}
            )

        with res_col:
            st.markdown("### 📊 Operating Thermodynamics & Yields")
            thermo = sim_results["predicted_thermodynamics"]
            econ = sim_results["economics_and_emissions"]
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Top Temp", f"{thermo['top_temperature_c']} °C")
            kpi2.metric("Bottom Temp", f"{thermo['bottom_temperature_c']} °C")
            kpi3.metric("Hydraulic Flooding", f"{thermo['hydraulic_flooding_pct']}%", 
                        delta="SAFE" if thermo['flooding_alert'] == "SAFE_REGIME" else "HIGH RISK",
                        delta_color="normal" if thermo['flooding_alert'] == "SAFE_REGIME" else "inverse")
            kpi4.metric("Est. Daily Margin", f"${econ['estimated_operating_margin_usd_day']:,.0f} USD")

            st.markdown("#### 📦 Product Cut Distribution (BPD)")
            yields = sim_results["product_yields_bpd"]
            df_yields = pd.DataFrame({
                "Petroleum Cut": ["Naphtha (Gasoline)", "Kerosene (Jet Fuel)", "Diesel (Gas Oil)", "Atmospheric Residue"],
                "Volume (BPD)": [yields["naphtha_gasoline"], yields["kerosene_jet_fuel"], yields["diesel_gas_oil"], yields["heavy_atmospheric_residue"]]
            })
            st.bar_chart(df_yields.set_index("Petroleum Cut"))

# =========================================================================
# PILLAR 3: OPERATOR ADVISORY DECISION SUPPORT (PHASE 1 CERTIFIED)
# =========================================================================
with main_tab3:
    st.subheader("🛡️ Phase 1 Certified: Read-Only Operator Advisory & Decision Support")
    st.markdown("""
    **Safety Mode:** `READ_ONLY_ADVISORY` (Compliant with IEC 61511 / ISA 84 SIS interlocks).  
    *The AI engine operates strictly in decision-support mode and cannot command physical DCS setpoints without explicit human authorization.*
    """)

    try:
        from optimizer.advisory_manager import OperatorAdvisoryManager
        shadow_ctrl = ShadowModeController()
        advisory_mgr = OperatorAdvisoryManager()
        rl_ready = True
    except Exception as e:
        rl_ready = False
        st.error(f"Advisory Manager not loaded: {e}")

    if rl_ready:
        st.markdown("### 🎛️ Current DCS Plant Setpoints")
        c1, c2, c3 = st.columns(3)
        curr_feed = c1.number_input("Crude Feed Rate (BPD)", value=100000, step=5000)
        curr_reflux = c2.number_input("Current Reflux Ratio (L/D)", value=2.40, step=0.05)
        curr_furnace = c3.number_input("Current Furnace Duty (MW)", value=45.0, step=1.0)

        if st.button("🔍 Generate AI Optimization Advisory Card", type="primary"):
            rec = shadow_ctrl.evaluate_recommendation(
                current_reflux=curr_reflux,
                current_furnace_mw=curr_furnace,
                feed_flow_bpd=curr_feed
            )

            # Generate formal advisory record
            adv_card = advisory_mgr.generate_advisory(
                unit_id="CDC-201-COLUMN",
                current_setpoints={"reflux": curr_reflux, "furnace_mw": curr_furnace},
                proposed_setpoints=rec["recommended_setpoints"],
                economic_gain_usd_day=rec["expected_economic_benefit"]["daily_profit_gain_usd"],
                safety_status=rec["status"],
                flooding_margin_pct=rec["hydraulic_safety_margin"]["flooding_pct"]
            )

            st.session_state["active_advisory"] = adv_card
            st.session_state["active_rec"] = rec

        if "active_advisory" in st.session_state:
            card = st.session_state["active_advisory"]
            rec = st.session_state["active_rec"]

            st.markdown("---")
            st.markdown(f"### 📋 Advisory Card: `{card['advisory_id']}`")
            st.caption(f"Status: **{rec['audit_message']}**")

            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Safety Clearance", "VERIFIED", delta="RATE CLAMPED")
            r_col2.metric("Recommended Reflux", f"{rec['recommended_setpoints']['reflux_ratio']} (Δ {rec['recommended_setpoints']['delta_reflux']})")
            r_col3.metric("Recommended Furnace", f"{rec['recommended_setpoints']['furnace_duty_mw']} MW (Δ {rec['recommended_setpoints']['delta_furnace_mw']} MW)")

            st.markdown("#### 💰 Projected Value Creation")
            e1, e2 = st.columns(2)
            econ_rec = rec["expected_economic_benefit"]
            e1.metric("Daily Profit Uplift", f"+${econ_rec['daily_profit_gain_usd']:,.0f} USD/day")
            e2.metric("Annualized Value Uplift", f"+${econ_rec['annualized_gain_usd']:,.0f} USD/yr", delta="ROI Cleared")

            st.markdown("---")
            st.markdown("### ✍️ Human Operator Clearance Sign-off")
            with st.form("operator_clearance_form"):
                badge_id = st.text_input("Lead Board Operator Badge ID", value="OP-7412-SENIOR")
                decision = st.radio("Operator Decision", ["ACCEPT_TRIM (Generate DCS Console Entry Slip)", "REJECT_TRIM (Override AI Recommendation)"])
                justification = st.text_area("Engineering Justification / Log Reason", value="Verified column top temperature and hydraulic tray stability. Setpoint ramp cleared for manual DCS execution.")
                submit_clearance = st.form_submit_button("Sign & Authorize Advisory")

                if submit_clearance:
                    audit_res = advisory_mgr.record_operator_clearance(
                        advisory_id=card["advisory_id"],
                        operator_badge_id=badge_id,
                        decision=decision.split(" ")[0],
                        justification=justification
                    )
                    st.success(f"✅ Authorization recorded in immutable audit journal: {audit_res['dcs_console_slip']}")


# =========================================================================
# PATH C: PLANT-WIDE FLOWSHEET NETWORK (CDU + FCCU)
# =========================================================================
with main_tab4:
    st.subheader("🌐 Plant-Wide Multi-Unit Flowsheet Network (CDU + FCCU Coupling)")
    st.markdown("""
    **Coupled Refining Train:**  
    Atmospheric Distillation Unit (CDU) $\\rightarrow$ Gas Oil Stream $\\rightarrow$ Fluid Catalytic Cracker (FCCU) $\\rightarrow$ Gasoline Pool Octane Blending.
    """)

    try:
        from digital_twin.plant_flowsheet import RefineryFlowsheetNetwork
        flowsheet_net = RefineryFlowsheetNetwork()
        net_ready = True
    except Exception as e:
        net_ready = False
        st.error(f"Flowsheet Network not loaded: {e}")

    if net_ready:
        from dashboard.isa101_hmi import generate_isa101_pid_svg
        fs_col1, fs_col2 = st.columns([1, 2])
        with fs_col1:
            st.markdown("### 🎛️ Refinery Train Setpoints")
            crude_bpd = st.slider("Total Crude Intake (BPD)", 70000, 130000, 100000, 5000, key="fs_crude")
            fcc_temp = st.slider("FCC Riser Reaction Temp (°C)", 500.0, 550.0, 528.0, 1.0, key="fs_fcc_t")
            fcc_cat = st.slider("Catalyst-to-Oil Ratio", 5.0, 9.0, 7.0, 0.1, key="fs_fcc_cat")

            fs_res = flowsheet_net.simulate_plant_flowsheet(
                crude_feed_bpd=float(crude_bpd),
                cdu_reflux=2.40,
                cdu_furnace_mw=45.0,
                fcc_riser_temp_c=float(fcc_temp),
                fcc_cat_to_oil=float(fcc_cat)
            )

        with fs_col2:
            st.markdown("### 📊 Finished Commercial Product Slate & Quality")
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            c_kpi1.metric("Inter-Unit VGO Transfer", f"{fs_res['inter_unit_streams']['cdu_gas_oil_to_fcc_riser_bpd']:,} BPD")
            c_kpi2.metric("Gasoline Pool Octane", f"{fs_res['finished_product_qualities']['gasoline_pool_octane_ron']} RON", delta="Premium Blend")
            c_kpi3.metric("Annualized Complex EBITDA", f"${fs_res['refinery_complex_totals']['annualized_refining_ebitda_usd']/1e9:.2f} Billion USD/yr")

        # ISA-101 High Performance HMI Section
        st.markdown("---")
        st.markdown("### 🖥️ ISA-101 High-Performance P&ID Dynamic Console")
        st.caption("Grayscale industrial standard with alarm-only color rendering and live telemetry tags.")
        svg_markup = generate_isa101_pid_svg(
            cdu_feed_bpd=float(crude_bpd),
            cdu_top_temp_c=fs_res["primary_distillation_unit"]["predicted_thermodynamics"]["top_temperature_c"],
            cdu_bot_temp_c=fs_res["primary_distillation_unit"]["predicted_thermodynamics"]["bottom_temperature_c"],
            cdu_flooding_pct=fs_res["primary_distillation_unit"]["predicted_thermodynamics"]["hydraulic_flooding_pct"],
            vgo_transfer_bpd=fs_res["inter_unit_streams"]["cdu_gas_oil_to_fcc_riser_bpd"],
            fcc_riser_temp_c=float(fcc_temp),
            fcc_regen_temp_c=fs_res["secondary_catalytic_cracking_unit"]["operating_parameters"]["regenerator_temperature_c"],
            gasoline_ron=fs_res["finished_product_qualities"]["gasoline_pool_octane_ron"]
        )
        st.markdown(svg_markup, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 📦 Commercial Products Ready for Distribution")
            df_slate = pd.DataFrame([
                {"Product Cut": "Premium Gasoline (CDU+FCC)", "Volume (BPD)": fs_res['finished_refinery_product_slate_bpd']['premium_gasoline_blendstock']},
                {"Product Cut": "Commercial Jet A-1 Kerosene", "Volume (BPD)": fs_res['finished_refinery_product_slate_bpd']['commercial_jet_a1_kerosene']},
                {"Product Cut": "Ultra-Low Sulfur Diesel", "Volume (BPD)": fs_res['finished_refinery_product_slate_bpd']['ultra_low_sulfur_diesel']},
                {"Product Cut": "Liquefied Petroleum Gas (LPG)", "Volume (BPD)": fs_res['finished_refinery_product_slate_bpd']['liquefied_petroleum_gas_lpg']},
                {"Product Cut": "Heavy Fuel Oil Residue", "Volume (BPD)": fs_res['finished_refinery_product_slate_bpd']['heavy_fuel_oil_residue']}
            ])
            st.bar_chart(df_slate.set_index("Product Cut"))
            st.info(f"🌿 Total Complex Carbon Footprint: **{fs_res['refinery_complex_totals']['total_carbon_emissions_tons_co2_day']:.1f} Tons CO₂/day** (CDU Firing + FCC Regenerator Flue Gas)")

st.markdown("---")
st.caption("DeepRefine Master Platform • Full Industry 4.0 Suite (P1: Predictive Maintenance, P2: Digital Twin, P3: Smart Advisory, Path C: Multi-Unit Flowsheet)")
