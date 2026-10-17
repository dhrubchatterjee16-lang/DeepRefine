"""
Digital Twin Real-Time Simulator & What-If Scenario Engine.
Provides high-speed queries on plant operating changes (crude blend switch, furnace duty adjustment, reflux tweaking).
Enforces physical and thermodynamic constraints while returning product cut yields and hydraulic flooding margins.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch

from digital_twin.pinn_surrogate import DistillationSurrogatePINN

class DigitalTwinSimulator:
    def __init__(self, checkpoint_path: Path = Path("checkpoints/digital_twin/pinn_surrogate.pt")):
        self.device = torch.device("cpu")
        self.model = DistillationSurrogatePINN(in_features=7, out_features=8)
        
        assert checkpoint_path.exists(), f"Missing Digital Twin checkpoint {checkpoint_path}"
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
        self.x_std = np.array(ckpt["x_std"], dtype=np.float32)
        self.y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
        self.y_std = np.array(ckpt["y_std"], dtype=np.float32)

    def simulate_what_if(
        self,
        feed_flow_bpd: float = 100000.0,
        feed_temp_c: float = 345.0,
        reflux_ratio: float = 2.4,
        furnace_duty_mw: float = 45.0,
        crude_composition: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Runs real-time neural surrogate inference on the refinery distillation column.
        Latency: < 1 millisecond.
        """
        if crude_composition is None:
            crude_composition = {
                "naphtha_pct": 0.22,
                "kerosene_pct": 0.16,
                "diesel_pct": 0.28,
                "residue_pct": 0.34
            }

        c1 = crude_composition.get("naphtha_pct", 0.22)
        c2 = crude_composition.get("kerosene_pct", 0.16)
        c3 = crude_composition.get("diesel_pct", 0.28)

        # Vector: [feed_flow, feed_temp, reflux, duty, c1, c2, c3]
        x_raw = np.array([feed_flow_bpd, feed_temp_c, reflux_ratio, furnace_duty_mw, c1, c2, c3], dtype=np.float32)
        x_norm = (x_raw - self.x_mean) / self.x_std

        x_tensor = torch.tensor(x_norm).unsqueeze(0)
        with torch.no_grad():
            y_norm = self.model(x_tensor).squeeze(0).numpy()

        y_pred = y_norm * self.y_std + self.y_mean

        top_temp = float(y_pred[0])
        bot_temp = float(y_pred[1])
        delta_p = float(y_pred[2])
        flooding_pct = float(np.clip(y_pred[3], 0.0, 100.0))

        # Enforce Mass Conservation post-processing
        raw_yields = np.maximum(y_pred[4:8], 0.0)
        total_yield = np.sum(raw_yields) or 1.0
        normalized_yields = raw_yields * (feed_flow_bpd / total_yield)

        # Revenue & Emission estimations
        # Indicative market prices ($/bbl): Naphtha: $85, Kerosene: $98, Diesel: $92, Residue: $55
        prices = np.array([85.0, 98.0, 92.0, 55.0])
        daily_revenue_usd = float(np.dot(normalized_yields, prices))
        fuel_cost_usd_day = furnace_duty_mw * 24.0 * 45.0 # ~$45 / MWh natural gas
        net_margin_usd_day = daily_revenue_usd - fuel_cost_usd_day
        carbon_emissions_ton_day = (furnace_duty_mw * 24.0 * 202.4) / 1000.0

        return {
            "status": "CONVERGED_DIGITAL_TWIN",
            "operating_conditions": {
                "feed_flow_bpd": feed_flow_bpd,
                "feed_temp_c": feed_temp_c,
                "reflux_ratio": reflux_ratio,
                "furnace_duty_mw": furnace_duty_mw,
                "crude_slate": crude_composition
            },
            "predicted_thermodynamics": {
                "top_temperature_c": round(top_temp, 1),
                "bottom_temperature_c": round(bot_temp, 1),
                "delta_pressure_bar": round(delta_p, 3),
                "hydraulic_flooding_pct": round(flooding_pct, 1),
                "flooding_alert": "CRITICAL_HYDRAULIC_LIMIT" if flooding_pct > 85.0 else "SAFE_REGIME"
            },
            "product_yields_bpd": {
                "naphtha_gasoline": round(float(normalized_yields[0]), 1),
                "kerosene_jet_fuel": round(float(normalized_yields[1]), 1),
                "diesel_gas_oil": round(float(normalized_yields[2]), 1),
                "atmospheric_residue": round(float(normalized_yields[3]), 1)
            },
            "economics_and_emissions": {
                "gross_revenue_usd_day": round(daily_revenue_usd, 2),
                "estimated_operating_margin_usd_day": round(net_margin_usd_day, 2),
                "carbon_footprint_co2_tons_day": round(carbon_emissions_ton_day, 1)
            }
        }
