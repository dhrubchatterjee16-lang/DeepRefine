"""
Refinery Data Generator:
Simulates realistic continuous sensor streams for critical refinery units:
- Crude Distillation Column (Top/Bottom Temperatures, Column Pressure)
- Wet Gas Compressor (Vibration X/Y, Discharge Pressure, Bearing Temperature)
- Heavy Oil Pump (Vibration, Discharge Flow, Motor Power)

Simulates normal baseline degradation and intermittent anomaly/failure trajectories.
Saves data directly to Parquet format for testing and model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_refinery_telemetry(
    num_days: int = 30,
    sampling_rate_sec: int = 1,
    anomaly_intervals: list = None,
    output_dir: Path = Path("data/raw/sensors")
):
    output_dir.mkdir(parents=True, exist_ok=True)
    maint_dir = Path("data/raw/maintenance_logs")
    maint_dir.mkdir(parents=True, exist_ok=True)

    total_seconds = num_days * 24 * 3600
    # Downsample default frequency to 10s for dataset manageable size if generating many days
    step_sec = max(1, sampling_rate_sec)
    n_points = total_seconds // step_sec
    
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * step_sec) for i in range(n_points)]
    
    # Base signals with physical baseline levels & noise
    # CDC: Crude Distillation Column
    cdc_top_temp = 115.0 + np.random.normal(0, 1.2, n_points)      # Celsius
    cdc_bot_temp = 345.0 + np.random.normal(0, 2.5, n_points)      # Celsius
    cdc_pressure = 1.45 + np.random.normal(0, 0.03, n_points)      # Bar
    
    # Compressor
    comp_vib_x = 2.1 + np.random.normal(0, 0.15, n_points)         # mm/s RMS
    comp_vib_y = 2.0 + np.random.normal(0, 0.15, n_points)         # mm/s RMS
    comp_temp = 82.0 + np.random.normal(0, 1.0, n_points)          # Celsius
    comp_discharge_p = 18.2 + np.random.normal(0, 0.2, n_points)   # Bar

    # Pump
    pump_flow = 420.0 + np.random.normal(0, 5.0, n_points)         # m3/h
    pump_vib = 1.8 + np.random.normal(0, 0.12, n_points)           # mm/s

    # Inject progressive degradation leading to failures
    # Default failure events if none provided: Day 14 and Day 27
    if anomaly_intervals is None:
        anomaly_intervals = [
            {"failure_day": 14, "ramp_days": 4, "type": "compressor_bearing"},
            {"failure_day": 27, "ramp_days": 5, "type": "cdc_tray_fouling"}
        ]
    
    work_orders = []
    
    for anom in anomaly_intervals:
        fail_idx = int((anom["failure_day"] * 24 * 3600) / step_sec)
        ramp_len = int((anom["ramp_days"] * 24 * 3600) / step_sec)
        start_ramp = max(0, fail_idx - ramp_len)
        
        ramp = np.linspace(0, 1, fail_idx - start_ramp)
        
        if anom["type"] == "compressor_bearing":
            # Vibration spikes exponentially and temperature drifts upward
            comp_vib_x[start_ramp:fail_idx] += (ramp ** 2) * 9.5
            comp_vib_y[start_ramp:fail_idx] += (ramp ** 2) * 8.0
            comp_temp[start_ramp:fail_idx] += ramp * 28.0
            
            work_orders.append({
                "work_order_id": f"WO-{anom['failure_day']:03d}",
                "equipment_id": "WGC-101-COMPRESSOR",
                "failure_time": timestamps[fail_idx].isoformat(),
                "failure_mode": "BEARING_RUNAWAY_VIBRATION",
                "action_taken": "Replaced thrust bearing and re-aligned shaft"
            })
            
        elif anom["type"] == "cdc_tray_fouling":
            # Top temperature drops, bottom temperature surges, pressure drop builds up
            cdc_top_temp[start_ramp:fail_idx] -= ramp * 15.0
            cdc_bot_temp[start_ramp:fail_idx] += ramp * 22.0
            cdc_pressure[start_ramp:fail_idx] += ramp * 0.45
            
            work_orders.append({
                "work_order_id": f"WO-{anom['failure_day']:03d}",
                "equipment_id": "CDC-201-COLUMN",
                "failure_time": timestamps[fail_idx].isoformat(),
                "failure_mode": "TRAY_FOULING_DELTA_P",
                "action_taken": "Emergency wash cycle and antifoulant dosing increase"
            })

    telemetry_df = pd.DataFrame({
        "timestamp": timestamps,
        "CDC_TopTemp": cdc_top_temp,
        "CDC_BotTemp": cdc_bot_temp,
        "CDC_Pressure": cdc_pressure,
        "Comp_VibrationX": comp_vib_x,
        "Comp_VibrationY": comp_vib_y,
        "Comp_Temp": comp_temp,
        "Comp_DischargePres": comp_discharge_p,
        "Pump_Flow": pump_flow,
        "Pump_Vibration": pump_vib
    })

    file_out = output_dir / "refinery_sensors.parquet"
    telemetry_df.to_parquet(file_out, index=False)
    print(f"Generated {len(telemetry_df):,} telemetry points saved to: {file_out}")

    wo_df = pd.DataFrame(work_orders)
    wo_file = maint_dir / "work_orders.parquet"
    wo_df.to_parquet(wo_file, index=False)
    print(f"Generated {len(wo_df)} maintenance logs saved to: {wo_file}")

if __name__ == "__main__":
    generate_refinery_telemetry(num_days=30, sampling_rate_sec=10)
