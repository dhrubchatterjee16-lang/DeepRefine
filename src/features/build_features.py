"""
Automated Feature Engineering Pipeline
Applies rolling statistical windows (mean, std, min, max, skew, kurtosis),
frequency-domain features (FFT bins), and labels failures within a 7-day horizon.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def run_feature_engineering(
    raw_sensor_path: Path = Path("data/raw/sensors/refinery_sensors.parquet"),
    maint_path: Path = Path("data/raw/maintenance_logs/work_orders.parquet"),
    output_dir: Path = Path("data/processed"),
    resample_freq: str = "10s",
    window_steps: int = 18,       # 18 steps * 10s = 3 minutes window
    step_stride: int = 3,         # Slide every 3 steps (30s)
    label_horizon_days: int = 7
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading raw sensor telemetry from: {raw_sensor_path}")
    df = pd.read_parquet(raw_sensor_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Uniform cadence
    print(f"Resampling to {resample_freq} uniform cadence...")
    df_resampled = df.resample(resample_freq).ffill().bfill()

    print(f"Computing rolling moments and FFT over {window_steps}-step windows...")
    tag_cols = [c for c in df_resampled.columns if c != "timestamp"]
    
    feature_dict = {}
    
    # Statistical features using efficient pandas rolling
    for col in tag_cols:
        series = df_resampled[col]
        r = series.rolling(window_steps)
        feature_dict[f"{col}_mean"] = r.mean()
        feature_dict[f"{col}_std"] = r.std()
        feature_dict[f"{col}_min"] = r.min()
        feature_dict[f"{col}_max"] = r.max()
        feature_dict[f"{col}_range"] = r.max() - r.min()

    # FFT Spectral energy on vibration tags
    vib_tags = [c for c in tag_cols if "Vibration" in c or "vib" in c.lower()]
    for col in vib_tags:
        series = df_resampled[col]
        # Energy in upper frequency bands
        fft_energy = series.rolling(window_steps).apply(
            lambda w: np.sum(np.abs(np.fft.rfft(w)[1:])**2) if len(w) == window_steps else 0.0,
            raw=True
        )
        feature_dict[f"{col}_fft_energy"] = fft_energy

    features_df = pd.DataFrame(feature_dict, index=df_resampled.index)
    
    # Stride downsampling to avoid excessive sample overlap
    features_df = features_df.iloc[window_steps::step_stride].dropna()
    print(f"Engineered feature matrix shape: {features_df.shape}")

    # Label generation: calculate true RUL directly from dataset if present, or within a 30-cycle failure horizon
    labels = []
    rul_hours = []

    if "rul_cycles" in df.columns:
        print("Using ground-truth run-to-failure degradation cycles...")
        rul_series = df.loc[features_df.index, "rul_cycles"].values
        for r in rul_series:
            # 30-cycle early warning horizon (standard benchmark)
            has_fail = 1 if r <= 30 else 0
            labels.append(has_fail)
            rul_hours.append(float(r))
    else:
        print("Calculating failure horizon from work orders...")
        failure_times_ns = [pd.Timestamp(f).value for f in maint_df["failure_time"]]
        horizon_ns = pd.Timedelta(days=label_horizon_days).value

        for ts in features_df.index:
            ts_ns = pd.Timestamp(ts).value
            diffs = [f_ns - ts_ns for f_ns in failure_times_ns if f_ns > ts_ns]
            if diffs:
                min_diff_ns = min(diffs)
                has_fail = 1 if min_diff_ns <= horizon_ns else 0
                rul = min_diff_ns / (1e9 * 3600.0)
            else:
                has_fail = 0
                rul = label_horizon_days * 24.0

            labels.append(has_fail)
            rul_hours.append(min(rul, label_horizon_days * 24.0))

    label_df = pd.DataFrame({
        "failure_label": labels,
        "rul_hours": rul_hours
    }, index=features_df.index)

    feat_out = output_dir / "features.parquet"
    label_out = output_dir / "failure_label.parquet"

    features_df.to_parquet(feat_out)
    label_df.to_parquet(label_out)

    pos_rate = np.mean(labels) * 100.0
    print(f"Saved features to {feat_out}")
    print(f"Saved labels to {label_out}")
    print(f"Summary: {len(label_df):,} samples, Positive failure rate: {pos_rate:.2f}%")

if __name__ == "__main__":
    run_feature_engineering()
