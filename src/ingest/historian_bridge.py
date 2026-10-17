"""
Industrial Refinery Historian Interface (OSIsoft PI / Aspen InfoPlus.21 Bridge)
Simulates an enterprise historian reader following Purdue Model Level 3 DMZ security standards.
Provides exception & deadband compression to minimize bandwidth usage.
"""

from typing import Dict, List, Optional
import numpy as np
import time

class RefineryHistorianBridge:
    def __init__(self, deadband_pct: float = 0.005):
        """
        deadband_pct: Minimum fractional change (0.5%) required to record a new historian point
        (Boxcar/Swinging Door compression standard).
        """
        self.deadband_pct = deadband_pct
        self.last_recorded_values: Dict[str, float] = {}

    def apply_deadband_filter(self, tag_dict: Dict[str, float]) -> Dict[str, float]:
        """Filters out micro-noise to mimic real OSIsoft PI exception compression."""
        compressed = {}
        for tag, val in tag_dict.items():
            if tag not in self.last_recorded_values:
                self.last_recorded_values[tag] = val
                compressed[tag] = val
            else:
                last_val = self.last_recorded_values[tag]
                ref = max(abs(last_val), 1.0)
                if abs(val - last_val) / ref >= self.deadband_pct:
                    self.last_recorded_values[tag] = val
                    compressed[tag] = val
        return compressed
