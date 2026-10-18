"""
LIMS (Laboratory Information Management System) Assay Reconciliation & Bias Update Engine.

Addresses LLM Council Finding:
Virtual soft sensors drift over operational cycles due to crude slate variation, column fouling,
and catalyst aging. Without automated lab assay feedback, soft sensors cause undetected off-spec product.

Features:
1. Ingestion of physical laboratory test assays (ASTM D86, Knock Engine RON, Pensky-Martens Flash).
2. Statistical Gross Error Filtering (Chauvenet's Criterion / 3-sigma outlier rejection for lab anomalies).
3. Exponential Bias Updating:
   bias(t) = (1 - alpha) * bias(t-1) + alpha * (y_lab - y_model(t_sample))
4. Dynamic online bias compensation applied to all continuous inferential properties.
5. Long-term calibration drift detector alerting technical services when model retuning is required.
"""

from typing import Dict, List, Optional, Tuple
import time
import numpy as np

class LIMSAssayReconciliationEngine:
    def __init__(self, filter_alpha: float = 0.25):
        self.alpha = filter_alpha  # Bias learning rate (0.15 - 0.35 typical for refinery LIMS)

        # Persistent soft sensor biases: y_corrected = y_model + bias
        self.active_biases = {
            "gasoline_ron": 0.0,
            "kerosene_flash_c": 0.0,
            "kerosene_freeze_c": 0.0,
            "diesel_d86_95_c": 0.0,
            "naphtha_rvp_psi": 0.0
        }

        # Laboratory measurement standard deviations (sigma from ASTM reproducibility tables)
        self.astm_reproducibility_sigma = {
            "gasoline_ron": 0.35,       # ASTM D2699 CFR knock engine reproducibility
            "kerosene_flash_c": 1.5,    # ASTM D93 Pensky-Martens closed-cup
            "kerosene_freeze_c": 1.2,   # ASTM D2386 freezing point
            "diesel_d86_95_c": 2.8,     # ASTM D86 automated distillation
            "naphtha_rvp_psi": 0.15     # ASTM D323 / D5191 mini-method RVP
        }

        # History log for drift auditing
        self.assay_audit_trail: List[Dict] = []

    def reconcile_lab_sample(
        self,
        property_name: str,
        lab_measured_value: float,
        model_prediction_at_sample_time: float,
        sample_tag_id: str = "LAB-SAMPLE-001"
    ) -> Dict:
        """
        Processes a newly returned LIMS lab assay result.
        Rejects gross lab entry errors, calculates innovation error,
        and smooths the persistent bias correction term.
        """
        if property_name not in self.active_biases:
            return {
                "status": "REJECTED_UNKNOWN_PROPERTY",
                "message": f"Property '{property_name}' is not registered in LIMS engine."
            }

        sigma = self.astm_reproducibility_sigma[property_name]
        raw_error = lab_measured_value - model_prediction_at_sample_time

        # 1. Gross Error Detection on Lab Assay (Reject 3.5-sigma outliers, e.g. transcription typo)
        z_score = abs(raw_error) / sigma
        if z_score > 3.5:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sample_tag_id": sample_tag_id,
                "property": property_name,
                "lab_value": lab_measured_value,
                "model_value": model_prediction_at_sample_time,
                "z_score": round(float(z_score), 2),
                "status": "GROSS_ERROR_REJECTED",
                "message": f"Lab value rejected: deviation {raw_error:.2f} exceeds 3.5-sigma reproducibility threshold."
            }
            self.assay_audit_trail.append(record)
            return record

        # 2. Exponential Bias Update
        old_bias = self.active_biases[property_name]
        new_bias = (1.0 - self.alpha) * old_bias + self.alpha * raw_error
        self.active_biases[property_name] = round(float(new_bias), 3)

        # 3. Model Drift Evaluation
        # If smoothed bias exceeds 2*sigma, flag model for technical services retuning
        drift_alert = bool(abs(new_bias) > (2.0 * sigma))

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_tag_id": sample_tag_id,
            "property": property_name,
            "lab_value": round(float(lab_measured_value), 2),
            "model_value": round(float(model_prediction_at_sample_time), 2),
            "raw_residual": round(float(raw_error), 2),
            "previous_bias": round(float(old_bias), 3),
            "updated_bias": self.active_biases[property_name],
            "z_score": round(float(z_score), 2),
            "model_drift_alert": drift_alert,
            "status": "ASSAY_RECONCILED_SUCCESS"
        }
        self.assay_audit_trail.append(record)
        return record

    def apply_bias_compensation(self, property_name: str, raw_model_prediction: float) -> Dict:
        """Applies real-time LIMS-calibrated bias to live inferential soft sensor predictions."""
        bias = self.active_biases.get(property_name, 0.0)
        compensated = raw_model_prediction + bias
        return {
            "property": property_name,
            "raw_model_val": round(float(raw_model_prediction), 2),
            "active_lims_bias": round(float(bias), 3),
            "compensated_val": round(float(compensated), 2)
        }
