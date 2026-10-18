"""
Data Validation and Reconciliation (DVR) Engine per VDI 2048.

Refinery Mass and Energy Conservation Law Enforcement:
In brownfield plants, raw instrumentation contains systematic offsets, density shifts,
and sensor drifts that violate first-principles conservation.
This engine:
1. Solves constrained Weighted Least Squares (WLS) optimization:
   min_x (x - y)^T Sigma^-1 (x - y)  subject to  A * x = 0
2. Global Test for Gross Error Detection (GED) via Chi-Square distribution (chi2).
3. Individual measurement penalty test (normalized residuals) to identify failing meters.
4. Feeds reconciled, mass-conserved balances directly into the PINN / Digital Twin.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

class RefineryDVREngine:
    def __init__(self):
        # Streams in the coupled CDU + FCCU plant:
        # x0: Crude Feed (CDU In)
        # x1: CDU Overhead Naphtha (Out)
        # x2: CDU Kerosene Draw (Out)
        # x3: CDU Diesel Draw (Out)
        # x4: CDU Atmospheric Gas Oil (AGO) -> FCCU Feed (Internal Transfer)
        # x5: CDU Atmospheric Residue (Out)
        # x6: FCCU Fuel Gas / Off-Gas (Out)
        # x7: FCCU Liquified Petroleum Gas (LPG) (Out)
        # x8: FCCU Cracked Gasoline (Out)
        # x9: FCCU Light Cycle Oil (LCO) (Out)
        # x10: FCCU Decant / Slurry Oil (Out)
        # x11: FCCU Coke burned on catalyst (Out)
        self.stream_names = [
            "crude_feed_tph",             # 0
            "cdu_naphtha_tph",            # 1
            "cdu_kerosene_tph",           # 2
            "cdu_diesel_tph",             # 3
            "cdu_gas_oil_transfer_tph",   # 4
            "cdu_residue_tph",            # 5
            "fcc_fuel_gas_tph",           # 6
            "fcc_lpg_tph",                # 7
            "fcc_gasoline_tph",           # 8
            "fcc_lco_tph",                # 9
            "fcc_slurry_tph",             # 10
            "fcc_coke_tph"                # 11
        ]
        self.n_streams = len(self.stream_names)

        # Constraint Matrix A (A * x = 0):
        # Row 0: CDU Mass Balance: Crude - (Naphtha + Kero + Diesel + AGO + Residue) = 0
        # Row 1: FCCU Mass Balance: AGO - (FuelGas + LPG + Gasoline + LCO + Slurry + Coke) = 0
        self.A = np.zeros((2, self.n_streams))
        # CDU balance
        self.A[0, 0] =  1.0   # Feed in
        self.A[0, 1] = -1.0   # Naphtha out
        self.A[0, 2] = -1.0   # Kero out
        self.A[0, 3] = -1.0   # Diesel out
        self.A[0, 4] = -1.0   # Gas Oil out
        self.A[0, 5] = -1.0   # Residue out

        # FCCU balance
        self.A[1, 4] =  1.0   # Gas oil in
        self.A[1, 6] = -1.0   # Fuel gas out
        self.A[1, 7] = -1.0   # LPG out
        self.A[1, 8] = -1.0   # Gasoline out
        self.A[1, 9] = -1.0   # LCO out
        self.A[1, 10] = -1.0  # Slurry out
        self.A[1, 11] = -1.0  # Coke out

        # Meter standard uncertainties (sigma in % of reading, converted to covariance matrix Sigma)
        self.base_uncertainty_pct = {
            "crude_feed_tph": 1.0,           # High-precision custody transfer Coriolis
            "cdu_naphtha_tph": 2.5,          # Orifice meter
            "cdu_kerosene_tph": 2.5,
            "cdu_diesel_tph": 2.0,
            "cdu_gas_oil_transfer_tph": 2.0,
            "cdu_residue_tph": 3.0,          # Viscous slurry orifice
            "fcc_fuel_gas_tph": 4.0,         # Ultrasonic flare/gas meter
            "fcc_lpg_tph": 2.5,
            "fcc_gasoline_tph": 1.5,
            "fcc_lco_tph": 2.5,
            "fcc_slurry_tph": 3.5,
            "fcc_coke_tph": 5.0              # Flue gas carbon balance calculation
        }

    def reconcile_measurements(
        self,
        raw_measurements: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Executes VDI 2048 Data Reconciliation.
        Returns:
        - Reconciled consistent values strictly adhering to conservation.
        - Gross Error Detection (GED) test outcome.
        - Identified sensor anomalies / flags.
        """
        # Vectorize raw measurements
        y = np.zeros(self.n_streams)
        sigma = np.zeros(self.n_streams)
        for i, name in enumerate(self.stream_names):
            val = float(raw_measurements.get(name, 10.0))
            y[i] = val
            pct = self.base_uncertainty_pct[name]
            sigma[i] = max(0.1, val * (pct / 100.0))

        # Covariance matrix Sigma and precision W = Sigma^-1
        Sigma = np.diag(sigma ** 2)
        W = np.diag(1.0 / (sigma ** 2))

        # Analytical WLS Solution with linear constraints:
        # x_hat = y - Sigma * A^T * (A * Sigma * A^T)^-1 * (A * y)
        A_Sigma_AT = self.A @ Sigma @ self.A.T
        try:
            inv_A_Sigma_AT = np.linalg.inv(A_Sigma_AT)
        except np.linalg.LinAlgError:
            inv_A_Sigma_AT = np.linalg.pinv(A_Sigma_AT)

        imbalance_r = self.A @ y  # Prior mass unbalance
        correction = Sigma @ self.A.T @ inv_A_Sigma_AT @ imbalance_r
        x_hat = y - correction

        # Covariance of residuals: V = Sigma * A^T * (A * Sigma * A^T)^-1 * A * Sigma
        V = Sigma @ self.A.T @ inv_A_Sigma_AT @ self.A @ Sigma

        # Residuals: r_meas = y - x_hat
        residuals = y - x_hat

        # Global Chi-Square Test for Gross Errors
        # Objective J = (y - x_hat)^T * W * (y - x_hat)
        J = float(residuals.T @ W @ residuals)
        dof = self.A.shape[0] # degrees of freedom = rank(A) = 2
        chi2_crit = float(stats.chi2.ppf(confidence_level, df=dof))

        gross_error_detected = bool(J > chi2_crit)

        # Individual Normalized Residual Test (tau-test)
        flagged_meters = []
        tau_threshold = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)) # ~1.96

        for i, name in enumerate(self.stream_names):
            var_resid = V[i, i]
            if var_resid > 1e-6:
                tau_i = abs(residuals[i]) / np.sqrt(var_resid)
                if tau_i > tau_threshold:
                    flagged_meters.append({
                        "stream": name,
                        "raw_val": round(float(y[i]), 2),
                        "reconciled_val": round(float(x_hat[i]), 2),
                        "deviation_pct": round(float(abs(residuals[i]) / max(0.1, y[i]) * 100), 2),
                        "tau_score": round(float(tau_i), 2)
                    })

        reconciled_dict = {name: round(float(x_hat[i]), 3) for i, name in enumerate(self.stream_names)}
        raw_dict = {name: round(float(y[i]), 3) for i, name in enumerate(self.stream_names)}

        # Verified mass balances after reconciliation
        cdu_closure_error = float(self.A[0] @ x_hat)
        fcc_closure_error = float(self.A[1] @ x_hat)

        return {
            "vdi2048_status": "CONVERGED_RECONCILED",
            "gross_error_detected": gross_error_detected,
            "global_chi2_statistic": round(J, 3),
            "chi2_critical_threshold": round(chi2_crit, 3),
            "flagged_faulty_meters": flagged_meters,
            "raw_measurements": raw_dict,
            "reconciled_measurements": reconciled_dict,
            "mass_balance_closure_post_dvr": {
                "cdu_imbalance_tph": round(cdu_closure_error, 5),
                "fcc_imbalance_tph": round(fcc_closure_error, 5)
            }
        }
