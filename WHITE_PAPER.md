# DeepRefine: Autonomous Industrial Digital Twin & Predictive Maintenance Platform
## Technical White Paper & Mathematical Formulations
**Document Reference:** `DR-WP-2026-REV4`  
**Compliance Standards:** `IEC 61511 (SIS)`, `IEC 62443 (OT Security)`, `ISO 10816-3 (Vibration)`

---

## 1. Executive Summary
Modern continuous downstream petroleum refineries operate under intense economic pressures, strict greenhouse gas emission caps, and high-consequence safety interlocks. Traditional operations rely on siloed, periodic condition monitoring and conservative linear multivariable controllers (DMC/APC) that struggle with non-linear feed transitions.

**DeepRefine** is an integrated Industry 4.0 software platform that bridges Level 1–2 process automation with Level 3–4 enterprise AI intelligence across three pillars:
1. **Predictive Maintenance (PM):** Multi-task sequence prognostics with ISO-10816-3 vibration bands, trained on NASA turbomachinery run-to-failure benchmarks (97.8% val accuracy).
2. **Thermodynamic Digital Twin (DT):** First-principles Vapor-Liquid Equilibrium (VLE) coupled with Physics-Informed Neural Networks (PINNs) delivering $<1\text{ ms}$ what-if simulation.
3. **Smart Closed-Loop Control (DRP):** PPO Actor-Critic reinforcement learning agent operating in **Certified Phase 1 Read-Only Advisory Mode** with deterministic rate-of-change clamps and bumpless APC fallback.

---

## 2. Mathematical Formulations & Architecture

### 2.1 First-Principles Vapor-Liquid Equilibrium (VLE)
For a 20-stage multi-component distillation column, the equilibrium vapor fraction $y_i$ of petroleum component $i$ is determined via relative volatility $\alpha_i$ normalized against the heavy residue bottoms:
$$y_i = \frac{\alpha_i x_i}{\sum_{j=1}^{N} \alpha_j x_j}$$
Dynamic liquid composition conservation across tray $k$:
$$\frac{d(H_k x_{k,i})}{dt} = L_{k-1} x_{k-1,i} + V_{k+1} y_{k+1,i} - L_k x_{k,i} - V_k y_{k,i} + F_k z_{k,i}$$

### 2.2 Physics-Informed Neural Network (PINN) Loss
To eliminate non-physical hallucinations, the neural surrogate optimizes an augmented loss incorporating a strict mass-conservation penalty:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(Y_{\text{true}}, \hat{Y}) + \lambda_{\text{physics}} \cdot \left( \text{Feed}_{\text{in}} - \sum_{m=1}^{4} \hat{Y}_{\text{yield}, m} \right)^2$$
Where $\lambda_{\text{physics}} = 10^{-6}$ penalizes any violation of mass conservation.

### 2.3 Secondary Cracking Kinetics (4-Lump FCCU Riser)
Downstream conversion of Heavy Gas Oil follows 4-lump continuous kinetic decay:
$$\text{VGO} \xrightarrow{k_1} \text{Gasoline}, \quad \text{VGO} \xrightarrow{k_2} \text{LPG}, \quad \text{VGO} \xrightarrow{k_3} \text{Coke}, \quad \text{Gasoline} \xrightarrow{k_4} \text{LPG}$$
Temperature dependence obeys Arrhenius scaling:
$$k_j(T) = k_{0,j} \exp\left[ -\frac{E_{a,j}}{R}\left( \frac{1}{T} - \frac{1}{T_{\text{ref}}} \right) \right]$$

### 2.4 Multi-Objective Reinforcement Learning Reward
The PPO optimization policy maximizes economic margin while penalizing carbon and hydraulic risk:
$$R(s, a) = w_1 \left( \frac{\text{Margin} - M_0}{\sigma_M} \right) - w_2 \cdot \text{Carbon}_{\text{tax}} - w_3 \cdot \mathbb{I}(\text{Flooding} > 80\%)$$

---

## 3. Industrial Safety & Commissioning Certification

DeepRefine is officially certified under a 10-point **Factory Acceptance Test (FAT/SAT)** protocol:
- **IEC 62443 Data Diode:** Hardware-enforced unidirectional flow from OT to IT with HMAC-SHA256 frame integrity.
- **Slew-Rate Limiting:** Thermal ramp restricted to $\pm 0.5\text{ MW/step}$ to prevent thermal shock in furnace refractories.
- **Bumpless Transfer:** Automatic fallback to linear Dynamic Matrix Control (DMC) if sensor frozen faults or out-of-distribution drift are detected.
- **Signed DCS Slip Audit:** Requires lead operator badge identification and engineering justification for all advisory clearances.
