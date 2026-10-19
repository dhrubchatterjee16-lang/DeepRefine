# DeepRefine: Open Industrial Reference Architecture & Testbed

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Standards](https://img.shields.io/badge/ISA--95%20%7C%20IEC%2062443%20%7C%20VDI%202048-Compliant-green.svg)](https://www.isa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Architectural Positioning**: DeepRefine is an open-source reference architecture and simulation testbed for **Purdue Model Level 3 Supervisory Real-Time Optimization (RTO)**, **Physics-Informed Digital Twins (PINN)**, and **API 670 Machinery Diagnostics** in continuous brownfield chemical and petroleum refining plants.

---

## 📌 Architectural Scope & Safety Boundaries

> [!IMPORTANT]
> **Safety & Operational Boundaries (IEC 61511 / IEC 62443 / OSHA 1910.119 PSM):**
> 1. **Phase 1 Read-Only Operator Advisory Mode**: DeepRefine possesses **zero direct write authority** over physical Level 1 actuators (control valves, variable frequency drives, or heaters). All AI/RTO recommendations are presented as bounded advisory targets requiring human operator sign-off and DCS validation.
> 2. **Level 3 Supervisory Subordination**: DeepRefine operates strictly at **Level 3**, generating target vectors for classical **Level 2 Dynamic Matrix Controllers (DMC)**. The DMC enforces hard slew-rate limits and multivariable dynamic coupling.
> 3. **Hardware-Enforced SIS Priority**: Emergency shutdown (ESD) elements and safety instrumented functions (SIF) reside entirely in independent physical safety systems (e.g., Triconex / HIMA TMR 2oo3 logic solvers). A 1500 ms hardware dry-contact watchdog forces instant fallback to safe park mode if supervisory communications lapse.
> 4. **HIL Benchmark Testbed**: All plant dynamic responses and telemetry are driven by verified first-principles thermodynamic and kinetic benchmark simulators designed for rapid what-if testing and algorithm verification.

---

## 🏗️ Master Architectural Schema (Purdue / ISA-95)

```
Level 4: Enterprise Network & ERP
┌───────────────────────────────────────────────────────────────────────────────┐
│ • Streamlit ISA-101 Unified Operations Console (5 Dedicated Tabs)             │
│ • SAP PM / IBM Maximo Maintenance Dispatch (ISO 14224 Equipment Taxonomy)     │
│ • FastAPI Production Services & Prometheus Metric Telemetry Scrapers          │
└───────────────────────────────────────▲───────────────────────────────────────┘
                                        │  IEC 62443-3-3 Level 3.5 IDMZ
                                        │  (mTLS x.509 + Unidirectional Telemetry)
Level 3: Supervisory Advisory & Digital Twin
┌───────────────────────────────────────────────────────────────────────────────┐
│ • Pillar 1: Predictive Maintenance (API 670 Orbit Vectors, ISO 14224, Bi-LSTM)│
│ • Pillar 2: Thermodynamic Digital Twin (20-Tray VLE, 4-Lump FCCU, VDI 2048)   │
│ • Pillar 3: Supervisory RTO (PPO Continuous Policy + KKT Constrained QP DMC)  │
│ • Inferential Soft Sensors: FCC RON/MON, Jet Flash/Freeze, Diesel D86 95%    │
│ • LIMS Closed-Loop Feedback: Exponential bias update with 3.5σ Chauvenet gate │
└───────────────────────────────────────▲───────────────────────────────────────┘
                                        │  Read-Only Hardware Permissives &
                                        │  1500 ms Heartbeat Watchdog Relay
Level 1-2: Regulatory DCS & Independent SIL-3 SIS
┌───────────────────────────────────────────────────────────────────────────────┐
│ • Level 2 Dynamic DMC (FOPDT FIR Step-Response Convolution over P=25, M=5)    │
│ • Level 1 PID Regulatory Control Loops (Flow, Pressure, Level, Temperature)   │
│ • Level 0 Field Actuators & SIL-3 TMR 2oo3 Critical Slide Valve dP Interlock  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 💎 Core Technical Innovations

### 1. Rotating Machinery Reliability & Diagnostics (Pillar 1)
* **API 670 Proximity Orbit Analysis**: Orthogonal 90° X/Y eddy-current probe vector decomposition ($S_{\max}$), tracking 1X unbalance, 2X misalignment, and 0.43X oil whirl hydrodynamic instability on hydrodynamic journal bearings.
* **API 617 & 610 Machine Hydraulics**: Continuous ASME PTC 10 dynamic distance-to-surge monitoring for centrifugal compressors and NPSHa cavitation margin tracking for API 610 pumps.
* **ISO 14224 Failure Taxonomy**: Standardized mapping of mechanical failure modes, severity levels, and automated SAP Maintenance Notification (Type M2) work orders.
* **Multi-Task Bi-LSTM Network**: Simultaneous classification of 5-day mechanical failure horizon and continuous Remaining Useful Life (RUL in hours) with serialized TorchScript SCADA edge binary (< 10 ms inference).

### 2. Coupled Thermodynamic Digital Twin & Soft Sensors (Pillar 2)
* **20-Tray Dynamic VLE Distillation Engine**: Multi-component state-space vapor-liquid equilibrium (Naphtha, Kerosene, Diesel, Atmospheric Residue) with tray hydraulic flooding checks via Fair's correlation.
* **Coupled 4-Lump FCCU Reactor**: Heavy gas oil catalytic cracking kinetics with reactor-regenerator thermal balance and coke burn emissions modeling.
* **VDI 2048 Data Validation & Reconciliation (DVR)**: Constrained Weighted Least Squares (WLS) mass conservation engine with global $\chi^2$ Gross Error Detection (GED) and normalized residual penalty tests ($\tau$-test) to isolate faulty plant flowmeters.
* **True Mass-Conserving PINN Surrogate**: High-speed neural surrogate (< 1 ms) enforcing metric ton/hour ($\text{tph}$) mass balance using petroleum cut specific gravities ($\text{SG} \times \rho_{\text{water}}$).
* **Inferential Soft Sensors with LIMS Reconciliation**: Real-time virtual property predictors (RON/MON, Pensky-Martens flash point, freeze point, D86 95%) with automated lab assay feedback and exponential bias updating.

### 3. Supervisory RTO & Constrained Dynamic DMC (Pillar 3)
* **Level 3 Supervisory RTO**: Reinforcement learning continuous setpoint policy optimizing refinery crack margins and energy consumption under safe operational bounds.
* **Level 2 Constrained Dynamic DMC**: True Karush-Kuhn-Tucker (KKT) constrained Quadratic Programming (QP) solver optimizing over control horizon $M=5$ and prediction horizon $P=25$, enforcing actuator rate limits ($-\Delta u_{\max} \le \Delta u \le \Delta u_{\max}$) and absolute MV boundaries.
* **FMEA Runtime Watchdog & Bumpless APC Transfer**: Multi-tier fault detection with automatic failover to traditional linear APC if sensor sentinel checks fail.

### 4. Operational Technology (OT) Security & ISA-101 HMI
* **IEC 62443 Level 3.5 IDMZ**: Authenticated reverse-proxy staging architecture separating Level 4 IT applications from Level 2/3 plant control, blocking any direct valve commands.
* **ISA-101 High Performance HMI**: Greyscale SVG P&ID plant mimic designed per ANSI/ISA-101.01 ergonomics to reduce operator fatigue and surface abnormal situations instantly.

---

## ⚡ Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-org/DeepRefine.git
cd DeepRefine
pip install -r requirements.txt
```

### 2. One-Command Master Launcher
Launch pre-flight integrity checks and the ISA-101 Streamlit Command Center:
```bash
python run_quickstart.py
```
* **Dashboard URL**: `http://localhost:8501`
  * **Tab 1: Plant Overview & P&ID Mimic**: ISA-101 high-performance console with live process telemetry.
  * **Tab 2: Asset Health & API 670 Diagnostics**: Orbit plots, 1X/2X harmonics, and ISO 14224 work orders.
  * **Tab 3: Digital Twin & VDI 2048 DVR**: Mass balance reconciliation, gross error detection, and what-if simulation.
  * **Tab 4: Quality Inferentials & LIMS**: Soft sensor property predictions and lab assay calibration updates.
  * **Tab 5: Supervisory RTO & Dynamic DMC**: Level 3 setpoint policy, constrained QP moves, and SIS status.

### 3. Run Automated Industrial Verification Suites
Execute the full council hardening and acceptance test matrix:
```bash
# 1. Council Hardening Verification (5 Core Industrial Pillars)
python src/test_council_hardening.py

# 2. 10-Point Factory & Site Acceptance Test Protocol (FAT / SAT)
python src/test_fat_sat_protocol.py
```

---

## 📊 AI/ML vs Classical Engineering Taxonomy

| Architectural Subsystem | Implementation Methodology | Classification | Standard Compliance |
|---|---|---|---|
| **API 670 Vibration & Orbits** | Vector combination ($S_{\max}$), 1X/2X harmonic decomposition | Classical Deterministic | API 670 / ISO 10816-3 |
| **Machinery Anomaly & RUL** | Multi-Task Bidirectional LSTM with TorchScript edge compilation | Deep Learning (AI) | ISO 14224 |
| **Sensor Sentinel** | Chauvenet 3.5$\sigma$ outlier gate, ADC frozen channel detector | Statistical Quality Control | ASTM E178 |
| **VLE Column Distillation** | Dynamic 20-tray MESH state-space VLE differential equations | First-Principles Physics | AIChE / GPSA |
| **FCCU Cracking Reactor** | 4-lump Arrhenius reaction network + regenerator thermal balance | First-Principles Kinetics | UOP / Kellogg Kinetics |
| **VDI 2048 DVR Engine** | Constrained Weighted Least Squares (WLS) + Global $\chi^2$ Test | Analytical Optimization | VDI 2048 / ISO 5168 |
| **Digital Twin Surrogate** | Deep Residual MLP with metric tons/hr ($\text{tph}$) mass penalty | Physics-Informed ML (PINN) | AIChE / VDI 2048 |
| **Inferential Soft Sensors** | Empirical polynomial correlations with LIMS exponential bias | Semi-Empirical Inferential | ASTM D86 / D93 / D2386 |
| **Supervisory RTO** | Continuous PPO Actor-Critic Setpoint Guidance | Reinforcement Learning (AI) | ISA-95 Level 3 |
| **Dynamic Matrix Control** | KKT Constrained Quadratic Programming (QP) with FIR convolution | Advanced Process Control (APC) | Cutler & Ramaker (1979) |
| **Safety Interlock Gateway** | Emulated TMR 2oo3 majority voting + 1500 ms dry-contact watchdog | Hardware Interlock Emulation | IEC 61511 (SIL-3) |
| **OT Network Security** | IDMZ Reverse Proxy with mTLS certificate inspection | Cyber-Physical Security | IEC 62443-3-3 |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
