# Digital Refineries & AI: Full-Stack Industry 4.0 Platform

## 📌 Executive Overview
This repository contains a full-stack, enterprise-grade **Digital Refinery & AI Platform** covering the complete lifecycle of modern continuous chemical process plants. Built according to ISA-95 (Purdue Model) standards, it synthesizes three core technological pillars:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DIGITAL REFINERY & AI MASTER PLATFORM                        │
├─────────────────────────┬─────────────────────────┬─────────────────────────────┤
│   PILLAR 1: PREDICTIVE  │   PILLAR 2: DIGITAL     │   PILLAR 3: SMART CONTROL   │
│   MAINTENANCE (PM)      │   TWIN MODEL (DT)       │   & RL OPTIMIZATION (DRP)   │
├─────────────────────────┼─────────────────────────┼─────────────────────────────┤
│ • NASA C-MAPSS dataset  │ • 20-Tray VLE Column    │ • Actor-Critic (PPO) RL     │
│ • Bi-LSTM Multi-Task Net│ • Thermodynamic Physics │ • Multi-Objective Reward    │
│ • ISO-10816-3 Vibration │ • Physics-Informed (PINN)│ • Safe Shadow-Mode Trims    │
│ • Sensor Sentinel Gate  │ • <1 ms Neural Surrogate│ • Hydraulic Interlocks      │
│ • SAP PM / Maximo Bridge│ • Carbon & Economics    │ • +$234k/yr Value Creation  │
└─────────────────────────┴─────────────────────────┴─────────────────────────────┘
```

---

## 🏗️ Master Architectural Schema

```mermaid
flowchart TD
    subgraph Level12["Level 1-2: Plant Floor & Edge"]
        Sensors["Industrial Sensor Network\n(OPC-UA / 4-20mA / HART)"]
        EdgeCollector["collector.py & historian_bridge.py\n(Swinging-Door Compression)"]
        Sensors --> EdgeCollector
    end

    subgraph Level3P1["Level 3: Predictive Maintenance (Pillar 1)"]
        Sentinel["sentinel.py\n(Frozen ADC & Drift Detection)"]
        Integrity["integrity.py\n(ISO-10816 Vibration & Transient Filter)"]
        BiLSTM["lstm_pm.py & export_edge.py\n(Bi-LSTM 97.8% Val Acc @ 8.8ms)"]
        Explain["explainability.py\n(Gradient-based Attribution)"]
        EdgeCollector --> Sentinel --> Integrity --> BiLSTM --> Explain
    end

    subgraph Level3P2["Level 3: Thermodynamic Digital Twin (Pillar 2)"]
        Physics["physics_engine.py\n(First-Principles Dynamic VLE)"]
        PINN["pinn_surrogate.py\n(Mass-Conserving Neural Surrogate)"]
        TwinSim["twin_engine.py\n(Real-Time What-If Lab)"]
        Physics --> PINN --> TwinSim
    end

    subgraph Level3P3["Level 3: Autonomous RL Optimization (Pillar 3)"]
        Env["refinery_env.py\n(Multi-Objective Gym Env)"]
        RL["rl_agent.py\n(Actor-Critic Setpoint Policy)"]
        Shadow["shadow_controller.py\n(Hydraulic Safety Interlock)"]
        TwinSim --> Env --> RL --> Shadow
    end

    subgraph Level4["Level 4: Enterprise Command Center & ERP"]
        SAP["cmms.py\n(SAP Notification Type M2 / Maximo)"]
        UI["app.py\n(Master 3-Pillar Streamlit UI)"]
        API["api.py\n(FastAPI & Prometheus Metrics)"]
        Explain --> SAP
        Shadow --> UI
        TwinSim --> UI
        BiLSTM --> UI
        BiLSTM --> API
    end
```

---

## 📂 Codebase Inventory

Located at: `C:\Users\VICTUS\.gemini\antigravity\scratch\refinery-pm`

```
refinery-pm/
├── docker-compose.yml                     # Kafka, Zookeeper, Prometheus, Grafana, Collector
├── requirements.txt                       # Pure PyTorch, FastAPI, Streamlit, Pandas
├── README.md                              # Master architectural runbook
├── .gitignore                             # Industrial repository ignore rules
├── monitoring/
│   └── prometheus.yml                     # Prometheus scraper configuration
├── checkpoints/
│   ├── best_refinery_pm.pt                # Bi-LSTM checkpoint (97.84% val acc)
│   ├── feature_scaler.json                # Feature normalization dictionary
│   ├── edge/
│   │   └── refinery_pm_edge.pt            # Standalone TorchScript binary (8.8 ms)
│   ├── digital_twin/
│   │   └── pinn_surrogate.pt              # Mass-conserving PINN surrogate model
│   └── optimizer/
│       └── refinery_policy.pt             # Trained Actor-Critic RL controller weights
├── data/
│   ├── raw/
│   │   ├── real_benchmark/PM_train.txt    # NASA C-MAPSS industrial turbomachinery data
│   │   ├── sensors/                       # Parquet sensor lake
│   │   └── maintenance_logs/              # Work order failure logs
│   ├── processed/
│   │   ├── features.parquet               # 140 engineered feature dimensions
│   │   └── failure_label.parquet          # Continuous RUL and failure horizon labels
│   ├── alerts.jsonl                       # Real-time alert journal
│   ├── cmms_work_orders.jsonl             # Auto-dispatched SAP work orders
│   └── operator_feedback.jsonl            # Technician human-in-the-loop review log
├── src/
│   ├── ingest/
│   │   ├── collector.py                   # Industrial OPC-UA reader -> Kafka
│   │   ├── historian_bridge.py            # OSIsoft PI deadband exception filter
│   │   ├── download_real_benchmark.py     # Real benchmark dataset ingestion
│   │   └── generate_mock_data.py          # Synthetic dataset generator
│   ├── features/
│   │   └── build_features.py              # Rolling moments and FFT spectral pipeline
│   ├── model/
│   │   ├── lstm_pm.py                     # Multi-task Bi-LSTM PyTorch network
│   │   ├── trainer.py                     # Training & validation loop
│   │   ├── explainability.py              # Gradient-based sensor attribution
│   │   ├── sentinel.py                    # Sensor data quality & drift gatekeeper
│   │   ├── integrity.py                   # ISO-10816 vibration & transient filter
│   │   ├── cmms.py                        # SAP PM work order dispatch
│   │   ├── auto_retrain.py                # Active learning watcher
│   │   └── export_edge.py                 # TorchScript C++ SCADA edge exporter
│   ├── digital_twin/
│   │   ├── physics_engine.py              # First-principles dynamic VLE simulator
│   │   ├── pinn_surrogate.py              # Physics-Informed Neural Network
│   │   ├── train_surrogate.py             # PINN calibration and training
│   │   ├── twin_engine.py                 # Real-time what-if scenario lab
│   │   └── verify_twin.py                 # Digital Twin verification test
│   ├── optimizer/
│   │   ├── refinery_env.py                # Multi-objective Gymnasium RL environment
│   │   ├── rl_agent.py                    # Actor-Critic policy network
│   │   ├── train_rl.py                    # Policy gradient training loop
│   │   └── shadow_controller.py           # Industrial shadow-mode safety controller
│   ├── serve/
│   │   ├── api.py                         # FastAPI inference service + metrics
│   │   └── stream_consumer.py             # Sliding buffer streaming consumer
│   ├── dashboard/
│   │   └── app.py                         # Master 3-Pillar Streamlit Command Center
│   ├── run_enterprise_validation.py       # Pillar 1 test suite
│   ├── run_pipeline_demo.py               # Smoke test script
│   └── run_master_platform_validation.py  # Complete 3-Pillar integration test
```

---

## 🚀 How to Run the Platform

### 1. Execute the 3-Pillar Master Validation Suite
Validates all three systems in a single automated run:
```powershell
cd C:\Users\VICTUS\.gemini\antigravity\scratch\refinery-pm
python src\run_master_platform_validation.py
```

### 2. Launch the Master Unified Command Center (Streamlit)
```powershell
streamlit run src\dashboard\app.py
```
* **Tab 1 (Predictive Maintenance):** Live asset health, ISO-10816 severity bands, auto-dispatched SAP work orders, and technician feedback.
* **Tab 2 (Digital Twin Simulator):** Real-time what-if lab with interactive crude throughput, furnace duty, and reflux ratio controls.
* **Tab 3 (Autonomous RL Optimization):** AI closed-loop setpoint recommendations, safety interlocks, and annualized value creation ($/yr).

### 3. Launch Distributed API & Metrics
```powershell
python src\serve\api.py
```
- Interactive Swagger: `http://localhost:8000/docs`
- Prometheus Scrape: `http://localhost:8000/metrics`
