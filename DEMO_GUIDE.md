# DeepRefine: 5-Minute Live Demonstration Walkthrough Script
## A Step-by-Step Guide for Client, Executive, or Interview Presentations

---

### Prerequisites
Launch the live platform before your presentation:
```powershell
cd C:\Users\VICTUS\.gemini\antigravity\scratch\refinery-pm
streamlit run src\dashboard\app.py
```

---

### Step 1: The Problem & Predictive Maintenance (Tab 1)
**What to say:**
> *"Welcome to DeepRefine. We start on plant asset reliability. Traditional refineries inspect rotating equipment on fixed calendar schedules, leading to catastrophic bearing seizures. DeepRefine ingests high-frequency vibration and temperature telemetry."*

**What to do:**
1. In the asset dropdown, select **`WGC-101-COMPRESSOR`**.
2. Point out:
   - Asset Health Index is degrading (**58%**).
   - ISO 10816-3 machinery vibration classification has tripped into **Zone C (Unsatisfactory)**.
   - Financial exposure calculates **\$78,500 USD** at risk.
   - The right sidebar shows an automated **SAP Notification Type M2** work order generated with exact replacement parts (SKF thrust bearing) and isolation procedures.

---

### Step 2: The Thermodynamic Digital Twin (Tab 2)
**What to say:**
> *"Now we move to unit simulation. Traditional AspenTech simulation runs can take minutes. DeepRefine trained a mass-conserving Physics-Informed Neural Network (PINN) that converges in under 1 millisecond."*

**What to do:**
1. Switch to **"🔮 Pillar 2: Distillation Digital Twin"**.
2. Move the **Crude Intake slider** from `100,000` to `115,000 BPD`.
3. Move the **Furnace Duty slider** up to `48 MW`.
4. Point out:
   - Instantaneous product cut reallocation (Naphtha, Kerosene, Diesel, and Residue bar charts).
   - Column flooding margin and daily carbon footprint ($233\text{ Tons CO}_2/\text{day}$).

---

### Step 3: Safe Closed-Loop Advisory & Human Clearance (Tab 3)
**What to say:**
> *"Here is how we ensure refinery safety. Plant directors never allow an AI to touch physical valves directly. DeepRefine is certified in Phase 1 Read-Only Operator Advisory Mode with strict slew-rate clamps."*

**What to do:**
1. Switch to **"⚡ Pillar 3: Operator Advisory (RL)"**.
2. Click **"🔍 Generate AI Optimization Advisory Card"**.
3. Point out:
   - Slew-rate limiter has clamped the furnace ramp to prevent refractory thermal shock.
   - Economic value calculation projects **+\$227,000 to \$234,000/year** profit gain.
4. Show the **Human Clearance Sign-Off**: Enter an operator badge ID, select "ACCEPT_TRIM", and click **"Sign & Authorize Advisory"** to demonstrate the immutable DCS console slip audit.

---

### Step 4: The Plant-Wide Flowsheet & ISA-101 Control Room (Tab 4)
**What to say:**
> *"Finally, we connect the entire refinery train. We route the gas oil stream from the crude column directly into the secondary Fluid Catalytic Cracker (FCCU), rendered in high-performance ISA-101 HMI grayscale."*

**What to do:**
1. Switch to **"🌐 Path C: Plant Flowsheet Network"**.
2. Scroll to the **ISA-101 High-Performance P&ID Console**.
3. Show the dynamic SVG overlay: The Crude Column (CDU) piping connecting to the FCC Riser and Regenerator.
4. Adjust the **FCC Riser Temperature slider** from $500^\circ\text{C}$ to $532^\circ\text{C}$.
5. Show how the predicted **Gasoline Pool Octane jumps to 92.5+ RON**, unlocking **\$2.77 Billion/year** in annualized complex EBITDA.
