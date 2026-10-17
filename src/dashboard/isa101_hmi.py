"""
ISA-101 High-Performance Human-Machine Interface (HMI) Visual Component.
Renders an interactive, industrial-standard Process & Instrumentation Diagram (P&ID) SVG.
Conforms to ISA-101 standards:
- Muted grayscale structural background to minimize visual fatigue.
- Color reserved exclusively for state anomalies:
  * Green/Gray = Normal Steady-State
  * Yellow/Amber = High/Warning Alarm (Flooding > 75%, Vibration Zone C)
  * Bright Red = Critical Danger / Trip Threshold (Flooding > 80%, Zone D)
- Real-time digital readout tags overlaid on transfer lines and vessels.
"""

def generate_isa101_pid_svg(
    cdu_feed_bpd: float = 100000.0,
    cdu_top_temp_c: float = 87.7,
    cdu_bot_temp_c: float = 348.0,
    cdu_flooding_pct: float = 0.3,
    vgo_transfer_bpd: float = 27170.0,
    fcc_riser_temp_c: float = 528.0,
    fcc_regen_temp_c: float = 690.0,
    gasoline_ron: float = 92.5
) -> str:
    # Determine alarm color indicators
    cdu_color = "#3b4252" # Normal dark slate
    if cdu_flooding_pct > 80.0:
        cdu_color = "#bf616a" # Alarm red
    elif cdu_flooding_pct > 75.0:
        cdu_color = "#ebcb8b" # Warning yellow

    fcc_color = "#3b4252"
    if fcc_riser_temp_c > 545.0:
        fcc_color = "#bf616a"
    elif fcc_riser_temp_c > 535.0:
        fcc_color = "#ebcb8b"

    svg_code = f"""
    <svg width="100%" height="480" viewBox="0 0 950 480" xmlns="http://www.w3.org/2000/svg" style="background-color: #1e222b; border-radius: 8px; font-family: monospace;">
        <!-- P&ID Background Grid / Headers -->
        <rect x="0" y="0" width="950" height="40" fill="#252a34"/>
        <text x="20" y="26" fill="#88c0d0" font-size="14" font-weight="bold">ISA-101 HIGH-PERFORMANCE HMI // INTEGRATED CDU-FCCU REFINING TRAIN</text>
        <text x="750" y="26" fill="#a3be8c" font-size="12">STATUS: CLOSED_LOOP_ADVISORY</text>

        <!-- PIPING LINES (Gray neutral tone) -->
        <!-- Crude Feed Line to CDU -->
        <path d="M 30 260 L 120 260" stroke="#707a8a" stroke-width="4" fill="none"/>
        <polygon points="120,256 128,260 120,264" fill="#707a8a"/>

        <!-- CDU Overhead to Naphtha -->
        <path d="M 180 90 L 180 60 L 320 60" stroke="#707a8a" stroke-width="3" fill="none"/>
        <polygon points="320,57 328,60 320,63" fill="#707a8a"/>

        <!-- CDU Side-cut Gas Oil to FCC Riser -->
        <path d="M 240 260 L 450 260 L 450 340 L 520 340" stroke="#88c0d0" stroke-width="4" stroke-dasharray="6,4" fill="none"/>
        <polygon points="520,336 528,340 520,344" fill="#88c0d0"/>

        <!-- FCC Riser to Disengager -->
        <path d="M 540 340 L 540 160" stroke="#d08770" stroke-width="6" fill="none"/>

        <!-- Catalyst Circulation Line (Regen -> Riser) -->
        <path d="M 720 240 L 540 340" stroke="#ebcb8b" stroke-width="3" stroke-dasharray="4,4" fill="none"/>

        <!-- EQUIPMENT 1: CRUDE DISTILLATION COLUMN (CDU) -->
        <rect x="120" y="90" width="120" height="280" rx="20" fill="{cdu_color}" stroke="#4c566a" stroke-width="3"/>
        <text x="145" y="115" fill="#eceff4" font-size="13" font-weight="bold">CDU-201</text>
        <text x="135" y="132" fill="#d8dee9" font-size="10">20-Tray Column</text>

        <!-- Trays -->
        <line x1="130" y1="160" x2="230" y2="160" stroke="#4c566a" stroke-width="2"/>
        <line x1="130" y1="190" x2="230" y2="190" stroke="#4c566a" stroke-width="2"/>
        <line x1="130" y1="220" x2="230" y2="220" stroke="#4c566a" stroke-width="2"/>
        <line x1="130" y1="250" x2="230" y2="250" stroke="#4c566a" stroke-width="2"/>
        <line x1="130" y1="280" x2="230" y2="280" stroke="#4c566a" stroke-width="2"/>
        <line x1="130" y1="310" x2="230" y2="310" stroke="#4c566a" stroke-width="2"/>

        <!-- CDU Telemetry Overlays -->
        <rect x="250" y="80" width="130" height="35" rx="4" fill="#2e3440" stroke="#434c5e"/>
        <text x="258" y="96" fill="#88c0d0" font-size="10">TI-201 Top Temp</text>
        <text x="258" y="110" fill="#eceff4" font-size="12" font-weight="bold">{cdu_top_temp_c} °C</text>

        <rect x="250" y="340" width="130" height="35" rx="4" fill="#2e3440" stroke="#434c5e"/>
        <text x="258" y="356" fill="#88c0d0" font-size="10">TI-202 Bot Temp</text>
        <text x="258" y="370" fill="#eceff4" font-size="12" font-weight="bold">{cdu_bot_temp_c} °C</text>

        <!-- Feed Inflow Tag -->
        <rect x="15" y="205" width="100" height="40" rx="4" fill="#2e3440" stroke="#434c5e"/>
        <text x="20" y="222" fill="#88c0d0" font-size="10">FI-101 Crude</text>
        <text x="20" y="238" fill="#eceff4" font-size="12" font-weight="bold">{cdu_feed_bpd:,.0f} BPD</text>

        <!-- Inter-unit VGO Stream Tag -->
        <rect x="290" y="285" width="140" height="40" rx="4" fill="#2e3440" stroke="#88c0d0"/>
        <text x="296" y="302" fill="#88c0d0" font-size="10">INTER-UNIT TRANSFER</text>
        <text x="296" y="318" fill="#a3be8c" font-size="12" font-weight="bold">{vgo_transfer_bpd:,.0f} BPD VGO</text>

        <!-- EQUIPMENT 2: FCC RISER & REACTOR DISENGAGER -->
        <rect x="500" y="100" width="90" height="110" rx="15" fill="{fcc_color}" stroke="#4c566a" stroke-width="3"/>
        <text x="515" y="125" fill="#eceff4" font-size="12" font-weight="bold">FCCU-301</text>
        <text x="512" y="140" fill="#d8dee9" font-size="9">Disengager</text>

        <!-- EQUIPMENT 3: FCC REGENERATOR -->
        <circle cx="760" cy="240" r="65" fill="#3b4252" stroke="#4c566a" stroke-width="3"/>
        <text x="725" y="235" fill="#eceff4" font-size="12" font-weight="bold">REGEN-302</text>
        <text x="720" y="250" fill="#ebcb8b" font-size="10">Coke Burner</text>

        <!-- FCC Telemetry Overlays -->
        <rect x="500" y="50" width="135" height="35" rx="4" fill="#2e3440" stroke="#434c5e"/>
        <text x="508" y="66" fill="#d08770" font-size="10">TI-301 Riser Temp</text>
        <text x="508" y="80" fill="#eceff4" font-size="12" font-weight="bold">{fcc_riser_temp_c} °C</text>

        <rect x="695" y="325" width="135" height="35" rx="4" fill="#2e3440" stroke="#434c5e"/>
        <text x="703" y="341" fill="#ebcb8b" font-size="10">TI-302 Regen Temp</text>
        <text x="703" y="355" fill="#eceff4" font-size="12" font-weight="bold">{fcc_regen_temp_c} °C</text>

        <!-- Octane Quality Tag -->
        <rect x="610" y="115" width="130" height="40" rx="4" fill="#2e3440" stroke="#a3be8c"/>
        <text x="618" y="132" fill="#a3be8c" font-size="10">AI QUALITY PREDICT</text>
        <text x="618" y="148" fill="#eceff4" font-size="13" font-weight="bold">{gasoline_ron} RON Octane</text>

        <!-- Footer Legend -->
        <rect x="20" y="440" width="14" height="14" fill="#3b4252"/>
        <text x="40" y="452" fill="#d8dee9" font-size="11">Normal / Non-alarm</text>

        <rect x="180" y="440" width="14" height="14" fill="#ebcb8b"/>
        <text x="200" y="452" fill="#d8dee9" font-size="11">Warning Alert (Zone C)</text>

        <rect x="370" y="440" width="14" height="14" fill="#bf616a"/>
        <text x="390" y="452" fill="#d8dee9" font-size="11">Critical Interlock Trip</text>
    </svg>
    """
    return svg_code
