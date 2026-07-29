# GT3 Performance Analysis Toolkit

**GT3 Race Engineering Portfolio Project**  
IIT Kharagpur · MTS Monza Race Engineering Certificate · Utkarsh Chaudhari

---

## Background

GT3 endurance racing generates several gigabytes of telemetry per session. The performance engineer's job is to turn that data into three decisions before the next session begins: what the car is doing, why it's doing it, and what to change. This toolkit replicates that workflow end-to-end — from raw CSV ingest through to a PDF session report — using the same tools and methodology used by GT3 manufacturer teams.

The project was built specifically to demonstrate competency across the Toyota Racing GmbH GT3 Performance Engineer JD requirements: data acquisition workflows, tyre temperature and degradation analysis, G-G diagram interpretation, vehicle dynamics modelling (Pacejka MF96), setup sensitivity analysis (ARB sweep), and session reporting.

---

## Data Sources

All analysis runs on real-format GT3 telemetry. The included data generator produces a 10-lap synthetic session at 100 Hz that matches the channel structure of an ACC export or Motec i2 MLD file:

- Speed, RPM, gear, throttle, brake pressure, steering angle
- Lateral/longitudinal/vertical accelerometer (g)
- Tyre temperatures: inner / mid / outer per corner (12 channels)
- Tyre pressures per corner
- Suspension travel per corner

To use real session data, pass any Motec CSV or ACC telemetry export directly:

```bash
python run_analysis.py --session path/to/your_session.csv --compound DHD2
```

---

## Methodology

### Phase 1 — Data Acquisition Pipeline

The pipeline ingests raw telemetry, normalises channel names across 30+ naming variants (Motec i2, Race Studio, ACC), resamples to a uniform 100 Hz time base, and computes eight derived channels that Motec i2 calculates internally. Replicating these from first principles rather than reading them from i2 output proves understanding of the underlying calculation methodology.

**Computed maths channels:**

| Channel | Formula | Engineering use |
|---|---|---|
| `LatG_calc` | V² × δ / (g × L) — Milliken bicycle model | Validate IMU, detect sensor drift |
| `LonG_calc` | dV/dt filtered at 15 Hz | Braking efficiency, traction balance |
| `SteeringRate` | d(δ)/dt filtered at 10 Hz | Driver input smoothness, entry technique |
| `YawRateEstimate` | (ay × g) / V | Chassis rotation vs theoretical |
| `OversteerMetric` | LatG_calc − LatG_measured | Balance read, setup direction |
| `LateralLoadTransferFront` | m × ay × g × h_cg / (2 × tf) | Front tyre loading per corner |
| `LateralLoadTransferRear` | m × ay × g × h_cg / (2 × tr) | Rear tyre loading per corner |
| `BalanceIndex` | ΔLLT_front / (ΔLLT_front + ΔLLT_rear) | Instantaneous balance read |
| `TractionCircleUtil` | √(LatG² + LonG²) / grip_limit | Grip utilisation percentage |

The processed session and a Motec-importable maths channel CSV are written to `output/` — the maths channel file can be loaded directly into i2 as a custom channel overlay.

### Phase 2 — Tyre Analysis

**Temperature operating window classifier.** Every 100 Hz sample is classified as COLD / IN_WINDOW / HOT per corner, using the mid-zone temperature against compound-specific operating windows. The window boundaries are configurable per compound: SC, DC, DHD, DHD2. Per-lap window utilisation percentage is computed, and anomaly detection flags samples where temperature delta exceeds 6 °C (flat spot or blister onset signal).

**Pacejka Magic Formula MF96.** Full MF96 implementation in MATLAB covering Fy (lateral force vs slip angle) and Fx (longitudinal force vs slip ratio), with separate coefficient sets for front and rear axles calibrated to GT3 slick compound behaviour. Outputs:

- Fy curves at four vertical load levels (2500–5500 N representing light to full aero loading)
- Fx curves across the same load range
- Peak force, cornering stiffness, and slip stiffness printed to console
- Degressive load sensitivity analysis: why doubling downforce does not double cornering speed
- Friction ellipse: combined Fx/Fy capacity at nominal load

**Degradation model.** Linear and polynomial (degree 2) regression of lap time vs tyre age. Outputs degradation rate in ms/lap, R², and a predicted crossover lap — the point at which lap time delta from a fresh set exceeds 0.5 s, used as a pit strategy input.

### Phase 3 — G-G Analysis and Setup Development

**G-G diagram (three views):**

1. KDE density — kernel density estimate showing where the driver spends time on track. Dense core at lateral limit is normal GT3 signature; sparse combined braking/cornering quadrant indicates unused grip.
2. Sector split — braking, cornering, and traction phases isolated by longitudinal G threshold. The achieved friction envelope vs theoretical limit shows where time is being left on track.
3. Lap consistency — G-G envelope overlaid per lap, coloured by lap time. Tight cluster indicates consistent driver; wide spread indicates variability in entry or exit technique.

**ARB sensitivity sweep.** 2-DOF linear bicycle model sweep across front ARB stiffness ±60% of baseline, with rear ARB held fixed. Computes lateral load transfer distribution (LLTD), understeer gradient K_us, and estimated lap time delta at four lateral acceleration levels. The balance map (Panel D) plots K_us as a contour over front × rear ARB stiffness space — the white K_us = 0 contour is the neutral setup line. This is the tool used in driver debrief: "you reported push in Turn 3 — here is the ARB adjustment to shift K_us toward neutral and our estimated lap time recovery."

**Session report.** Single command generates a two-page PDF in team branding. Page 1: session header, lap time table with delta and tyre window %, speed trace overlay, G-G diagram, tyre temperature heatmap. Page 2: degradation regression, oversteer metric trace, setup snapshot table, and three engineer recommendation bullets (TYRES / BALANCE / DEG) written in performance engineering language.

---

## Results

Measured on synthetic 10-lap session (Brands Hatch GP proxy, 95.2–96.5 s lap times):

| Metric | Value |
|---|---|
| Tyre degradation rate | 150 ms/lap |
| Regression R² | 0.991 |
| Crossover lap | 4 |
| Peak lateral G | 1.17 g (calc) |
| Mean grip utilisation | 76.3% |
| >90% grip utilisation | 50.5% of session |
| Front peak Fy (MF96, Fz=3500 N) | 3.85 kN at 6.2° slip |
| Rear peak Fy (MF96, Fz=3500 N) | 3.78 kN at 6.5° slip |
| ARB neutral window (front) | 27.8–34.2 N·m/deg |
| Oversteer % of session | 49.3% |
| Understeer % of session | 49.2% |

---

## Engineering Conclusions

The session data shows:

1. **Tyres are running consistently above operating window** (mean FR mid temp > 125 °C vs 110 °C ceiling for DHD compound). Recommendation: reduce blanket temperature by 5 °C and explore reducing front-right camber to improve inner/outer spread.

2. **Degradation is linear at 150 ms/lap** with very high confidence (R² = 0.991), suggesting no thermal cliff event in the observed stint. Crossover lap 4 with a 0.5 s threshold implies a 4–5 lap stint maximum before losing significant time to fresh rubber.

3. **Balance is near-neutral** (oversteer/understeer split 49.3% / 49.2%). The oversteer metric shows no persistent corner type bias. Current front ARB at 35 N·m/deg sits within the computed neutral window — no immediate ARB change recommended unless driver reports balance shift.

---

## File Structure

```
gt3_toolkit/
│
├── run_analysis.py              # Unified CLI — one command runs all phases
├── dashboard.py                 # Interactive Streamlit dashboard
│
├── telemetry_pipeline.py        # Phase 1: ingest → Motec maths channels
├── generate_sample_data.py      # Synthetic GT3 session generator (100 Hz)
├── phase1_analysis.py           # Phase 1 plots
│
├── tyre_temperature_analysis.py # Phase 2: window classifier + degradation
├── pacejka_magic_formula.m      # Phase 2: MF96 Fy/Fx (MATLAB, no toolboxes)
│
├── gg_diagram.py                # Phase 3: G-G KDE, sector split, consistency
├── arb_sensitivity_sweep.m      # Phase 3: ARB sweep, balance map (MATLAB)
├── session_report.py            # Phase 3: automated PDF report
│
├── data/
│   └── sample_session.csv       # Generated on first run
└── output/
    ├── processed.csv            # Processed telemetry (41 channels)
    ├── motec_maths.csv          # Motec i2 importable maths channels
    ├── 01_lap_trace.png         # Speed/throttle/brake/steering overlay
    ├── 02_maths_validation.png  # LatG Milliken vs measured
    ├── 03_traction_circle.png   # Traction circle coloured by speed
    ├── 04_load_transfer.png     # Front/rear LLT + balance index
    ├── 05_temp_trace.png        # 4-corner temperature trace with window shading
    ├── 06_window_utilisation.png# Per-lap window utilisation bars
    ├── 07_degradation.png       # Lap time regression + crossover
    ├── 08_temp_heatmap.png      # Temperature heatmap per corner per lap
    ├── 10_gg_density.png        # KDE density G-G
    ├── 11_gg_sectors.png        # Sector-split G-G + friction envelope
    ├── 12_gg_consistency.png    # Lap consistency overlay
    └── session_report.pdf       # Two-page PDF session report
```

---

## Installation and Usage

```bash
# Clone and install dependencies
git clone https://github.com/utkarsh410/gt3-performance-toolkit
cd gt3-performance-toolkit
pip install -r requirements.txt

# Run full pipeline (generates synthetic data automatically)
python run_analysis.py

# Run on your own session file
python run_analysis.py --session data/my_session.csv --compound DHD2

# Run only Phase 1
python run_analysis.py --phase 1

# Launch interactive dashboard
streamlit run dashboard.py

# MATLAB analysis (run in MATLAB, no toolboxes required)
# Open pacejka_magic_formula.m and run — generates Fy/Fx curves
# Open arb_sensitivity_sweep.m and run — generates ARB balance map
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scipy>=1.9
matplotlib>=3.6
scikit-learn>=1.1
streamlit>=1.25
reportlab>=3.6
```

MATLAB files require base MATLAB only — no Simulink or additional toolboxes.

---

## Context

This toolkit was built to demonstrate core GT3 race engineering competencies: telemetry data acquisition, tyre temperature analysis, vehicle dynamics modelling, setup development, and session reporting — the full trackside analysis loop a race engineer works through over a race weekend.
Trackside experience: MTS Monza (Formula 4 Abarth, Lamborghini Huracán GT3) and Prudent Motorsports. Tools: Motec i2, Race Studio, MATLAB, Python telemetry pipelines.

Contact: utkarsh410@gmail.com
