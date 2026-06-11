"""
phase1_analysis.py
------------------
Phase 1 visualisation module — validates the pipeline output and generates
the plots you'd produce during a Motec i2 session review.

Plots generated:
  1. Speed trace + maths channel overlay (lap comparison)
  2. Steering rate vs lateral G (driver input quality)
  3. Traction circle (G-G diagram preview — full version in Phase 3)
  4. Oversteer metric trace per lap
  5. Lateral load transfer front vs rear
  6. Maths channel validation: LatG_calc vs measured LatG

Run:
  python phase1_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from pathlib import Path

from generate_sample_data import generate_session
from telemetry_pipeline import TelemetryPipeline, VehicleParams


# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0F0F0F",
    "axes.facecolor":    "#1A1A1A",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#CCCCCC",
    "axes.titlecolor":   "#FFFFFF",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#2A2A2A",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "text.color":        "#CCCCCC",
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  "#333333",
    "font.family":       "monospace",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
})

TOYOTA_RED   = "#E60012"
ACCENT_CYAN  = "#00B4D8"
ACCENT_AMBER = "#F4A261"
ACCENT_GREEN = "#52B788"
GREY_MID     = "#888888"


def load_or_generate(csv_path: str = "data/sample_session.csv") -> pd.DataFrame:
    """Load existing CSV or generate synthetic session."""
    if not Path(csv_path).exists():
        print("No session file found — generating synthetic data...")
        os.makedirs("data", exist_ok=True)
        generate_session(csv_path)
    pipe = TelemetryPipeline(csv_path, verbose=True)
    df   = pipe.process()
    pipe.export("output/processed.csv")
    pipe.export_motec_csv("output/motec_maths.csv")
    pipe.summary()
    return df


def get_lap(df: pd.DataFrame, lap_num: int) -> pd.DataFrame:
    return df[df["LapNumber"] == lap_num].copy().reset_index(drop=True)


def normalise_time(lap_df: pd.DataFrame) -> np.ndarray:
    t = lap_df["Time"].values
    return t - t[0]


# ── Plot 1: Multi-channel lap trace ───────────────────────────────────────────
def plot_lap_trace(df: pd.DataFrame, lap_a: int = 1, lap_b: int = 5) -> plt.Figure:
    """
    Speed, throttle, brake, and steering — two laps overlaid.
    Standard first thing you open in Motec i2.
    """
    la = get_lap(df, lap_a)
    lb = get_lap(df, lap_b)
    ta = normalise_time(la)
    tb = normalise_time(lb)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"LAP TRACE — Lap {lap_a} vs Lap {lap_b}  |  GT3 Performance Analysis",
        color="#FFFFFF", fontsize=11, fontweight="bold", y=0.98
    )
    fig.patch.set_facecolor("#0F0F0F")

    channels = [
        ("Speed",       "km/h",   "Speed",         0,   290),
        ("ThrottlePos", "%",      "Throttle",       0,   110),
        ("BrakePress",  "bar",    "Brake pressure", 0,   110),
        ("SteeringAngle","°",     "Steering angle", -250, 250),
    ]

    colors_a = TOYOTA_RED
    colors_b = ACCENT_CYAN

    for ax, (col, unit, title, ymin, ymax) in zip(axes, channels):
        if col in la.columns:
            ax.plot(ta, la[col], color=colors_a, lw=0.8, alpha=0.9, label=f"Lap {lap_a}")
        if col in lb.columns:
            ax.plot(tb, lb[col], color=colors_b, lw=0.8, alpha=0.7, label=f"Lap {lap_b}")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Plot 2: Maths channel validation ─────────────────────────────────────────
def plot_maths_validation(df: pd.DataFrame, lap_num: int = 3) -> plt.Figure:
    """
    Validates LatG_calc (Milliken bicycle model) vs measured LateralAcc.
    OversteerMetric derived from the difference.
    This is the plot you show Toyota to prove you know what i2 is doing.
    """
    lap = get_lap(df, lap_num)
    t   = normalise_time(lap)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"MATHS CHANNEL VALIDATION — Lap {lap_num}  |  LatG Milliken vs Measured",
        color="#FFFFFF", fontsize=11, fontweight="bold", y=0.98
    )
    fig.patch.set_facecolor("#0F0F0F")

    # Panel 1: LatG_calc vs measured
    ax = axes[0]
    ax.plot(t, lap["LateralAcc"], color=GREY_MID, lw=0.7, alpha=0.8, label="Measured (IMU)")
    ax.plot(t, lap["LatG_calc"],  color=TOYOTA_RED, lw=1.0, alpha=0.9, label="Calculated (Milliken)")
    ax.axhline(0, color="#444444", lw=0.5)
    ax.set_ylabel("Lateral G")
    ax.set_title("Lateral G — measured vs Milliken bicycle model", loc="left", pad=3)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.4)

    # Panel 2: Oversteer metric
    ax = axes[1]
    os_metric = lap["OversteerMetric"].values
    ax.fill_between(t, os_metric, 0,
                    where=(os_metric > 0), color=ACCENT_AMBER, alpha=0.6, label="Oversteer")
    ax.fill_between(t, os_metric, 0,
                    where=(os_metric < 0), color=ACCENT_CYAN, alpha=0.5, label="Understeer")
    ax.axhline(0, color="#444444", lw=0.5)
    ax.set_ylabel("OS metric (g)")
    ax.set_title("Oversteer metric  (+ = oversteer, − = understeer)", loc="left", pad=3)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.4)

    # Panel 3: Steering rate
    ax = axes[2]
    ax.plot(t, lap["SteeringRate"], color=ACCENT_GREEN, lw=0.7, alpha=0.9)
    ax.axhline(0, color="#444444", lw=0.5)
    ax.set_ylabel("°/s")
    ax.set_xlabel("Time (s)")
    ax.set_title("Steering angle rate  (Motec maths channel)", loc="left", pad=3)
    ax.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Plot 3: Traction circle (preview) ────────────────────────────────────────
def plot_traction_circle(df: pd.DataFrame, lap_num: int = 1) -> plt.Figure:
    """
    G-G diagram coloured by speed — preview of Phase 3.
    Shows grip utilisation and cornering balance at a glance.
    """
    lap = get_lap(df, lap_num)

    lat   = lap["LateralAcc"].values
    lon   = lap["LonG_calc"].values
    speed = lap["Speed"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(
        f"TRACTION CIRCLE — Lap {lap_num}  |  Coloured by speed (km/h)",
        color="#FFFFFF", fontsize=11, fontweight="bold"
    )
    fig.patch.set_facecolor("#0F0F0F")

    # Colour by speed using a speed-coloured scatter
    sc = ax.scatter(lat, lon, c=speed, cmap="plasma", s=1.5, alpha=0.5, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Speed (km/h)", color="#CCCCCC")
    cbar.ax.yaxis.set_tick_params(color="#CCCCCC")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#CCCCCC")

    # Grip circle overlay
    theta = np.linspace(0, 2 * np.pi, 200)
    r = VehicleParams().grip_limit_g
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color="#444444", lw=1.2, ls="--", label=f"Grip limit ({r}g)")

    ax.axhline(0, color="#333333", lw=0.5)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_xlabel("Lateral G")
    ax.set_ylabel("Longitudinal G")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Plot 4: Lateral load transfer ─────────────────────────────────────────────
def plot_load_transfer(df: pd.DataFrame, lap_num: int = 2) -> plt.Figure:
    """Front vs rear lateral load transfer — key for setup development discussion."""
    lap = get_lap(df, lap_num)
    t   = normalise_time(lap)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"LATERAL LOAD TRANSFER — Lap {lap_num}  |  Front vs Rear Axle",
        color="#FFFFFF", fontsize=11, fontweight="bold", y=0.98
    )
    fig.patch.set_facecolor("#0F0F0F")

    ax = axes[0]
    ax.plot(t, lap["LateralLoadTransferFront"], color=TOYOTA_RED, lw=0.8, label="Front axle")
    ax.plot(t, lap["LateralLoadTransferRear"],  color=ACCENT_CYAN, lw=0.8, alpha=0.8, label="Rear axle")
    ax.set_ylabel("ΔLoad (N)")
    ax.set_title("Lateral load transfer per axle", loc="left", pad=3)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, lap["BalanceIndex"], color=ACCENT_AMBER, lw=0.8)
    ax.axhline(VehicleParams().front_weight_dist,
               color="#555555", lw=1.0, ls="--", label="Static front bias (41.5%)")
    ax.fill_between(t, lap["BalanceIndex"], VehicleParams().front_weight_dist,
                    alpha=0.2, color=ACCENT_AMBER)
    ax.set_ylim(0.2, 0.8)
    ax.set_ylabel("Balance index")
    ax.set_xlabel("Time (s)")
    ax.set_title("Balance index  (0 = rear-dominated, 1 = front-dominated)", loc="left", pad=3)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("output", exist_ok=True)

    print("\n" + "═" * 56)
    print("  PHASE 1 — GT3 TELEMETRY PIPELINE & MATHS CHANNELS")
    print("═" * 56)

    df = load_or_generate()

    print("\nGenerating plots...")
    plots = [
        (plot_lap_trace(df, lap_a=1, lap_b=5),    "output/01_lap_trace.png"),
        (plot_maths_validation(df, lap_num=3),      "output/02_maths_validation.png"),
        (plot_traction_circle(df, lap_num=1),       "output/03_traction_circle.png"),
        (plot_load_transfer(df, lap_num=2),         "output/04_load_transfer.png"),
    ]

    for fig, path in plots:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {path}")

    print("\n✓ Phase 1 complete.")
    print("  Output files:")
    print("    output/processed.csv      — full processed session")
    print("    output/motec_maths.csv    — Motec-importable maths channels")
    print("    output/01_lap_trace.png   — speed/throttle/brake/steering overlay")
    print("    output/02_maths_validation.png — LatG Milliken vs measured")
    print("    output/03_traction_circle.png  — G-G preview (Phase 3 expands this)")
    print("    output/04_load_transfer.png    — front/rear LLT + balance index\n")


if __name__ == "__main__":
    main()
