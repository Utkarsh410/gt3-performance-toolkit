"""
tyre_temperature_analysis.py
-----------------------------
Phase 2 — Tyre temperature operating window classifier + degradation model.

What this module does:
  1. Loads processed session telemetry (from Phase 1 pipeline)
  2. Classifies each sample as COLD / IN_WINDOW / HOT per corner
  3. Computes per-lap window utilisation % (how much of the lap is in window)
  4. Detects tyre temp anomalies (sudden delta spike = possible flat spot / blister)
  5. Degradation regression: lap time delta vs tyre age
     - Linear fit
     - Polynomial fit
     - Predicts crossover lap (when to pit for fresh rubber)
  6. Generates 4 publication-quality plots

GT3 tyre operating windows (Pirelli DHD2 / Michelin slick approximation):
  Optimal inner: 90–115 °C
  Optimal mid:   85–110 °C
  Optimal outer: 80–105 °C
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

warnings.filterwarnings("ignore")

# ── plot style (Motec dark theme) ─────────────────────────────────────────────
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
    "grid.linewidth":    0.4,
    "text.color":        "#CCCCCC",
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  "#444444",
    "font.family":       "monospace",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
})

TOYOTA_RED   = "#E60012"
CYAN         = "#00B4D8"
AMBER        = "#F4A261"
GREEN        = "#52B788"
COLD_COL     = "#4A90D9"
HOT_COL      = "#E63946"
WINDOW_COL   = "#52B788"


# ── GT3 tyre operating windows ────────────────────────────────────────────────
TYRE_WINDOWS = {
    # corner: {zone: (min_C, max_C)}
    "FL": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
    "FR": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
    "RL": {"inner": (88, 112), "mid": (83, 108), "outer": (78, 103)},
    "RR": {"inner": (88, 112), "mid": (83, 108), "outer": (78, 103)},
}

# Channel name map: (corner, zone) → column name in processed CSV
TEMP_CHANNELS = {
    ("FL", "inner"): "TyreTempFL_inner",
    ("FL", "mid"):   "TyreTempFL_mid",
    ("FL", "outer"): "TyreTempFL_outer",
    ("FR", "inner"): "TyreTempFR_inner",
    ("FR", "mid"):   "TyreTempFR_mid",
    ("FR", "outer"): "TyreTempFR_outer",
    ("RL", "inner"): "TyreTempRL_inner",
    ("RL", "mid"):   "TyreTempRL_mid",
    ("RL", "outer"): "TyreTempRL_outer",
    ("RR", "inner"): "TyreTempRR_inner",
    ("RR", "mid"):   "TyreTempRR_mid",
    ("RR", "outer"): "TyreTempRR_outer",
}


class TyreTemperatureAnalyser:
    """
    Classifies tyre temperature state per sample and per lap,
    then fits a degradation model linking lap time to tyre age.

    Parameters
    ----------
    processed_csv : path to the output of TelemetryPipeline.export()
    windows       : dict of operating windows (defaults to GT3 slick)
    """

    def __init__(
        self,
        processed_csv: str | Path = "output/processed.csv",
        windows: dict | None = None,
    ) -> None:
        self.csv     = Path(processed_csv)
        self.windows = windows or TYRE_WINDOWS
        self.df: pd.DataFrame | None = None
        self.lap_summary: pd.DataFrame | None = None

    def load(self) -> "TyreTemperatureAnalyser":
        self.df = pd.read_csv(self.csv, low_memory=False)
        print(f"[TyreAnalyser] Loaded {len(self.df):,} samples from {self.csv.name}")
        return self

    def classify(self) -> "TyreTemperatureAnalyser":
        """
        Add a classification column per corner:
          -1 = COLD (below window), 0 = IN_WINDOW, +1 = HOT (above window)

        Uses the MID zone as the representative temperature per corner
        (best compromise between inner noisy/high and outer conservative).
        """
        df = self.df

        for corner in ["FL", "FR", "RL", "RR"]:
            mid_col = TEMP_CHANNELS.get((corner, "mid"))
            if mid_col not in df.columns:
                continue
            lo, hi = self.windows[corner]["mid"]
            col    = f"State_{corner}"
            df[col] = np.where(
                df[mid_col] < lo, -1,          # COLD
                np.where(df[mid_col] > hi, 1,  # HOT
                0)                             # IN_WINDOW
            )

        # Representative car-level window state: worst corner
        state_cols = [f"State_{c}" for c in ["FL", "FR", "RL", "RR"]
                      if f"State_{c}" in df.columns]
        if state_cols:
            # Most extreme state wins: cold = -1, hot = 1 override in-window = 0
            df["State_car"] = df[state_cols].apply(
                lambda row: 1 if (row == 1).any() else (-1 if (row == -1).any() else 0),
                axis=1
            )

        print(f"[TyreAnalyser] Classification complete.")
        self.df = df
        return self

    def lap_summary_stats(self) -> "TyreTemperatureAnalyser":
        """Per-lap aggregates: window utilisation %, peak temps, mean temps, lap time."""
        df = self.df
        records = []

        for lap_num, lap_df in df.groupby("LapNumber"):
            rec = {"LapNumber": lap_num}

            # Lap time
            rec["LapTime"] = lap_df["LapTime"].iloc[0] if "LapTime" in lap_df else np.nan

            # Tyre age proxy: lap number itself (laps since new set)
            rec["TyreAge"] = lap_num - 1

            # Per-corner mean mid temp + window utilisation
            for corner in ["FL", "FR", "RL", "RR"]:
                mid_col   = TEMP_CHANNELS.get((corner, "mid"))
                state_col = f"State_{corner}"
                if mid_col in lap_df.columns:
                    rec[f"Temp_{corner}_mean"] = lap_df[mid_col].mean()
                    rec[f"Temp_{corner}_max"]  = lap_df[mid_col].max()
                if state_col in lap_df.columns:
                    rec[f"InWindow_{corner}"] = (lap_df[state_col] == 0).mean() * 100

            # Car-level window utilisation
            if "State_car" in lap_df.columns:
                rec["InWindow_car"] = (lap_df["State_car"] == 0).mean() * 100
                rec["TimeCold_pct"] = (lap_df["State_car"] == -1).mean() * 100
                rec["TimeHot_pct"]  = (lap_df["State_car"] == 1).mean() * 100

            records.append(rec)

        self.lap_summary = pd.DataFrame(records)
        print(f"[TyreAnalyser] Lap summary computed: {len(self.lap_summary)} laps")
        return self

    def detect_anomalies(self, delta_threshold_C: float = 8.0) -> pd.DataFrame:
        """
        Flag samples where temp delta between adjacent samples exceeds threshold.
        Sudden +delta = aggressive corner loading / flat spot onset.
        Sudden -delta = possible blister / delamination.
        """
        df   = self.df.copy()
        flags = []

        for corner in ["FL", "FR", "RL", "RR"]:
            inner_col = TEMP_CHANNELS.get((corner, "inner"))
            if inner_col not in df.columns:
                continue
            delta = df[inner_col].diff().abs()
            spikes = df[delta > delta_threshold_C][["Time", "LapNumber", inner_col]].copy()
            spikes["corner"]  = corner
            spikes["delta_C"] = delta[spikes.index]
            flags.append(spikes)

        if flags:
            anomalies = pd.concat(flags).sort_values("Time")
            print(f"[TyreAnalyser] Anomalies detected: {len(anomalies)} samples "
                  f"(Δ > {delta_threshold_C} °C)")
            return anomalies
        return pd.DataFrame()

    def fit_degradation(self) -> dict:
        """
        Fit lap time vs tyre age:
          1. Linear regression
          2. Polynomial (degree 2) regression
          3. Predict crossover lap (when delta from fresh > 0.5 s — pit trigger)

        Returns dict with fit parameters and crossover prediction.
        """
        ls  = self.lap_summary.dropna(subset=["LapTime", "TyreAge"])
        X   = ls["TyreAge"].values.reshape(-1, 1)
        y   = ls["LapTime"].values
        t0  = y[0]  # reference lap time (fresh tyre)

        # Linear fit
        slope, intercept, r, p, se = linregress(X.ravel(), y)
        linear_pred = slope * X.ravel() + intercept

        # Polynomial fit (degree 2)
        poly_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression())
        ])
        poly_pipe.fit(X, y)
        poly_pred = poly_pipe.predict(X)

        # Crossover lap prediction (0.5 s per lap above reference = pit trigger)
        pit_trigger_s = 0.5
        crossover_laps = []
        for age in range(0, 30):
            pred = slope * age + intercept
            if pred - t0 > pit_trigger_s:
                crossover_laps.append(age)
                break

        crossover = crossover_laps[0] if crossover_laps else None

        results = {
            "slope":         slope,
            "intercept":     intercept,
            "r_squared":     r**2,
            "linear_pred":   linear_pred,
            "poly_pred":     poly_pred,
            "lap_ages":      X.ravel(),
            "lap_times":     y,
            "crossover_lap": crossover,
            "t0":            t0,
        }

        print(f"\n[TyreAnalyser] Degradation model:")
        print(f"  Linear: LapTime = {slope:.4f}·age + {intercept:.3f}  (R² = {r**2:.3f})")
        print(f"  Deg rate: {slope:.3f} s/lap  ({slope*1000:.0f} ms/lap)")
        if crossover:
            print(f"  Crossover lap: {crossover}  (delta > {pit_trigger_s} s from fresh)")
        else:
            print(f"  Crossover: not reached in session")

        return results


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_tyre_temp_trace(analyser: TyreTemperatureAnalyser,
                          lap_num: int = 5) -> plt.Figure:
    """
    4-corner temperature traces for one lap, shaded by operating window state.
    Closest thing to what you'd look at in Race Studio post-session.
    """
    df  = analyser.df
    lap = df[df["LapNumber"] == lap_num].copy()
    t   = lap["Time"].values - lap["Time"].values[0]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"TYRE TEMPERATURE TRACE — Lap {lap_num}  |  Window classification",
        color="white", fontsize=11, fontweight="bold"
    )
    corners = [("FL", "Front Left"), ("FR", "Front Right"),
               ("RL", "Rear Left"),  ("RR", "Rear Right")]
    colors_zone = {"inner": TOYOTA_RED, "mid": CYAN, "outer": AMBER}

    for idx, (corner, label) in enumerate(corners):
        ax = fig.add_subplot(2, 2, idx + 1)
        lo_mid, hi_mid = analyser.windows[corner]["mid"]

        # Window shading
        ax.axhspan(lo_mid, hi_mid, color=WINDOW_COL, alpha=0.10, label="Operating window")
        ax.axhline(lo_mid, color=WINDOW_COL, lw=0.8, ls="--", alpha=0.5)
        ax.axhline(hi_mid, color=WINDOW_COL, lw=0.8, ls="--", alpha=0.5)

        for zone in ["inner", "mid", "outer"]:
            col = TEMP_CHANNELS.get((corner, zone))
            if col in lap.columns:
                ax.plot(t, lap[col], color=colors_zone[zone],
                        lw=0.9, alpha=0.85, label=zone)

        # Colour background by state
        state_col = f"State_{corner}"
        if state_col in lap.columns:
            cold_mask = lap[state_col].values == -1
            hot_mask  = lap[state_col].values == 1
            for mask, col in [(cold_mask, COLD_COL), (hot_mask, HOT_COL)]:
                if mask.any():
                    # Fill spans
                    in_seg = False
                    seg_start = 0
                    for k, v in enumerate(mask):
                        if v and not in_seg:
                            seg_start = t[k]; in_seg = True
                        elif not v and in_seg:
                            ax.axvspan(seg_start, t[k], color=col, alpha=0.08)
                            in_seg = False
                    if in_seg:
                        ax.axvspan(seg_start, t[-1], color=col, alpha=0.08)

        ax.set_title(label, loc="left", pad=3)
        ax.set_ylabel("°C")
        ax.set_xlabel("Time (s)" if idx >= 2 else "")
        ax.set_ylim(60, 135)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_window_utilisation(analyser: TyreTemperatureAnalyser) -> plt.Figure:
    """
    Stacked bar per lap: % time COLD / IN_WINDOW / HOT for the full car.
    At a glance you can see when the tyres came fully online and when they overheated.
    """
    ls  = analyser.lap_summary
    lap_nums = ls["LapNumber"].values

    cold_pct   = ls["TimeCold_pct"].fillna(0).values
    window_pct = ls["InWindow_car"].fillna(0).values
    hot_pct    = ls["TimeHot_pct"].fillna(0).values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "TYRE OPERATING WINDOW — Per-lap utilisation",
        color="white", fontsize=11, fontweight="bold"
    )

    # Stacked bar
    ax = axes[0]
    bar_w = 0.7
    ax.bar(lap_nums, cold_pct,   bar_w, label="Cold (<window)",    color=COLD_COL,   alpha=0.85)
    ax.bar(lap_nums, window_pct, bar_w, bottom=cold_pct,
           label="In window",   color=WINDOW_COL, alpha=0.85)
    ax.bar(lap_nums, hot_pct,    bar_w, bottom=cold_pct + window_pct,
           label="Hot (>window)", color=HOT_COL,   alpha=0.85)

    ax.set_ylabel("% of lap time")
    ax.set_title("Time in each thermal state per lap", loc="left", pad=3)
    ax.set_ylim(0, 100)
    ax.axhline(100, color="#333333", lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Per-corner window utilisation lines
    ax2 = axes[1]
    corner_colors = [TOYOTA_RED, CYAN, AMBER, GREEN]
    for (corner, _), color in zip([("FL", "Front Left"), ("FR", "Front Right"),
                                    ("RL", "Rear Left"),  ("RR", "Rear Right")],
                                   corner_colors):
        col = f"InWindow_{corner}"
        if col in ls.columns:
            ax2.plot(lap_nums, ls[col], color=color, lw=1.5,
                     marker="o", ms=4, label=corner)

    ax2.axhline(90, color="#555555", lw=0.8, ls="--", label="90% target")
    ax2.set_ylabel("% time in window")
    ax2.set_xlabel("Lap number")
    ax2.set_title("Per-corner window utilisation  (target ≥ 90%)", loc="left", pad=3)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_degradation(analyser: TyreTemperatureAnalyser,
                     deg_results: dict) -> plt.Figure:
    """
    Lap time vs tyre age: scatter + linear + polynomial fits + crossover marker.
    The deliverable for "when should we pit?" engineering discussion.
    """
    ls   = analyser.lap_summary
    ages = deg_results["lap_ages"]
    lts  = deg_results["lap_times"]
    lin  = deg_results["linear_pred"]
    poly = deg_results["poly_pred"]
    t0   = deg_results["t0"]
    cross = deg_results["crossover_lap"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "TYRE DEGRADATION MODEL — Lap time vs tyre age",
        color="white", fontsize=11, fontweight="bold"
    )

    # Left: raw + fits
    ax = axes[0]
    ax.scatter(ages, lts, color=CYAN, s=35, zorder=5, label="Measured lap times", alpha=0.9)
    ax.plot(ages, lin,  color=TOYOTA_RED,  lw=2.0, label=f"Linear  (R²={deg_results['r_squared']:.3f})")
    ax.plot(ages, poly, color=AMBER, lw=1.5, ls="--", label="Polynomial (deg 2)")

    # Crossover marker
    if cross is not None:
        cross_time = deg_results["slope"] * cross + deg_results["intercept"]
        ax.axvline(cross, color=HOT_COL, lw=1.5, ls="--", alpha=0.8)
        ax.plot(cross, cross_time, "r*", ms=14, zorder=6, label=f"Pit trigger (Lap {cross})")
        ax.annotate(f"Δ > 0.5 s\nPit lap {cross}",
                    xy=(cross, cross_time),
                    xytext=(cross + 0.3, cross_time + 0.03),
                    color=HOT_COL, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=HOT_COL, lw=0.8))

    ax.axhline(t0 + 0.5, color="#555555", lw=0.8, ls=":", label="t₀ + 0.5 s trigger")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Degradation regression", loc="left", pad=3)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: rolling delta from fresh
    ax2 = axes[1]
    delta_lt = lts - t0
    delta_lin = lin - lin[0]
    ax2.bar(ages, delta_lt, 0.6, color=CYAN, alpha=0.75, label="Measured Δ")
    ax2.plot(ages, delta_lin, color=TOYOTA_RED, lw=2.0, label="Linear trend")
    ax2.axhline(0.5, color=HOT_COL, lw=1.2, ls="--", label="0.5 s pit threshold")
    ax2.axhline(0.0, color="#444444", lw=0.5)

    if cross is not None:
        ax2.axvline(cross, color=HOT_COL, lw=1.0, ls="--", alpha=0.7)

    ax2.set_xlabel("Tyre age (laps)")
    ax2.set_ylabel("Δ lap time vs fresh (s)")
    ax2.set_title("Delta from fresh tyre", loc="left", pad=3)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Annotation box
    info = (
        f"Deg rate:  {deg_results['slope']*1000:.0f} ms/lap\n"
        f"R²:        {deg_results['r_squared']:.3f}\n"
        f"Crossover: Lap {cross if cross else 'N/A'}"
    )
    ax.text(0.98, 0.05, info, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(facecolor="#1A1A1A", edgecolor="#444444",
                      boxstyle="round,pad=0.4"),
            color="#CCCCCC")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_temp_heatmap(analyser: TyreTemperatureAnalyser) -> plt.Figure:
    """
    Heatmap: mean tyre temperature per corner per lap.
    Rows = corners, columns = laps. Green = in window, blue = cold, red = hot.
    Mirrors the format a performance engineer uses in a post-race debrief.
    """
    ls      = analyser.lap_summary
    corners = ["FL", "FR", "RL", "RR"]
    laps    = ls["LapNumber"].values

    temp_data = np.zeros((4, len(laps)))
    for i, corner in enumerate(corners):
        col = f"Temp_{corner}_mean"
        if col in ls.columns:
            temp_data[i, :] = ls[col].values

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle(
        "TYRE TEMPERATURE HEATMAP — Mean mid temp per corner per lap",
        color="white", fontsize=11, fontweight="bold"
    )

    # Custom colourmap: blue (cold) → green (window) → red (hot)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "tyre",
        [(0.0, "#2B5EA7"), (0.35, "#4A90D9"),
         (0.45, "#52B788"), (0.55, "#52B788"),
         (0.65, "#F4A261"), (1.0, "#E63946")],
        N=256
    )

    im = ax.imshow(temp_data, aspect="auto", cmap=cmap, vmin=65, vmax=125,
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("°C", color="#CCCCCC")
    cbar.ax.yaxis.set_tick_params(color="#CCCCCC")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#CCCCCC")

    # Window boundary markers on colourbar
    cbar.ax.axhline((90 - 65) / (125 - 65), color="white", lw=0.8, ls="--")
    cbar.ax.axhline((110 - 65) / (125 - 65), color="white", lw=0.8, ls="--")

    ax.set_yticks(range(4))
    ax.set_yticklabels(corners, color="#CCCCCC")
    ax.set_xticks(range(len(laps)))
    ax.set_xticklabels([f"L{int(l)}" for l in laps], color="#888888", fontsize=8)
    ax.set_xlabel("Lap number")
    ax.set_title("Blue = cold  |  Green = in window  |  Red = hot", loc="left", pad=3)

    # Annotate values
    for i in range(4):
        for j in range(len(laps)):
            val = temp_data[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("output", exist_ok=True)

    print("\n" + "═" * 56)
    print("  PHASE 2 — TYRE TEMPERATURE ANALYSIS")
    print("  Operating window classifier + degradation model")
    print("═" * 56)

    # Check processed CSV exists, generate if not
    if not Path("output/processed.csv").exists():
        print("Processed CSV not found — running Phase 1 pipeline first...")
        from phase1_analysis import load_or_generate
        load_or_generate()

    # Build analyser
    analyser = (
        TyreTemperatureAnalyser("output/processed.csv")
        .load()
        .classify()
        .lap_summary_stats()
    )

    # Anomaly detection
    anomalies = analyser.detect_anomalies(delta_threshold_C=6.0)
    if not anomalies.empty:
        print(f"\nTop anomalies (highest delta):")
        top = anomalies.nlargest(5, "delta_C")[["LapNumber", "corner", "delta_C"]]
        print(top.to_string(index=False))

    # Degradation fit
    deg_results = analyser.fit_degradation()

    # Print per-lap summary
    print("\nPer-lap summary:")
    summary_cols = ["LapNumber", "LapTime", "InWindow_car", "TimeCold_pct", "TimeHot_pct"]
    available = [c for c in summary_cols if c in analyser.lap_summary.columns]
    print(analyser.lap_summary[available].to_string(index=False, float_format="{:.2f}".format))

    # Generate plots
    print("\nGenerating plots...")
    plots = [
        (plot_tyre_temp_trace(analyser, lap_num=5),        "output/phase2_05_temp_trace.png"),
        (plot_window_utilisation(analyser),                 "output/phase2_06_window_utilisation.png"),
        (plot_degradation(analyser, deg_results),           "output/phase2_07_degradation.png"),
        (plot_temp_heatmap(analyser),                       "output/phase2_08_temp_heatmap.png"),
    ]

    for fig, path in plots:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {path}")

    print("\n✓ Phase 2 Python analysis complete.")
    print("\n  MATLAB file: pacejka_magic_formula.m")
    print("  Run in MATLAB to generate Fy/Fx curves and friction ellipse.")
    print("  No external toolboxes required — pure MATLAB script.\n")


if __name__ == "__main__":
    main()
