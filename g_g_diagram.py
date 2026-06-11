"""
gg_diagram.py
-------------
Phase 3 — Full G-G diagram analysis module.

Builds on Phase 1 traction circle preview with:
  1. KDE density G-G (shows where the driver actually spends time)
  2. Multi-lap overlay with grip limit utilisation stats
  3. Sector-split G-G (braking / cornering / traction zones isolated)
  4. Lap-to-lap G-G comparison (driver consistency metric)
  5. Underutilised grip identification (the setup recommendation output)

This is the primary tool for:
  - Identifying whether the car is traction/braking/cornering limited
  - Quantifying how close to the limit the driver is operating
  - Flagging setup imbalances (front/rear bias visible in G-G shape)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from pathlib import Path

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F0F0F", "axes.facecolor": "#1A1A1A",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#CCCCCC",
    "axes.titlecolor": "#FFFFFF", "xtick.color": "#888888",
    "ytick.color": "#888888", "grid.color": "#2A2A2A",
    "grid.linestyle": "--", "grid.linewidth": 0.4,
    "text.color": "#CCCCCC", "legend.facecolor": "#1A1A1A",
    "legend.edgecolor": "#444444", "font.family": "monospace",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})

TOYOTA_RED = "#E60012"
CYAN       = "#00B4D8"
AMBER      = "#F4A261"
GREEN      = "#52B788"
GRIP_LIMIT = 2.8   # g — matches VehicleParams from Phase 1


def _load_session(csv_path: str = "output/processed.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def _get_lap(df: pd.DataFrame, lap: int) -> pd.DataFrame:
    return df[df["LapNumber"] == lap].reset_index(drop=True)


def _grip_circle(r: float = GRIP_LIMIT, n: int = 300):
    t = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(t), r * np.sin(t)


def _sector_masks(lap_df: pd.DataFrame):
    """
    Split lap into three dynamic phases based on longitudinal G:
      Braking    : LonG_calc < -0.3 g
      Traction   : LonG_calc >  0.2 g
      Cornering  : everything else (peak lateral phase)
    """
    lon = lap_df["LonG_calc"].values if "LonG_calc" in lap_df else lap_df["LongAcc"].values
    return {
        "Braking":   lon < -0.3,
        "Cornering": (lon >= -0.3) & (lon <= 0.2),
        "Traction":  lon > 0.2,
    }


# ── Plot 1: KDE density G-G ───────────────────────────────────────────────────
def plot_gg_density(df: pd.DataFrame,
                    laps: list[int] | None = None) -> plt.Figure:
    """
    Kernel density estimate G-G diagram.
    Dense regions = where the driver spends most time.
    Sparse edge regions = limit events.
    """
    if laps is None:
        laps = sorted(df["LapNumber"].dropna().unique())

    subset = df[df["LapNumber"].isin(laps)]
    lat    = subset["LateralAcc"].values
    lon    = (subset["LonG_calc"].values
              if "LonG_calc" in subset else subset["LongAcc"].values)

    # Downsample for KDE speed
    idx    = np.random.choice(len(lat), min(8000, len(lat)), replace=False)
    lat_s, lon_s = lat[idx], lon[idx]

    kde    = gaussian_kde(np.vstack([lat_s, lon_s]), bw_method=0.08)
    xi     = np.linspace(-3.2, 3.2, 220)
    yi     = np.linspace(-3.2, 3.2, 220)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi     = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle("G-G DIAGRAM — KDE Density  |  GT3 Session",
                 color="white", fontsize=12, fontweight="bold")

    # Density fill
    cmap_gg = plt.cm.get_cmap("plasma").copy()
    cmap_gg.set_under("#0F0F0F")
    cf = ax.contourf(Xi, Yi, Zi, levels=30, cmap=cmap_gg,
                     vmin=Zi.max() * 0.02)
    ax.contour(Xi, Yi, Zi, levels=10, colors="white", alpha=0.12, linewidths=0.4)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.75)
    cbar.set_label("Density", color="#CCCCCC")
    cbar.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888888")

    # Grip circle
    gx, gy = _grip_circle()
    ax.plot(gx, gy, color="#555555", lw=1.4, ls="--", label=f"Grip limit ({GRIP_LIMIT}g)")

    # 75% utilisation circle
    gx2, gy2 = _grip_circle(GRIP_LIMIT * 0.75)
    ax.plot(gx2, gy2, color="#333333", lw=0.8, ls=":", label="75% utilisation")

    # Axis annotations
    ax.axhline(0, color="#2A2A2A", lw=0.6)
    ax.axvline(0, color="#2A2A2A", lw=0.6)
    for txt, xy in [("BRAKE", (-0.2, -2.9)), ("DRIVE", (-0.2, 2.7)),
                    ("LEFT",  (-3.0, 0.1)),  ("RIGHT", (2.5, 0.1))]:
        ax.text(*xy, txt, color="#444444", fontsize=8, fontstyle="italic")

    ax.set_xlabel("Lateral G  (left +ve)")
    ax.set_ylabel("Longitudinal G  (drive +ve)")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.2)

    # Utilisation stats
    combined = np.sqrt(lat**2 + lon**2)
    util_pct  = (combined / GRIP_LIMIT).clip(0, 2)
    stats_txt = (
        f"Samples: {len(lat):,}\n"
        f"Mean util: {util_pct.mean():.1%}\n"
        f">90% grip: {(util_pct > 0.9).mean():.1%}\n"
        f"Peak combined: {combined.max():.2f}g"
    )
    ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes,
            fontsize=8, va="bottom",
            bbox=dict(facecolor="#1A1A1A", edgecolor="#333333",
                      boxstyle="round,pad=0.4"), color="#AAAAAA")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Plot 2: Sector-split G-G ──────────────────────────────────────────────────
def plot_gg_sectors(df: pd.DataFrame, lap_num: int = 3) -> plt.Figure:
    """
    Same G-G but coloured by driving phase.
    Braking zone shape ≠ cornering zone shape — imbalance reveals setup issues.
    """
    lap   = _get_lap(df, lap_num)
    lat   = lap["LateralAcc"].values
    lon   = (lap["LonG_calc"].values if "LonG_calc" in lap
             else lap["LongAcc"].values)
    masks = _sector_masks(lap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"G-G SECTOR ANALYSIS — Lap {lap_num}  |  Phase isolation",
                 color="white", fontsize=11, fontweight="bold")

    # Left: all phases on one plot, coloured
    ax = axes[0]
    phase_colors = {"Braking": TOYOTA_RED, "Cornering": CYAN, "Traction": GREEN}
    for phase, mask in masks.items():
        if mask.sum() > 10:
            ax.scatter(lat[mask], lon[mask], s=1.2, alpha=0.55,
                       color=phase_colors[phase], label=f"{phase} ({mask.sum():,} pts)")

    gx, gy = _grip_circle()
    ax.plot(gx, gy, color="#555555", lw=1.2, ls="--")
    ax.axhline(0, color="#2A2A2A", lw=0.5)
    ax.axvline(0, color="#2A2A2A", lw=0.5)
    ax.set_xlabel("Lateral G")
    ax.set_ylabel("Longitudinal G")
    ax.set_title("Phase overlay", loc="left", pad=3)
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right", markerscale=5)
    ax.grid(True, alpha=0.2)

    # Right: max lateral G per longitudinal G band (friction ellipse envelope)
    ax2 = axes[1]
    lon_bands = np.linspace(-2.5, 2.5, 60)
    max_lat_left  = []
    max_lat_right = []
    for lo, hi in zip(lon_bands[:-1], lon_bands[1:]):
        band = (lon >= lo) & (lon < hi)
        if band.sum() > 5:
            max_lat_left.append(-lat[band].min())
            max_lat_right.append(lat[band].max())
        else:
            max_lat_left.append(np.nan)
            max_lat_right.append(np.nan)

    lon_mid = (lon_bands[:-1] + lon_bands[1:]) / 2
    ax2.fill_betweenx(lon_mid, [-x for x in max_lat_left],
                      max_lat_right, color=CYAN, alpha=0.15, label="Achieved envelope")
    ax2.plot([-x for x in max_lat_left], lon_mid,
             color=CYAN, lw=1.5, alpha=0.8)
    ax2.plot(max_lat_right, lon_mid,
             color=CYAN, lw=1.5, alpha=0.8)

    gx, gy = _grip_circle()
    ax2.plot(gx, gy, color="#555555", lw=1.2, ls="--", label=f"Theoretical limit ({GRIP_LIMIT}g)")

    ax2.axhline(0, color="#2A2A2A", lw=0.5)
    ax2.axvline(0, color="#2A2A2A", lw=0.5)
    ax2.set_xlabel("Lateral G")
    ax2.set_ylabel("Longitudinal G")
    ax2.set_title("Achieved friction envelope vs theoretical limit", loc="left", pad=3)
    ax2.set_xlim(-3.3, 3.3); ax2.set_ylim(-3.3, 3.3)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)

    # Gap annotation — where is the car leaving time?
    ax2.text(0.02, 0.02,
             "Gap to limit = unused grip.\n"
             "Wide gap in braking quadrant\n"
             "→ late braking opportunity.\n"
             "Gap in cornering → setup or\n"
             "   driver confidence issue.",
             transform=ax2.transAxes, fontsize=7, va="bottom",
             color="#888888",
             bbox=dict(facecolor="#1A1A1A", edgecolor="#333333",
                       boxstyle="round,pad=0.3"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Plot 3: Lap consistency comparison ────────────────────────────────────────
def plot_gg_consistency(df: pd.DataFrame,
                         laps: list[int] | None = None) -> plt.Figure:
    """
    G-G envelope per lap overlaid — shows driver consistency.
    Tight cluster = consistent, wide spread = variability.
    """
    if laps is None:
        all_laps = sorted(df["LapNumber"].dropna().unique())
        laps = all_laps[:min(6, len(all_laps))]

    cmap = plt.cm.get_cmap("RdYlGn", len(laps))
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle("G-G LAP CONSISTENCY — Envelope overlay per lap",
                 color="white", fontsize=11, fontweight="bold")

    for i, lap_num in enumerate(laps):
        lap = _get_lap(df, lap_num)
        lat = lap["LateralAcc"].values
        lon = (lap["LonG_calc"].values if "LonG_calc" in lap
               else lap["LongAcc"].values)

        # Convex-hull-like envelope: max |lat| per lon band
        lon_bands = np.linspace(-2.8, 2.8, 40)
        env_lat_pos, env_lat_neg = [], []
        for lo, hi in zip(lon_bands[:-1], lon_bands[1:]):
            band = (lon >= lo) & (lon < hi)
            if band.sum() > 3:
                env_lat_pos.append(lat[band].max())
                env_lat_neg.append(lat[band].min())
            else:
                env_lat_pos.append(np.nan)
                env_lat_neg.append(np.nan)

        lon_mid = (lon_bands[:-1] + lon_bands[1:]) / 2
        color = cmap(i / len(laps))
        lap_time = df[df["LapNumber"] == lap_num]["LapTime"].iloc[0]
        ax.plot(env_lat_pos, lon_mid, color=color, lw=1.2, alpha=0.85,
                label=f"Lap {int(lap_num)}  {lap_time:.3f}s")
        ax.plot(env_lat_neg, lon_mid, color=color, lw=1.2, alpha=0.85)

    gx, gy = _grip_circle()
    ax.plot(gx, gy, color="#444444", lw=1.0, ls="--", label=f"Grip limit")
    ax.axhline(0, color="#2A2A2A", lw=0.5)
    ax.axvline(0, color="#2A2A2A", lw=0.5)
    ax.set_xlabel("Lateral G")
    ax.set_ylabel("Longitudinal G")
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.2)
    ax.set_title("Green = faster lap, Red = slower lap", loc="left", pad=3,
                 fontsize=8, color="#888888")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig