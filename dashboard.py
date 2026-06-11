"""
dashboard.py
------------
GT3 Performance Analysis Toolkit — Interactive Streamlit Dashboard

Run:
  streamlit run dashboard.py

Features:
  Sidebar  : upload CSV or use synthetic data / select compound
  Tab 1    : Session overview + lap time table
  Tab 2    : Speed & channel traces (lap selector)
  Tab 3    : G-G diagram (KDE + sector split)
  Tab 4    : Tyre temperature (heatmap + window utilisation)
  Tab 5    : Degradation model
  Tab 6    : Maths channel validation (LatG Milliken vs measured)
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import gaussian_kde, linregress

warnings.filterwarnings("ignore")

import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GT3 Performance Analysis Toolkit",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── dark style override ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .reportview-container { background: #0F0F0F; }
  .main .block-container { padding-top: 1rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      background-color: #1A1A1A;
      color: #AAAAAA;
      border-radius: 4px 4px 0 0;
      padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
      background-color: #E60012 !important;
      color: white !important;
  }
  .metric-card {
      background: #1A1A1A;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 12px 16px;
      margin: 4px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F0F0F", "axes.facecolor": "#1A1A1A",
    "axes.edgecolor": "#333", "axes.labelcolor": "#CCC",
    "axes.titlecolor": "#FFF", "xtick.color": "#888",
    "ytick.color": "#888", "grid.color": "#2A2A2A",
    "grid.linestyle": "--", "grid.linewidth": 0.4,
    "text.color": "#CCC", "legend.facecolor": "#1A1A1A",
    "legend.edgecolor": "#444", "font.family": "monospace",
    "font.size": 9, "axes.titleweight": "bold",
})

RED = "#E60012"; CYAN = "#00B4D8"; AMBER = "#F4A261"; GREEN = "#52B788"

COMPOUND_WINDOWS = {
    "SC":   {"FL": {"mid": (80, 105)}, "FR": {"mid": (80, 105)},
             "RL": {"mid": (78, 103)}, "RR": {"mid": (78, 103)}},
    "DC":   {"FL": {"mid": (83, 108)}, "FR": {"mid": (83, 108)},
             "RL": {"mid": (81, 106)}, "RR": {"mid": (81, 106)}},
    "DHD":  {"FL": {"mid": (85, 110)}, "FR": {"mid": (85, 110)},
             "RL": {"mid": (83, 108)}, "RR": {"mid": (83, 108)}},
    "DHD2": {"FL": {"mid": (87, 113)}, "FR": {"mid": (87, 113)},
             "RL": {"mid": (85, 110)}, "RR": {"mid": (85, 110)}},
}

TEMP_CH = {
    "FL": "TyreTempFL_mid", "FR": "TyreTempFR_mid",
    "RL": "TyreTempRL_mid", "RR": "TyreTempRR_mid",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def get_lap(df, lap_num):
    return df[df["LapNumber"] == lap_num].reset_index(drop=True)


def lap_time(df, lap_num):
    sub = df[df["LapNumber"] == lap_num]
    return sub["LapTime"].iloc[0] if "LapTime" in sub and len(sub) > 0 else np.nan


def best_lap_num(df):
    return int(df.groupby("LapNumber")["LapTime"].first().idxmin())


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Processing telemetry...")
def load_and_process(file_bytes, filename):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    from telemetry_pipeline import TelemetryPipeline
    pipe = TelemetryPipeline(tmp, verbose=False)
    df   = pipe.process()
    os.unlink(tmp)
    return df


@st.cache_data(show_spinner="Generating synthetic session...")
def get_synthetic():
    import os; os.makedirs("data", exist_ok=True)
    from generate_sample_data import generate_session
    generate_session("data/sample_session.csv")
    from telemetry_pipeline import TelemetryPipeline
    pipe = TelemetryPipeline("data/sample_session.csv", verbose=False)
    return pipe.process()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏁 GT3 Toolkit")
    st.markdown("**Toyota Racing GmbH**  \nPerformance Analysis Dashboard")
    st.markdown("---")

    data_src = st.radio("Data source", ["Synthetic session", "Upload CSV"])
    if data_src == "Upload CSV":
        uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"])
        if uploaded:
            df = load_and_process(uploaded.read(), uploaded.name)
            st.success(f"Loaded: {uploaded.name}")
        else:
            st.info("Using synthetic data until file is uploaded.")
            df = get_synthetic()
    else:
        df = get_synthetic()

    compound = st.selectbox("Tyre compound", ["SC", "DC", "DHD", "DHD2"], index=2)
    st.markdown("---")
    all_laps = sorted(df["LapNumber"].dropna().unique().astype(int).tolist())
    selected_laps = st.multiselect("Laps to analyse", all_laps, default=all_laps)
    if not selected_laps:
        selected_laps = all_laps
    ref_lap = st.selectbox("Reference lap", all_laps,
                            index=all_laps.index(best_lap_num(df))
                            if best_lap_num(df) in all_laps else 0)
    st.markdown("---")
    st.caption(f"Samples: {len(df):,}  |  {len(all_laps)} laps  |  100 Hz")
    st.caption("Utkarsh Chaudhari · IIT Kharagpur")


# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#E60012; font-family:monospace; margin-bottom:0'>GT3 PERFORMANCE ANALYSIS TOOLKIT</h2>"
    "<p style='color:#888; font-family:monospace; margin-top:4px'>Toyota Racing GmbH Customer Programme  ·  Portfolio Project</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Session Overview",
    "📈 Channel Traces",
    "⭕ G-G Diagram",
    "🌡️ Tyre Temperatures",
    "📉 Degradation",
    "🔬 Maths Validation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SESSION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    best  = best_lap_num(df)
    best_t = lap_time(df, best)
    n_laps = len(all_laps)
    v_max  = df["Speed"].max()
    lat_pk = df["LateralAcc"].abs().max() if "LateralAcc" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best lap", f"{best_t:.3f} s", f"Lap {best}")
    c2.metric("Laps", str(n_laps))
    c3.metric("Peak speed", f"{v_max:.0f} km/h")
    c4.metric("Peak lateral G", f"{lat_pk:.2f} g")

    st.markdown("---")
    st.subheader("Lap Time Summary")

    rows = []
    for lap in all_laps:
        lt     = lap_time(df, lap)
        delta  = lt - best_t
        lap_df = df[df["LapNumber"] == lap]
        pk_lat = lap_df["LateralAcc"].abs().max() if "LateralAcc" in lap_df else 0
        lon_ch = "LonG_calc" if "LonG_calc" in lap_df else "LongAcc"
        pk_lon = lap_df[lon_ch].abs().max() if lon_ch in lap_df else 0
        rows.append({
            "Lap": lap,
            "Time (s)": round(lt, 3),
            "Δ Best (s)": round(delta, 3) if delta > 0 else "REF",
            "Peak Lat G": round(pk_lat, 2),
            "Peak Brake G": round(pk_lon, 2),
        })

    lap_df_table = pd.DataFrame(rows)
    st.dataframe(
        lap_df_table.style
        .highlight_min(subset=["Time (s)"], color="#1C3B1C")
        .format({"Time (s)": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("Speed Trace — All Selected Laps")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    cmap = plt.cm.get_cmap("RdYlGn", len(selected_laps))
    for i, lap in enumerate(selected_laps):
        sub = get_lap(df, lap)
        t   = sub["Time"].values - sub["Time"].values[0]
        ax.plot(t, sub["Speed"], color=cmap(i / max(len(selected_laps)-1, 1)),
                lw=0.8, alpha=0.8, label=f"L{lap} {lap_time(df,lap):.3f}s")
    ax.set_ylabel("Speed (km/h)"); ax.set_xlabel("Time (s)")
    ax.set_title("Speed trace overlay — green = faster, red = slower", loc="left")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.image(fig_to_buf(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHANNEL TRACES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)
    lap_a = c1.selectbox("Lap A", all_laps, index=0, key="ta_a")
    lap_b = c2.selectbox("Lap B", all_laps,
                          index=min(4, len(all_laps)-1), key="ta_b")

    channels_available = {
        "Speed (km/h)":       "Speed",
        "Throttle (%)":       "ThrottlePos",
        "Brake (bar)":        "BrakePress",
        "Steering angle (°)": "SteeringAngle",
        "Lateral G":          "LateralAcc",
        "Long G (calc)":      "LonG_calc",
        "Oversteer metric":   "OversteerMetric",
        "Steering rate (°/s)":"SteeringRate",
    }
    sel_ch = st.multiselect(
        "Channels to plot",
        list(channels_available.keys()),
        default=["Speed (km/h)", "Throttle (%)", "Brake (bar)"]
    )

    if sel_ch:
        fig, axes = plt.subplots(len(sel_ch), 1, figsize=(12, 2.6 * len(sel_ch)),
                                  sharex=True)
        if len(sel_ch) == 1:
            axes = [axes]

        la = get_lap(df, lap_a)
        lb = get_lap(df, lap_b)
        ta = la["Time"].values - la["Time"].values[0]
        tb = lb["Time"].values - lb["Time"].values[0]

        for ax, ch_label in zip(axes, sel_ch):
            col = channels_available[ch_label]
            if col in la.columns:
                ax.plot(ta, la[col], color=RED, lw=0.85, alpha=0.9,
                        label=f"Lap {lap_a}  {lap_time(df,lap_a):.3f}s")
            if col in lb.columns:
                ax.plot(tb, lb[col], color=CYAN, lw=0.75, alpha=0.8,
                        label=f"Lap {lap_b}  {lap_time(df,lap_b):.3f}s")
            ax.set_ylabel(ch_label, fontsize=8)
            ax.axhline(0, color="#333", lw=0.4)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)
    else:
        st.info("Select at least one channel above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — G-G DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    gg_mode = st.radio("View", ["KDE Density (all selected laps)",
                                 "Sector split (single lap)",
                                 "Lap consistency envelope"],
                        horizontal=True)

    subset  = df[df["LapNumber"].isin(selected_laps)]
    lat_all = subset["LateralAcc"].values if "LateralAcc" in subset else np.zeros(10)
    lon_all = (subset["LonG_calc"].values if "LonG_calc" in subset
               else subset["LongAcc"].values if "LongAcc" in subset
               else np.zeros(10))

    if gg_mode == "KDE Density (all selected laps)":
        fig, ax = plt.subplots(figsize=(7, 7))
        idx  = np.random.choice(len(lat_all), min(6000, len(lat_all)), replace=False)
        if len(idx) > 50:
            kde  = gaussian_kde(np.vstack([lat_all[idx], lon_all[idx]]), bw_method=0.08)
            xi   = np.linspace(-3.2, 3.2, 200)
            yi   = np.linspace(-3.2, 3.2, 200)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            ax.contourf(Xi, Yi, Zi, levels=25, cmap="plasma",
                        vmin=Zi.max() * 0.02)
            ax.contour(Xi, Yi, Zi, levels=8, colors="white", alpha=0.1, linewidths=0.3)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(2.8*np.cos(theta), 2.8*np.sin(theta),
                color="#555", lw=1.2, ls="--", label="Grip limit (2.8g)")
        ax.plot(2.1*np.cos(theta), 2.1*np.sin(theta),
                color="#333", lw=0.7, ls=":", label="75% util")
        ax.axhline(0, color="#2A2A2A", lw=0.5); ax.axvline(0, color="#2A2A2A", lw=0.5)
        for txt, xy in [("BRAKE",(-0.3,-2.8)),("DRIVE",(-0.3,2.65)),
                         ("LEFT",(-3.0,0.1)),  ("RIGHT",(2.4,0.1))]:
            ax.text(*xy, txt, color="#444", fontsize=8)
        ax.set_xlabel("Lateral G"); ax.set_ylabel("Longitudinal G")
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.2)
        comb = np.sqrt(lat_all**2 + lon_all**2)
        ax.text(0.02, 0.02,
                f"Mean util: {(comb/2.8).mean():.1%}\n>90%: {(comb/2.8>0.9).mean():.1%}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="#1A1A1A", edgecolor="#333", boxstyle="round,pad=0.3"))
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)

    elif gg_mode == "Sector split (single lap)":
        sec_lap = st.selectbox("Select lap", all_laps, key="gg_sec")
        lap     = get_lap(df, sec_lap)
        lat     = lap["LateralAcc"].values if "LateralAcc" in lap else np.zeros(len(lap))
        lon     = (lap["LonG_calc"].values if "LonG_calc" in lap
                   else lap["LongAcc"].values if "LongAcc" in lap else np.zeros(len(lap)))
        brake   = lon < -0.3
        corner  = (lon >= -0.3) & (lon <= 0.2)
        trac    = lon > 0.2

        fig, ax = plt.subplots(figsize=(7, 7))
        for mask, color, label in [(brake, RED, "Braking"),
                                    (corner, CYAN, "Cornering"),
                                    (trac, GREEN, "Traction")]:
            if mask.sum() > 5:
                ax.scatter(lat[mask], lon[mask], s=1.5, alpha=0.5,
                           color=color, label=f"{label} ({mask.sum():,})")
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(2.8*np.cos(theta), 2.8*np.sin(theta), color="#555", lw=1.2, ls="--")
        ax.axhline(0, color="#2A2A2A", lw=0.5); ax.axvline(0, color="#2A2A2A", lw=0.5)
        ax.set_xlabel("Lateral G"); ax.set_ylabel("Longitudinal G")
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_aspect("equal")
        ax.legend(fontsize=8, markerscale=5, loc="upper right"); ax.grid(True, alpha=0.2)
        ax.set_title(f"Lap {sec_lap}  {lap_time(df,sec_lap):.3f} s", loc="left")
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)

    else:  # consistency
        cmap2 = plt.cm.get_cmap("RdYlGn", len(selected_laps))
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, ln in enumerate(selected_laps):
            lap = get_lap(df, ln)
            la  = lap["LateralAcc"].values if "LateralAcc" in lap else np.zeros(len(lap))
            lo  = (lap["LonG_calc"].values if "LonG_calc" in lap
                   else lap["LongAcc"].values if "LongAcc" in lap else np.zeros(len(lap)))
            bands = np.linspace(-2.8, 2.8, 40)
            ep, en = [], []
            for lo_b, hi_b in zip(bands[:-1], bands[1:]):
                m = (lo >= lo_b) & (lo < hi_b)
                ep.append(la[m].max() if m.sum() > 2 else np.nan)
                en.append(la[m].min() if m.sum() > 2 else np.nan)
            mid = (bands[:-1] + bands[1:]) / 2
            col = cmap2(i / max(len(selected_laps)-1, 1))
            ax.plot(ep, mid, color=col, lw=1.0, alpha=0.8,
                    label=f"L{ln} {lap_time(df,ln):.3f}s")
            ax.plot(en, mid, color=col, lw=1.0, alpha=0.8)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(2.8*np.cos(theta), 2.8*np.sin(theta), color="#444", lw=1.0, ls="--")
        ax.axhline(0, color="#2A2A2A", lw=0.5); ax.axvline(0, color="#2A2A2A", lw=0.5)
        ax.set_xlabel("Lateral G"); ax.set_ylabel("Longitudinal G")
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right", ncol=2); ax.grid(True, alpha=0.2)
        ax.set_title("Green = faster, Red = slower", loc="left", fontsize=8, color="#888")
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TYRE TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    windows = COMPOUND_WINDOWS[compound]
    st.caption(f"Compound: **{compound}** · FL/FR mid window: "
               f"{windows['FL']['mid'][0]}–{windows['FL']['mid'][1]} °C  "
               f"· RL/RR: {windows['RL']['mid'][0]}–{windows['RL']['mid'][1]} °C")

    # Heatmap
    st.subheader("Tyre Temperature Heatmap")
    corners = ["FL", "FR", "RL", "RR"]
    lap_means = []
    for lap in all_laps:
        sub = df[df["LapNumber"] == lap]
        row = {}
        for c in corners:
            col = TEMP_CH.get(c)
            row[c] = sub[col].mean() if col in sub else np.nan
        lap_means.append(row)
    hm_df = pd.DataFrame(lap_means, index=[f"L{l}" for l in all_laps]).T

    cmap_tyre = LinearSegmentedColormap.from_list(
        "tyre", [(0,"#2B5EA7"),(0.38,"#52B788"),(0.52,"#52B788"),
                 (0.68,"#F4A261"),(1.0,"#E63946")], N=256)

    fig, ax = plt.subplots(figsize=(max(8, len(all_laps)*0.9), 2.5))
    data = hm_df.values
    im   = ax.imshow(data, aspect="auto", cmap=cmap_tyre,
                     vmin=65, vmax=130, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="°C", shrink=0.85)
    ax.set_yticks(range(4)); ax.set_yticklabels(corners)
    ax.set_xticks(range(len(all_laps)))
    ax.set_xticklabels([f"L{l}" for l in all_laps], fontsize=7)
    for i in range(4):
        for j in range(len(all_laps)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
    fig.tight_layout()
    st.image(fig_to_buf(fig), use_container_width=True)

    # Window utilisation per lap
    st.subheader("Window Utilisation Per Lap")
    win_rows = []
    for lap in all_laps:
        sub = df[df["LapNumber"] == lap]
        row = {"Lap": lap}
        for c in corners:
            col = TEMP_CH.get(c)
            if col in sub:
                lo, hi = windows[c]["mid"]
                in_w = ((sub[col] >= lo) & (sub[col] <= hi)).mean() * 100
                row[f"{c} %"] = round(in_w, 1)
        win_rows.append(row)
    win_df = pd.DataFrame(win_rows)
    st.dataframe(win_df.set_index("Lap"), use_container_width=True)

    # Trace for selected lap
    st.subheader("Temperature Trace")
    tr_lap = st.selectbox("Lap", all_laps, key="temp_tr")
    sub    = get_lap(df, tr_lap)
    t      = sub["Time"].values - sub["Time"].values[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
    corner_colors = {"inner": RED, "mid": CYAN, "outer": AMBER}
    for ax, corner in zip(axes.ravel(), corners):
        lo, hi = windows[corner]["mid"]
        ax.axhspan(lo, hi, color=GREEN, alpha=0.08)
        ax.axhline(lo, color=GREEN, lw=0.7, ls="--", alpha=0.5)
        ax.axhline(hi, color=GREEN, lw=0.7, ls="--", alpha=0.5)
        for zone, col in corner_colors.items():
            ch = f"TyreTemp{corner}_{zone}"
            if ch in sub.columns:
                ax.plot(t, sub[ch], color=col, lw=0.85, alpha=0.85, label=zone)
        ax.set_title(corner, loc="left", pad=2)
        ax.set_ylabel("°C"); ax.set_ylim(55, 140)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    st.image(fig_to_buf(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    lt_series = df.groupby("LapNumber")["LapTime"].first()
    ages      = (lt_series.index.values - lt_series.index.min()).astype(float)
    times_arr = lt_series.values.astype(float)
    valid     = ~np.isnan(times_arr)
    ages_v, times_v = ages[valid], times_arr[valid]

    if len(ages_v) >= 2:
        slope, intercept, r, _, _ = linregress(ages_v, times_v)
        pred = slope * ages_v + intercept
        cross = next((int(a) for a in range(30)
                      if slope*a + intercept - times_v[0] > 0.5), None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Deg rate", f"{slope*1000:.0f} ms/lap")
        c2.metric("R²", f"{r**2:.3f}")
        c3.metric("Crossover lap", str(cross) if cross else "N/A")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.scatter(ages_v, times_v, color=CYAN, s=30, zorder=5, label="Lap times")
        ax.plot(ages_v, pred, color=RED, lw=2, label=f"Linear (R²={r**2:.3f})")
        ax.axhline(times_v[0] + 0.5, color="#555", lw=0.8, ls=":", label="+0.5 s threshold")
        if cross:
            ax.axvline(cross, color=RED, lw=1.0, ls="--", alpha=0.7)
            ax.text(cross+0.1, times_v[0]+0.55, f"Pit Lap {cross}", color=RED, fontsize=8)
        ax.set_xlabel("Tyre age (laps)"); ax.set_ylabel("Lap time (s)")
        ax.set_title("Degradation regression", loc="left")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        delta = times_v - times_v[0]
        ax2.bar(ages_v, delta, 0.6, color=CYAN, alpha=0.75, label="Measured Δ")
        ax2.plot(ages_v, pred - pred[0], color=RED, lw=2, label="Linear trend")
        ax2.axhline(0.5, color=RED, lw=1.0, ls="--", alpha=0.7, label="Pit threshold")
        ax2.axhline(0, color="#444", lw=0.4)
        ax2.set_xlabel("Tyre age (laps)"); ax2.set_ylabel("Δ from fresh (s)")
        ax2.set_title("Delta from fresh tyre", loc="left")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)
    else:
        st.warning("Need at least 2 laps for degradation analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MATHS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    **What this shows:** `LatG_calc` is computed from the steering angle via the
    Milliken bicycle model formula — independently of the IMU lateral accelerometer.
    Agreement between the two validates the pipeline. The gap between them is the
    **oversteer metric**: positive = rear sliding, negative = front pushing.
    """)

    val_lap = st.selectbox("Select lap", all_laps, key="val_lap")
    lap     = get_lap(df, val_lap)
    t       = lap["Time"].values - lap["Time"].values[0]

    has_calc = "LatG_calc" in lap.columns and "LateralAcc" in lap.columns
    has_os   = "OversteerMetric" in lap.columns
    has_sr   = "SteeringRate" in lap.columns

    if has_calc:
        n_panels = 1 + has_os + has_sr
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.8*n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        ax = axes[0]
        ax.plot(t, lap["LateralAcc"], color="#888", lw=0.7, alpha=0.8, label="Measured (IMU)")
        ax.plot(t, lap["LatG_calc"],  color=RED, lw=1.0, alpha=0.9, label="Calculated (Milliken)")
        ax.axhline(0, color="#333", lw=0.4)
        ax.set_ylabel("Lateral G")
        ax.set_title("LatG: IMU measured vs Milliken bicycle model", loc="left")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

        i = 1
        if has_os:
            ax2 = axes[i]; i += 1
            os = lap["OversteerMetric"].values
            ax2.fill_between(t, os, 0, where=(os>0), color=AMBER, alpha=0.7, label="Oversteer")
            ax2.fill_between(t, os, 0, where=(os<0), color=CYAN, alpha=0.6, label="Understeer")
            ax2.axhline(0, color="#333", lw=0.4)
            ax2.set_ylabel("OS metric (g)")
            ax2.set_title("Oversteer metric  (+ve = rear slides, −ve = front pushes)", loc="left")
            ax2.legend(fontsize=8, loc="upper right"); ax2.grid(True, alpha=0.3)

        if has_sr:
            ax3 = axes[i]
            ax3.plot(t, lap["SteeringRate"], color=GREEN, lw=0.7)
            ax3.axhline(0, color="#333", lw=0.4)
            ax3.set_ylabel("°/s")
            ax3.set_xlabel("Time (s)")
            ax3.set_title("Steering angle rate", loc="left"); ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)
    else:
        st.warning("LatG_calc not found — run the telemetry pipeline (Phase 1) first.")

    st.markdown("---")
    st.markdown(
        "**Interview context:** When Toyota asks *'what Motec maths channels do you use?'* — "
        "this tab is the answer. The pipeline recomputes LatG from steering geometry, "
        "validates against the IMU, and derives oversteer from the residual. "
        "That's exactly what i2's built-in maths channels do — except here you can see the formula."
    )
