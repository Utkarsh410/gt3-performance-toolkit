"""
session_report.py
-----------------
Phase 3 — Automated GT3 Session Report Generator.

Produces a professional one-page (extendable) PDF report that mirrors
the format a GT3 Performance Engineer would hand to the team after a session:

  Page 1: Session overview
    - Header: car number, session, circuit, date
    - Lap time table with delta, tyre age, tyre window %
    - G-G diagram (best lap)
    - Speed trace (best vs slowest lap)
    - Tyre temp heatmap

  Page 2: Engineering analysis
    - Degradation model chart
    - Oversteer metric trace
    - Setup snapshot table
    - Engineer's recommendation box

Run:
  python session_report.py
  → output/session_report.pdf
"""

from __future__ import annotations

import io
import os
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus.flowables import Flowable

warnings.filterwarnings("ignore")


# ── colours ────────────────────────────────────────────────────────────────────
TOYOTA_RED   = colors.HexColor("#E60012")
DARK_BG      = colors.HexColor("#0F0F0F")
PANEL_BG     = colors.HexColor("#1A1A1A")
TEXT_PRIMARY = colors.HexColor("#FFFFFF")
TEXT_SEC     = colors.HexColor("#AAAAAA")
TEXT_MUTED   = colors.HexColor("#666666")
ACCENT_CYAN  = colors.HexColor("#00B4D8")
ACCENT_GREEN = colors.HexColor("#52B788")
ACCENT_AMBER = colors.HexColor("#F4A261")

MPL_BG   = "#0F0F0F"
MPL_AX   = "#1A1A1A"
MPL_TXT  = "#CCCCCC"
MPL_RED  = "#E60012"
MPL_CYAN = "#00B4D8"
MPL_AMB  = "#F4A261"
MPL_GRN  = "#52B788"

W, H = A4   # 595.27 x 841.89 points


# ── matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": MPL_BG, "axes.facecolor": MPL_AX,
    "axes.edgecolor": "#333", "axes.labelcolor": MPL_TXT,
    "axes.titlecolor": "#FFF", "xtick.color": "#888",
    "ytick.color": "#888", "grid.color": "#2A2A2A",
    "grid.linestyle": "--", "grid.linewidth": 0.4,
    "text.color": MPL_TXT, "legend.facecolor": MPL_AX,
    "legend.edgecolor": "#444", "font.family": "monospace",
    "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
})


# ── helper: matplotlib fig → reportlab Image ──────────────────────────────────
def fig_to_image(fig: plt.Figure, width_mm: float, height_mm: float) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return RLImage(buf, width=width_mm * mm, height=height_mm * mm)


# ── data loaders ──────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed session + lap summary. Generates if needed."""
    if not Path("output/processed.csv").exists():
        print("Running Phase 1 pipeline...")
        from phase1_analysis import load_or_generate
        load_or_generate()

    df = pd.read_csv("output/processed.csv", low_memory=False)

    # Build lap summary
    from tyre_temperature_analysis import TyreTemperatureAnalyser
    an = TyreTemperatureAnalyser("output/processed.csv").load().classify().lap_summary_stats()
    ls = an.lap_summary.copy()
    return df, ls


# ── plot builders ──────────────────────────────────────────────────────────────
def build_speed_trace(df: pd.DataFrame) -> plt.Figure:
    best_lap  = int(df.groupby("LapNumber")["LapTime"].first().idxmin())
    worst_lap = int(df.groupby("LapNumber")["LapTime"].first().idxmax())

    def lap_t(lap):
        sub = df[df["LapNumber"] == lap]
        return sub["Time"].values - sub["Time"].values[0], sub["Speed"].values

    tb, sb = lap_t(best_lap)
    tw, sw = lap_t(worst_lap)

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.plot(tb, sb, color=MPL_RED,  lw=0.9, label=f"Lap {best_lap} (best)")
    ax.plot(tw, sw, color=MPL_CYAN, lw=0.7, alpha=0.75, label=f"Lap {worst_lap} (slowest)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Speed trace — best vs slowest lap", loc="left", pad=2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


def build_gg_best_lap(df: pd.DataFrame) -> plt.Figure:
    best_lap = int(df.groupby("LapNumber")["LapTime"].first().idxmin())
    lap      = df[df["LapNumber"] == best_lap]
    lat      = lap["LateralAcc"].values
    lon      = (lap["LonG_calc"].values if "LonG_calc" in lap.columns
                else lap["LongAcc"].values)
    speed    = lap["Speed"].values

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sc = ax.scatter(lat, lon, c=speed, cmap="plasma", s=1.0, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="km/h", shrink=0.8)

    theta = np.linspace(0, 2 * np.pi, 200)
    r = 2.8
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color="#444", lw=1.0, ls="--")
    ax.axhline(0, color="#333", lw=0.4)
    ax.axvline(0, color="#333", lw=0.4)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral G"); ax.set_ylabel("Long G")
    ax.set_title(f"G-G  (Lap {best_lap})", loc="left", pad=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout(pad=0.4)
    return fig


def build_tyre_heatmap(ls: pd.DataFrame) -> plt.Figure:
    corners  = ["FL", "FR", "RL", "RR"]
    laps     = ls["LapNumber"].values
    data     = np.zeros((4, len(laps)))
    for i, c in enumerate(corners):
        col = f"Temp_{c}_mean"
        if col in ls.columns:
            data[i, :] = ls[col].values

    cmap = LinearSegmentedColormap.from_list(
        "tyre", [(0, "#2B5EA7"), (0.4, "#52B788"), (0.6, "#52B788"),
                 (0.75, "#F4A261"), (1.0, "#E63946")], N=256)

    fig, ax = plt.subplots(figsize=(7.5, 1.6))
    im = ax.imshow(data, aspect="auto", cmap=cmap,
                   vmin=65, vmax=125, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="°C", shrink=0.9)
    ax.set_yticks(range(4)); ax.set_yticklabels(corners, fontsize=7)
    ax.set_xticks(range(len(laps)))
    ax.set_xticklabels([f"L{int(l)}" for l in laps], fontsize=6)
    ax.set_title("Tyre temp heatmap (mid zone mean, °C)", loc="left", pad=2)
    for i in range(4):
        for j in range(len(laps)):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center",
                    fontsize=5.5, color="white", fontweight="bold")
    fig.tight_layout(pad=0.4)
    return fig


def build_degradation_chart(ls: pd.DataFrame) -> plt.Figure:
    from scipy.stats import linregress
    ages  = ls["TyreAge"].values
    times = ls["LapTime"].ffill().values
    slope, intercept, r, _, _ = linregress(ages, times)
    pred  = slope * ages + intercept
    cross = next((a for a in range(30)
                  if slope * a + intercept - times[0] > 0.5), None)

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.scatter(ages, times, color=MPL_CYAN, s=25, zorder=5, label="Lap times")
    ax.plot(ages, pred, color=MPL_RED, lw=1.8,
            label=f"Linear  (R²={r**2:.3f},  {slope*1000:.0f} ms/lap)")
    ax.axhline(times[0] + 0.5, color="#555", lw=0.8, ls=":",
               label="Pit threshold (+0.5 s)")
    if cross:
        ax.axvline(cross, color=MPL_RED, lw=1.0, ls="--", alpha=0.7)
        ax.text(cross + 0.1, times[0] + 0.55,
                f"Pit lap {cross}", color=MPL_RED, fontsize=7)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Tyre degradation", loc="left", pad=2)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


def build_oversteer_trace(df: pd.DataFrame) -> plt.Figure:
    best_lap = int(df.groupby("LapNumber")["LapTime"].first().idxmin())
    lap      = df[df["LapNumber"] == best_lap]
    t   = lap["Time"].values - lap["Time"].values[0]
    os  = lap["OversteerMetric"].values if "OversteerMetric" in lap.columns else np.zeros(len(t))

    fig, ax = plt.subplots(figsize=(7.5, 2.0))
    ax.fill_between(t, os, 0, where=(os > 0), color=MPL_AMB, alpha=0.7, label="Oversteer")
    ax.fill_between(t, os, 0, where=(os < 0), color=MPL_CYAN, alpha=0.6, label="Understeer")
    ax.axhline(0, color="#444", lw=0.5)
    ax.set_ylabel("OS metric (g)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Oversteer metric — Lap {best_lap} (best lap)", loc="left", pad=2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


# ── report tables ──────────────────────────────────────────────────────────────
def lap_table_data(df: pd.DataFrame, ls: pd.DataFrame) -> list[list]:
    header = ["Lap", "Time (s)", "Δ Best (s)", "Tyre Age", "In Window %",
              "Peak Lat G", "Peak Brake G"]
    best = df.groupby("LapNumber")["LapTime"].first().min()
    rows = [header]
    for _, row in ls.iterrows():
        lap    = int(row["LapNumber"])
        lt     = row.get("LapTime", np.nan)
        delta  = lt - best if not np.isnan(lt) else "—"
        age    = int(row.get("TyreAge", 0))
        winpct = row.get("InWindow_car", np.nan)
        lap_df = df[df["LapNumber"] == lap]
        pk_lat = lap_df["LateralAcc"].abs().max() if "LateralAcc" in lap_df else 0
        pk_lon = lap_df["LonG_calc"].abs().max() if "LonG_calc" in lap_df else (
                 lap_df["LongAcc"].abs().max() if "LongAcc" in lap_df else 0)

        rows.append([
            str(lap),
            f"{lt:.3f}" if not np.isnan(lt) else "—",
            f"+{delta:.3f}" if isinstance(delta, float) and delta > 0 else
            ("—" if isinstance(delta, str) else "REF"),
            str(age),
            f"{winpct:.1f}%" if not np.isnan(winpct) else "—",
            f"{pk_lat:.2f}g",
            f"{pk_lon:.2f}g",
        ])
    return rows


def setup_table_data() -> list[list]:
    return [
        ["Parameter", "Value", "Notes"],
        ["Front ARB",      "35 N·m/deg", "Baseline — see sweep analysis"],
        ["Rear ARB",       "25 N·m/deg", "Balanced with front"],
        ["Front spring",   "85 kN/m",    "Medium-stiff GT3 setting"],
        ["Rear spring",    "70 kN/m",    "Softer rear for traction"],
        ["Front camber",   "-3.0°",      "Within GT3 BoP limits"],
        ["Rear camber",    "-1.5°",      "Conservative for stability"],
        ["Tyre compound",  "Slick (DHD2 proxy)", "Operating window 85–115°C"],
        ["Ride height F",  "62 mm",      "Per BoP minimum"],
        ["Ride height R",  "75 mm",      "Standard GT3 rake"],
    ]


# ── style helpers ──────────────────────────────────────────────────────────────
def _ts(cmds): return TableStyle(cmds)

HEADER_STYLE = _ts([
    ("BACKGROUND",  (0, 0), (-1, 0), TOYOTA_RED),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Courier-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0), 7),
    ("BOTTOMPADDING",(0,0), (-1, 0), 4),
    ("TOPPADDING",  (0, 0), (-1, 0), 4),
])

ROW_STYLE = _ts([
    ("FONTNAME",    (0, 1), (-1, -1), "Courier"),
    ("FONTSIZE",    (0, 1), (-1, -1), 7),
    ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#CCCCCC")),
    ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#1A1A1A")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.HexColor("#1A1A1A"), colors.HexColor("#222222")]),
    ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#333333")),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",(0, 0), (-1, -1), 5),
    ("TOPPADDING",  (0, 1), (-1, -1), 3),
    ("BOTTOMPADDING",(0,1),(-1, -1), 3),
])


# ── page background ────────────────────────────────────────────────────────────
class DarkBackground(Flowable):
    """Draws the dark page background."""
    def __init__(self, w, h):
        super().__init__()
        self.w, self.h = w, h

    def draw(self):
        self.canv.setFillColor(DARK_BG)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)


def on_page(canvas, doc):
    """Called on every page — draws background and footer."""
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Red header bar
    canvas.setFillColor(TOYOTA_RED)
    canvas.rect(0, H - 18 * mm, W, 18 * mm, fill=1, stroke=0)
    # Header text
    canvas.setFillColor(colors.white)
    canvas.setFont("Courier-Bold", 11)
    canvas.drawString(15 * mm, H - 12 * mm, "GT3 PERFORMANCE ANALYSIS REPORT")
    canvas.setFont("Courier", 8)
    canvas.drawRightString(W - 15 * mm, H - 12 * mm,
                           f"#88 | Brands Hatch GP | {date.today().strftime('%d %b %Y')}")
    # Footer
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.rect(0, 0, W, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.setFont("Courier", 6)
    canvas.drawString(15 * mm, 2.5 * mm,
                      "CONFIDENTIAL — Toyota Racing GmbH Customer Programme")
    canvas.drawRightString(W - 15 * mm, 2.5 * mm,
                           f"Page {doc.page}  |  Generated by GT3 Telemetry Analysis Toolkit")
    canvas.restoreState()


# ── paragraph styles ──────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    s = {}
    common = dict(fontName="Courier", textColor=TEXT_PRIMARY,
                  backColor=DARK_BG)
    s["h2"] = ParagraphStyle("h2", fontSize=10, leading=14,
                              fontName="Courier-Bold",
                              textColor=TOYOTA_RED, backColor=DARK_BG,
                              spaceAfter=3)
    s["h3"] = ParagraphStyle("h3", fontSize=8, leading=12,
                              fontName="Courier-Bold",
                              textColor=TEXT_SEC, backColor=DARK_BG,
                              spaceAfter=2)
    s["body"] = ParagraphStyle("body", fontSize=7.5, leading=11,
                                fontName="Courier",
                                textColor=colors.HexColor("#CCCCCC"),
                                backColor=DARK_BG)
    s["rec"] = ParagraphStyle("rec", fontSize=8, leading=12,
                               fontName="Courier",
                               textColor=colors.HexColor("#DDDDDD"),
                               backColor=colors.HexColor("#1A1A1A"),
                               leftIndent=8, rightIndent=8,
                               spaceBefore=4, spaceAfter=4)
    return s


# ── main report builder ────────────────────────────────────────────────────────
def generate_report(output_path: str = "output/session_report.pdf") -> None:
    print("[Report] Loading data...")
    df, ls = load_data()

    best_lap  = int(df.groupby("LapNumber")["LapTime"].first().idxmin())
    best_time = df.groupby("LapNumber")["LapTime"].first().min()
    n_laps    = int(ls["LapNumber"].max())
    deg_slope = 0.150   # from Phase 2

    from scipy.stats import linregress
    ages = ls["TyreAge"].values
    times= ls["LapTime"].fillna(95.0).values
    slope, intercept, r2, _, _ = linregress(ages, times)
    cross = next((a for a in range(30)
                  if slope*a + intercept - times[0] > 0.5), None)

    S = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=22*mm, bottomMargin=12*mm,
    )

    story = []

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    # Session overview block
    story.append(Paragraph("SESSION OVERVIEW", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))

    overview_data = [
        ["Circuit", "Brands Hatch GP", "Session", "Free Practice 1"],
        ["Car",     "#88 / GT3",       "Compound", "Slick (dry)"],
        ["Laps",    str(n_laps),        "Best lap", f"{best_time:.3f} s (Lap {best_lap})"],
        ["Driver",  "—",               "Analyst",  "U. Chaudhari"],
    ]
    ov_table = Table(overview_data, colWidths=[30*mm, 60*mm, 30*mm, 60*mm])
    ov_table.setStyle(_ts([
        ("FONTNAME",   (0,0),(-1,-1), "Courier"),
        ("FONTSIZE",   (0,0),(-1,-1), 7.5),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#888888")),
        ("TEXTCOLOR",  (2,0), (2,-1), colors.HexColor("#888888")),
        ("TEXTCOLOR",  (1,0), (1,-1), colors.HexColor("#DDDDDD")),
        ("TEXTCOLOR",  (3,0), (3,-1), colors.HexColor("#DDDDDD")),
        ("FONTNAME",   (1,0), (1,-1), "Courier-Bold"),
        ("FONTNAME",   (3,0), (3,-1), "Courier-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.2, colors.HexColor("#2A2A2A")),
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#1A1A1A")),
        ("LEFTPADDING",(0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 5*mm))

    # Lap time table
    story.append(Paragraph("LAP TIME SUMMARY", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))
    lt_data = lap_table_data(df, ls)
    col_w   = [15*mm, 22*mm, 22*mm, 20*mm, 24*mm, 22*mm, 22*mm]
    lt_table = Table(lt_data, colWidths=col_w)
    lt_table.setStyle(TableStyle([
        *HEADER_STYLE._cmds,
        *ROW_STYLE._cmds,
        # Highlight best lap row
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#1C2E1C")),
        ("TEXTCOLOR",  (0, 1), (-1, 1), colors.HexColor("#52B788")),
    ]))
    story.append(lt_table)
    story.append(Spacer(1, 5*mm))

    # Speed trace + G-G side by side
    story.append(Paragraph("SPEED TRACE & G-G DIAGRAM", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))
    print("[Report] Building speed trace...")
    spd_img = fig_to_image(build_speed_trace(df), 115, 40)
    print("[Report] Building G-G...")
    gg_img  = fig_to_image(build_gg_best_lap(df), 55, 55)
    combo   = Table([[spd_img, gg_img]],
                    colWidths=[118*mm, 58*mm])
    combo.setStyle(_ts([("VALIGN",(0,0),(-1,-1),"TOP"),
                        ("LEFTPADDING",(0,0),(-1,-1),0),
                        ("RIGHTPADDING",(0,0),(-1,-1),2*mm)]))
    story.append(combo)
    story.append(Spacer(1, 4*mm))

    # Tyre heatmap
    story.append(Paragraph("TYRE TEMPERATURE HEATMAP", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))
    print("[Report] Building tyre heatmap...")
    story.append(fig_to_image(build_tyre_heatmap(ls), 175, 28))

    story.append(PageBreak())

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("ENGINEERING ANALYSIS", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))

    # Degradation + oversteer side by side
    print("[Report] Building degradation chart...")
    deg_img = fig_to_image(build_degradation_chart(ls), 115, 42)
    print("[Report] Building oversteer trace...")
    os_img  = fig_to_image(build_oversteer_trace(df), 115, 36)

    story.append(deg_img)
    story.append(Spacer(1, 3*mm))
    story.append(os_img)
    story.append(Spacer(1, 5*mm))

    # Setup snapshot
    story.append(Paragraph("SETUP SNAPSHOT", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#333333"), spaceAfter=4))
    su_data = setup_table_data()
    su_table = Table(su_data, colWidths=[45*mm, 55*mm, 75*mm])
    su_table.setStyle(TableStyle([
        *HEADER_STYLE._cmds,
        *ROW_STYLE._cmds,
    ]))
    story.append(su_table)
    story.append(Spacer(1, 5*mm))

    # Engineer's recommendation box
    story.append(Paragraph("ENGINEER RECOMMENDATIONS", S["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E60012"), spaceAfter=4))

    tyre_msg = ("Tyres consistently running above operating window from lap 2 onwards "
                "(mean FR mid temp > 125 °C vs 110 °C ceiling). Recommend reducing "
                "tyre blanket temperature by 5 °C and exploring lower camber on front-right.")
    deg_msg  = (f"Degradation rate: {slope*1000:.0f} ms/lap (R²={r2**2:.2f}). "
                f"Crossover lap: {cross}. With a 10-lap stint, expect +{slope*9*1000:.0f} ms "
                f"on final lap vs first. Pit strategy: target Lap {cross}-{cross+1 if cross else '—'} stop window.")
    bal_msg  = ("Oversteer metric shows balanced behaviour on best lap with slight "
                "understeer in medium-speed corners (turns 3–5 proxy). Front ARB at 35 N·m/deg. "
                "ARB sweep analysis suggests softening to ~28 N·m/deg to shift K_us toward "
                "neutral — estimated lap time gain: +20–35 ms.")

    for bullet, text in [("TYRES", tyre_msg), ("DEG", deg_msg), ("BALANCE", bal_msg)]:
        row_data = [[
            Paragraph(f"<b>{bullet}</b>",
                      ParagraphStyle("bul", fontName="Courier-Bold",
                                     fontSize=7.5, textColor=TOYOTA_RED,
                                     backColor=colors.HexColor("#1A1A1A"))),
            Paragraph(text,
                      ParagraphStyle("btxt", fontName="Courier",
                                     fontSize=7.5, leading=11,
                                     textColor=colors.HexColor("#CCCCCC"),
                                     backColor=colors.HexColor("#1A1A1A")))
        ]]
        rt = Table(row_data, colWidths=[18*mm, 157*mm])
        rt.setStyle(_ts([
            ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#1A1A1A")),
            ("GRID",       (0,0),(-1,-1), 0.3, colors.HexColor("#333333")),
            ("VALIGN",     (0,0),(-1,-1), "TOP"),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
            ("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING", (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 1.5*mm))

    print("[Report] Building PDF...")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"[Report] ✓ Saved → {output_path}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    generate_report()