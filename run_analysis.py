"""
run_analysis.py
---------------
GT3 Performance Analysis Toolkit — Unified CLI Entry Point

One command runs the full pipeline end-to-end:

  python run_analysis.py --session data/sample_session.csv --compound SC

Options:
  --session       Path to raw telemetry CSV (ACC / Motec export)
                  Omit to auto-generate synthetic session data
  --compound      Tyre compound: SC (super control) | DC | DHD | DHD2
                  Adjusts operating window temperatures accordingly
  --laps          Comma-separated lap numbers to analyse  e.g. "1,3,5"
                  Default: all laps
  --best-vs       Lap number to use as reference. Default: fastest lap
  --output-dir    Output directory. Default: output/
  --report        Generate PDF session report (flag, default: True)
  --no-report     Skip PDF generation
  --phase         Run only a specific phase: 1 | 2 | 3 | all (default)
  --verbose       Print detailed pipeline logs

Example:
  python run_analysis.py
  python run_analysis.py --session data/my_race.csv --compound DHD2
  python run_analysis.py --session data/quali.csv --laps 3,7,12 --phase 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ── compound → tyre window map ────────────────────────────────────────────────
COMPOUND_WINDOWS = {
    "SC": {   # Super Control — harder, cooler window
        "FL": {"inner": (85, 110), "mid": (80, 105), "outer": (75,  100)},
        "FR": {"inner": (85, 110), "mid": (80, 105), "outer": (75,  100)},
        "RL": {"inner": (83, 108), "mid": (78, 103), "outer": (73,   98)},
        "RR": {"inner": (83, 108), "mid": (78, 103), "outer": (73,   98)},
    },
    "DC": {   # Double Control
        "FL": {"inner": (88, 113), "mid": (83, 108), "outer": (78, 103)},
        "FR": {"inner": (88, 113), "mid": (83, 108), "outer": (78, 103)},
        "RL": {"inner": (86, 111), "mid": (81, 106), "outer": (76, 101)},
        "RR": {"inner": (86, 111), "mid": (81, 106), "outer": (76, 101)},
    },
    "DHD": {  # Double Hard (endurance)
        "FL": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
        "FR": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
        "RL": {"inner": (88, 112), "mid": (83, 108), "outer": (78, 103)},
        "RR": {"inner": (88, 112), "mid": (83, 108), "outer": (78, 103)},
    },
    "DHD2": { # Double Hard 2 — hottest window (high-load circuits)
        "FL": {"inner": (92, 118), "mid": (87, 113), "outer": (82, 108)},
        "FR": {"inner": (92, 118), "mid": (87, 113), "outer": (82, 108)},
        "RL": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
        "RR": {"inner": (90, 115), "mid": (85, 110), "outer": (80, 105)},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GT3 Performance Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--session",    type=str, default=None,
                   help="Path to raw telemetry CSV. Omit to generate synthetic data.")
    p.add_argument("--compound",   type=str, default="DHD",
                   choices=["SC", "DC", "DHD", "DHD2"],
                   help="Tyre compound (sets operating window temperatures)")
    p.add_argument("--laps",       type=str, default=None,
                   help="Comma-separated lap numbers e.g. '1,3,5'. Default: all")
    p.add_argument("--best-vs",    type=int, default=None,
                   help="Reference lap number. Default: fastest lap")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Output directory. Default: output/")
    p.add_argument("--phase",      type=str, default="all",
                   choices=["1", "2", "3", "all"],
                   help="Run specific phase or all. Default: all")
    p.add_argument("--no-report",  action="store_true",
                   help="Skip PDF session report generation")
    p.add_argument("--verbose",    action="store_true",
                   help="Print detailed pipeline logs")
    return p.parse_args()


def banner():
    print("\n" + "═" * 62)
    print("  GT3 PERFORMANCE ANALYSIS TOOLKIT")
    print("  Toyota Racing GmbH Portfolio Project — U. Chaudhari")
    print("  IIT Kharagpur  ·  MTS Monza Race Engineering Certificate")
    print("═" * 62)


def run_phase1(session_csv: str, output_dir: str, verbose: bool) -> str:
    """Ingest → validate → resample → Motec maths channels → plots."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    from generate_sample_data import generate_session
    from telemetry_pipeline import TelemetryPipeline
    import matplotlib
    matplotlib.use("Agg")

    print("\n[Phase 1] Telemetry pipeline & Motec maths channels")
    print("─" * 50)

    if session_csv is None or not Path(session_csv).exists():
        if session_csv is None:
            print("  No session file specified — generating synthetic GT3 data...")
        else:
            print(f"  File not found: {session_csv} — generating synthetic data...")
        os.makedirs("data", exist_ok=True)
        session_csv = "data/sample_session.csv"
        generate_session(session_csv)

    pipe = TelemetryPipeline(session_csv, verbose=verbose)
    df   = pipe.process()
    processed_path = f"{output_dir}/processed.csv"
    pipe.export(processed_path)
    pipe.export_motec_csv(f"{output_dir}/motec_maths.csv")
    pipe.summary()

    # Phase 1 plots
    from phase1_analysis import (
        plot_lap_trace, plot_maths_validation,
        plot_traction_circle, plot_load_transfer
    )
    plots = [
        (plot_lap_trace(df),           f"{output_dir}/01_lap_trace.png"),
        (plot_maths_validation(df),    f"{output_dir}/02_maths_validation.png"),
        (plot_traction_circle(df),     f"{output_dir}/03_traction_circle.png"),
        (plot_load_transfer(df),       f"{output_dir}/04_load_transfer.png"),
    ]
    for fig, path in plots:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {path}")

    return processed_path


def run_phase2(processed_csv: str, compound: str,
               output_dir: str, verbose: bool) -> None:
    """Tyre temperature classification + degradation model."""
    import matplotlib.pyplot as plt
    from tyre_temperature_analysis import (
        TyreTemperatureAnalyser, TYRE_WINDOWS,
        plot_tyre_temp_trace, plot_window_utilisation,
        plot_degradation, plot_temp_heatmap
    )

    print("\n[Phase 2] Tyre temperature analysis & degradation model")
    print("─" * 50)

    windows = COMPOUND_WINDOWS.get(compound, TYRE_WINDOWS)
    print(f"  Compound : {compound}")
    print(f"  Window   : mid {windows['FL']['mid'][0]}–{windows['FL']['mid'][1]} °C (FL)")

    analyser = (
        TyreTemperatureAnalyser(processed_csv, windows=windows)
        .load().classify().lap_summary_stats()
    )

    anomalies = analyser.detect_anomalies(delta_threshold_C=6.0)
    deg_results = analyser.fit_degradation()

    plots = [
        (plot_tyre_temp_trace(analyser),            f"{output_dir}/05_temp_trace.png"),
        (plot_window_utilisation(analyser),          f"{output_dir}/06_window_utilisation.png"),
        (plot_degradation(analyser, deg_results),    f"{output_dir}/07_degradation.png"),
        (plot_temp_heatmap(analyser),                f"{output_dir}/08_temp_heatmap.png"),
    ]
    for fig, path in plots:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {path}")


def run_phase3(processed_csv: str, output_dir: str,
               generate_report: bool, verbose: bool) -> None:
    """G-G analysis + ARB context + PDF session report."""
    import matplotlib.pyplot as plt
    from gg_diagram import (
        _load_session, plot_gg_density,
        plot_gg_sectors, plot_gg_consistency
    )

    print("\n[Phase 3] G-G analysis & session report")
    print("─" * 50)

    df = _load_session(processed_csv)

    plots = [
        (plot_gg_density(df),        f"{output_dir}/10_gg_density.png"),
        (plot_gg_sectors(df),        f"{output_dir}/11_gg_sectors.png"),
        (plot_gg_consistency(df),    f"{output_dir}/12_gg_consistency.png"),
    ]
    for fig, path in plots:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {path}")

    if generate_report:
        from session_report import generate_report as gen_pdf
        report_path = f"{output_dir}/session_report.pdf"
        gen_pdf(report_path)


def print_summary(output_dir: str, t_elapsed: float, generate_report: bool) -> None:
    print("\n" + "═" * 62)
    print("  ANALYSIS COMPLETE")
    print(f"  Elapsed: {t_elapsed:.1f} s")
    print("═" * 62)
    print(f"\n  Output directory: {output_dir}/\n")

    files = {
        "Processed telemetry": "processed.csv",
        "Motec maths channels (i2 import)": "motec_maths.csv",
        "Lap trace overlay": "01_lap_trace.png",
        "Maths channel validation": "02_maths_validation.png",
        "Traction circle": "03_traction_circle.png",
        "Tyre temp trace": "05_temp_trace.png",
        "Window utilisation": "06_window_utilisation.png",
        "Degradation model": "07_degradation.png",
        "Tyre heatmap": "08_temp_heatmap.png",
        "G-G density (KDE)": "10_gg_density.png",
        "G-G sector analysis": "11_gg_sectors.png",
        "G-G lap consistency": "12_gg_consistency.png",
    }
    if generate_report:
        files["Session report (PDF)"] = "session_report.pdf"

    for label, fname in files.items():
        path = Path(output_dir) / fname
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists}  {label:<38}  {fname}")

    print(f"\n  MATLAB files (run separately):")
    print(f"     pacejka_magic_formula.m    — MF96 Fy/Fx curves")
    print(f"     arb_sensitivity_sweep.m    — ARB balance map\n")


def main():
    args = parse_args()
    banner()

    t0 = time.time()
    out = args.output_dir
    Path(out).mkdir(parents=True, exist_ok=True)

    generate_report = not args.no_report

    try:
        if args.phase in ("1", "all"):
            processed = run_phase1(args.session, out, args.verbose)
        else:
            processed = f"{out}/processed.csv"
            if not Path(processed).exists():
                print(f"[Error] {processed} not found. Run Phase 1 first.")
                sys.exit(1)

        if args.phase in ("2", "all"):
            run_phase2(processed, args.compound, out, args.verbose)

        if args.phase in ("3", "all"):
            run_phase3(processed, out, generate_report, args.verbose)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)

    print_summary(out, time.time() - t0, generate_report)


if __name__ == "__main__":
    main()
