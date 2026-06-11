"""
generate_sample_data.py
-----------------------
Generates synthetic GT3 telemetry data that mimics ACC / Motec CSV export format.
Used for development and testing when you don't have a live session file.

Channels generated (100 Hz):
  Time, LapNumber, Speed, RPM, Gear, ThrottlePos, BrakePress,
  SteeringAngle, LateralAcc, LongAcc, VertAcc,
  TyreTempFL_inner, TyreTempFL_mid, TyreTempFL_outer (x4 corners),
  TyrePressFL (x4), SuspTravelFL (x4), LapTime, Sector
"""

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 100       # Motec default
LAP_COUNT       = 10
LAP_DURATION_S  = 95.0      # ~1:35 base lap (GT3, mid-length circuit)
TRACK_LEN_M     = 4200      # e.g. Brands Hatch GP

np.random.seed(42)


def make_lap(lap_num: int, dt: float = 1 / SAMPLE_RATE_HZ) -> pd.DataFrame:
    """Build one lap of synthetic telemetry."""
    # Lap time degrades by ~0.15 s/lap (tyre deg) with small noise
    lap_time = LAP_DURATION_S + lap_num * 0.15 + np.random.normal(0, 0.05)
    n = int(lap_time * SAMPLE_RATE_HZ)
    t = np.linspace(0, lap_time, n)

    # ── speed profile: sinusoidal "circuit" with braking zones ─────────────
    base_speed = 120 + 80 * np.sin(2 * np.pi * t / lap_time)          # km/h
    # Four braking zones per lap
    for frac in [0.2, 0.42, 0.62, 0.85]:
        centre = int(frac * n)
        width  = int(0.04 * n)
        idx    = np.arange(max(0, centre - width), min(n, centre + width))
        base_speed[idx] -= 80 * np.exp(-0.5 * ((idx - centre) / (width / 2)) ** 2)
    speed_kmh = np.clip(base_speed + np.random.normal(0, 1.5, n), 30, 280)

    # ── derived channels ───────────────────────────────────────────────────
    speed_ms      = speed_kmh / 3.6
    acceleration  = np.gradient(speed_ms, dt)                          # m/s²
    long_acc_g    = acceleration / 9.81                                # g
    steering_rad  = 0.15 * np.sin(4 * np.pi * t / lap_time) + np.random.normal(0, 0.01, n)
    lat_acc_g     = (speed_ms ** 2 * steering_rad) / (9.81 * 2.6)     # 2.6 m wheelbase approx
    lat_acc_g     = np.clip(lat_acc_g, -3.0, 3.0)

    throttle      = np.clip(0.7 + 0.5 * np.sin(2 * np.pi * t / lap_time) + np.random.normal(0, 0.05, n), 0, 1)
    brake         = np.where(long_acc_g < -0.3, np.clip(-long_acc_g * 30, 0, 100), 0)
    throttle      = np.where(brake > 5, 0, throttle)                   # no overlap
    rpm           = 3000 + speed_kmh * 28 + np.random.normal(0, 150, n)
    rpm           = np.clip(rpm, 1000, 8500)
    gear          = np.clip((speed_kmh / 40).astype(int) + 1, 1, 6)

    # ── tyre temperatures (4 corners, inner / mid / outer) ─────────────────
    # GT3 operating window: ~80–110 °C (slick compound)
    # Temps build over the stint, right-side runs hotter on clockwise circuit
    stint_heat    = 10 * (lap_num / LAP_COUNT)                         # heat buildup
    base_temp     = 75 + stint_heat + np.cumsum(np.abs(lat_acc_g) * dt * 0.5)
    base_temp     = base_temp - base_temp[0]                           # normalise start
    base_temp     = 75 + stint_heat + base_temp * 0.8

    def corner_temp(side_bias: float, pos_bias: float) -> tuple:
        """side_bias: +ve = right (works harder CW), pos_bias: lateral load spread."""
        inner = base_temp + side_bias + pos_bias + np.random.normal(0, 1.5, n)
        mid   = inner - 5  + np.random.normal(0, 1, n)
        outer = inner - 10 + np.random.normal(0, 1.2, n)
        return inner, mid, outer

    fl_i, fl_m, fl_o = corner_temp(-5,  5)   # FL: less loaded CW
    fr_i, fr_m, fr_o = corner_temp(+8,  3)   # FR: most loaded CW
    rl_i, rl_m, rl_o = corner_temp(-3,  2)
    rr_i, rr_m, rr_o = corner_temp(+6,  1)

    # ── tyre pressures (cold ~25.5 psi, heat soaks to ~29) ─────────────────
    pres_base = 25.5 + stint_heat * 0.3
    tyre_pres = {
        "TyrePressFL": pres_base + 0.3 * np.sin(t) + np.random.normal(0, 0.05, n),
        "TyrePressFR": pres_base + 0.5 * np.sin(t) + np.random.normal(0, 0.05, n),
        "TyrePressRL": pres_base + 0.2 * np.sin(t) + np.random.normal(0, 0.05, n),
        "TyrePressRR": pres_base + 0.4 * np.sin(t) + np.random.normal(0, 0.05, n),
    }

    # ── suspension travel (mm) ──────────────────────────────────────────────
    susp = {
        "SuspTravelFL": 45 + 15 * lat_acc_g + np.random.normal(0, 1, n),
        "SuspTravelFR": 45 - 15 * lat_acc_g + np.random.normal(0, 1, n),
        "SuspTravelRL": 42 + 12 * lat_acc_g + np.random.normal(0, 1, n),
        "SuspTravelRR": 42 - 12 * lat_acc_g + np.random.normal(0, 1, n),
    }

    # ── vertical acc (road noise) ───────────────────────────────────────────
    vert_acc = np.random.normal(1.0, 0.12, n)

    # ── assemble ────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Time":           t,
        "LapNumber":      lap_num,
        "Speed":          speed_kmh,
        "RPM":            rpm,
        "Gear":           gear,
        "ThrottlePos":    throttle * 100,   # 0–100 %
        "BrakePress":     brake,            # 0–100 bar
        "SteeringAngle":  np.degrees(steering_rad),  # degrees
        "LateralAcc":     lat_acc_g,
        "LongAcc":        long_acc_g,
        "VertAcc":        vert_acc,
        "TyreTempFL_inner": fl_i, "TyreTempFL_mid": fl_m, "TyreTempFL_outer": fl_o,
        "TyreTempFR_inner": fr_i, "TyreTempFR_mid": fr_m, "TyreTempFR_outer": fr_o,
        "TyreTempRL_inner": rl_i, "TyreTempRL_mid": rl_m, "TyreTempRL_outer": rl_o,
        "TyreTempRR_inner": rr_i, "TyreTempRR_mid": rr_m, "TyreTempRR_outer": rr_o,
        **tyre_pres,
        **susp,
        "LapTime":        lap_time,
    })

    return df


def generate_session(output_path: str = "data/sample_session.csv") -> pd.DataFrame:
    laps = []
    cumulative_time = 0.0
    for lap in range(1, LAP_COUNT + 1):
        df = make_lap(lap)
        df["Time"] += cumulative_time
        cumulative_time = df["Time"].iloc[-1]
        laps.append(df)
        print(f"  Lap {lap:2d} generated — {df['LapTime'].iloc[0]:.3f} s  ({len(df):,} samples)")

    session = pd.concat(laps, ignore_index=True)
    session.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nSession saved → {output_path}")
    print(f"Total samples : {len(session):,}")
    print(f"Duration      : {session['Time'].iloc[-1]:.1f} s")
    return session


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_session("data/sample_session.csv")
