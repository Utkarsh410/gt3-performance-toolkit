"""
telemetry_pipeline.py
---------------------
Phase 1 core module — GT3 telemetry ingestion and Motec i2 maths channel replication.

What this does:
  1. Loads raw CSV telemetry (ACC / Motec export format)
  2. Validates and cleans channels
  3. Resamples to a unified time base (100 Hz by default)
  4. Computes derived channels that Motec i2 calculates internally:
       - Lateral g (from steering + speed, Milliken formula)
       - Oversteer metric (yaw rate vs lateral acc)
       - Steering angle rate (deg/s)
       - Balance index (front vs rear lateral load transfer)
       - Traction circle utilisation (combined g as fraction of grip limit)
  5. Exports a clean processed CSV ready for analysis modules

Usage:
  from telemetry_pipeline import TelemetryPipeline

  pipe = TelemetryPipeline("data/sample_session.csv")
  session = pipe.process()          # returns processed DataFrame
  pipe.export("output/processed.csv")
  pipe.summary()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── vehicle constants (GR86 GT3 / Lamborghini Huracán GT3 proxy) ──────────────
@dataclass
class VehicleParams:
    """Physical constants for the GT3 car under analysis."""
    wheelbase_m:      float = 2.620    # metres (Huracán GT3)
    track_front_m:    float = 1.640    # metres
    track_rear_m:     float = 1.620    # metres
    mass_kg:          float = 1350.0   # kg inc. driver (GT3 Balance of Performance)
    cg_height_m:      float = 0.440    # metres
    front_weight_dist: float = 0.415   # fraction of mass on front axle
    grip_limit_g:     float = 2.8      # maximum combined lateral+longitudinal g
    sample_rate_hz:   float = 100.0

    @property
    def rear_weight_dist(self) -> float:
        return 1.0 - self.front_weight_dist


# ── channel name normalisation map ───────────────────────────────────────────
CHANNEL_ALIASES = {
    # Common Motec i2 / ACC / Race Studio naming variants
    "speed_kmh":    ["Speed", "speed", "SPEED", "vCar", "Vehicle Speed"],
    "rpm":          ["RPM", "rpm", "EngineRPM", "Engine RPM"],
    "gear":         ["Gear", "gear", "GearPosition"],
    "throttle":     ["ThrottlePos", "Throttle", "tApp", "ThrottleApplication"],
    "brake":        ["BrakePress", "Brake", "bPres", "BrakePresFront"],
    "steering":     ["SteeringAngle", "Steering", "steerAngle", "SteeringWheelAngle"],
    "lat_acc":      ["LateralAcc", "AccelLat", "LatG", "Lateral Acceleration"],
    "long_acc":     ["LongAcc", "AccelLon", "LonG", "Longitudinal Acceleration"],
    "vert_acc":     ["VertAcc", "AccelVert", "VertG"],
    "lap_number":   ["LapNumber", "Lap", "LapNum"],
    "lap_time":     ["LapTime", "laptime"],
    "time":         ["Time", "time", "TIME", "Timestamp"],
    # Tyre temps
    "tfl_i": ["TyreTempFL_inner", "TyreTempFLI", "WheelTempFL_I"],
    "tfl_m": ["TyreTempFL_mid",   "TyreTempFLM", "WheelTempFL_M"],
    "tfl_o": ["TyreTempFL_outer", "TyreTempFLO", "WheelTempFL_O"],
    "tfr_i": ["TyreTempFR_inner", "TyreTempFRI", "WheelTempFR_I"],
    "tfr_m": ["TyreTempFR_mid",   "TyreTempFRM", "WheelTempFR_M"],
    "tfr_o": ["TyreTempFR_outer", "TyreTempFRO", "WheelTempFR_O"],
    "trl_i": ["TyreTempRL_inner", "TyreTempRLI", "WheelTempRL_I"],
    "trl_m": ["TyreTempRL_mid",   "TyreTempRLM", "WheelTempRL_M"],
    "trl_o": ["TyreTempRL_outer", "TyreTempRLO", "WheelTempRL_O"],
    "trr_i": ["TyreTempRR_inner", "TyreTempRRI", "WheelTempRR_I"],
    "trr_m": ["TyreTempRR_mid",   "TyreTempRRM", "WheelTempRR_M"],
    "trr_o": ["TyreTempRR_outer", "TyreTempRRO", "WheelTempRR_O"],
    # Pressures
    "pfl": ["TyrePressFL", "TyrePressureFrontLeft"],
    "pfr": ["TyrePressFR", "TyrePressureFrontRight"],
    "prl": ["TyrePressRL", "TyrePressureRearLeft"],
    "prr": ["TyrePressRR", "TyrePressureRearRight"],
    # Suspension
    "sfl": ["SuspTravelFL", "SuspensionTravelFL"],
    "sfr": ["SuspTravelFR", "SuspensionTravelFR"],
    "srl": ["SuspTravelRL", "SuspensionTravelRL"],
    "srr": ["SuspTravelRR", "SuspensionTravelRR"],
}


def _resolve_channel(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the actual column name in df that matches the canonical channel."""
    candidates = CHANNEL_ALIASES.get(canonical, [canonical])
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _butter_lowpass(data: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (Motec i2 uses similar for smoothing)."""
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


class TelemetryPipeline:
    """
    Ingests raw GT3 telemetry, validates channels, and computes Motec i2
    maths channels from first principles.

    Parameters
    ----------
    filepath : path to raw CSV telemetry file
    params   : VehicleParams dataclass (defaults to GT3 proxy values)
    verbose  : print processing steps
    """

    def __init__(
        self,
        filepath: str | Path,
        params: VehicleParams | None = None,
        verbose: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.params   = params or VehicleParams()
        self.verbose  = verbose
        self._raw: pd.DataFrame | None = None
        self._processed: pd.DataFrame | None = None
        self._channel_map: dict[str, str] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def process(self) -> pd.DataFrame:
        """Full pipeline: load → validate → resample → compute channels → return."""
        self._load()
        self._map_channels()
        self._validate()
        self._resample()
        self._compute_maths_channels()
        self._log("Pipeline complete.")
        return self._processed

    def export(self, output_path: str | Path = "output/processed.csv") -> None:
        """Write processed DataFrame to CSV."""
        if self._processed is None:
            raise RuntimeError("Call process() before export().")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._processed.to_csv(out, index=False, float_format="%.5f")
        self._log(f"Exported → {out}  ({len(self._processed):,} rows, {len(self._processed.columns)} channels)")

    def export_motec_csv(self, output_path: str | Path = "output/motec_maths.csv") -> None:
        """
        Export ONLY the computed maths channels in a Motec i2 compatible CSV
        format so they can be imported as a 'Maths channel' overlay in i2.

        Format: Time, channel1, channel2, ...
        """
        if self._processed is None:
            raise RuntimeError("Call process() before export_motec_csv().")
        maths_cols = [
            "Time",
            "LatG_calc",
            "LonG_calc",
            "SteeringRate",
            "OversteerMetric",
            "TractionCircleUtil",
            "LateralLoadTransferFront",
            "LateralLoadTransferRear",
            "BalanceIndex",
            "YawRateEstimate",
        ]
        existing = [c for c in maths_cols if c in self._processed.columns]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._processed[existing].to_csv(out, index=False, float_format="%.5f")
        self._log(f"Motec maths CSV → {out}  ({len(existing)-1} channels)")

    def summary(self) -> None:
        """Print a concise session summary — mirrors Motec i2 Session Info pane."""
        if self._processed is None:
            raise RuntimeError("Call process() first.")
        df = self._processed
        print("\n" + "═" * 56)
        print("  GT3 TELEMETRY PIPELINE — SESSION SUMMARY")
        print("═" * 56)
        print(f"  File          : {self.filepath.name}")
        print(f"  Samples       : {len(df):,}")
        print(f"  Duration      : {df['Time'].iloc[-1]:.2f} s")
        print(f"  Sample rate   : {self.params.sample_rate_hz:.0f} Hz")
        print(f"  Channels      : {len(df.columns)}")

        if "LapNumber" in df.columns:
            laps = df.groupby("LapNumber")["LapTime"].first()
            print(f"\n  Laps: {len(laps)}")
            print(f"  Best lap      : {laps.min():.3f} s  (Lap {laps.idxmin()})")
            print(f"  Slowest lap   : {laps.max():.3f} s  (Lap {laps.idxmax()})")

        print(f"\n  Speed max     : {df['Speed'].max():.1f} km/h")
        print(f"  Lat G peak    : {df['LatG_calc'].abs().max():.3f} g")
        print(f"  Lon G peak    : {df['LonG_calc'].abs().max():.3f} g")
        print(f"  Traction util : {df['TractionCircleUtil'].mean():.1%} avg  |  {df['TractionCircleUtil'].max():.1%} peak")
        print(f"\n  Oversteer     : {(df['OversteerMetric'] > 0.05).mean():.1%} of session")
        print(f"  Understeer    : {(df['OversteerMetric'] < -0.05).mean():.1%} of session")
        print("═" * 56 + "\n")

    # ── internal steps ────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._log(f"Loading {self.filepath.name} ...")
        self._raw = pd.read_csv(self.filepath, low_memory=False)
        self._log(f"  Rows: {len(self._raw):,}  |  Columns: {len(self._raw.columns)}")

    def _map_channels(self) -> None:
        """Build canonical → actual column name map."""
        for canonical in CHANNEL_ALIASES:
            actual = _resolve_channel(self._raw, canonical)
            if actual:
                self._channel_map[canonical] = actual
        found = len(self._channel_map)
        total = len(CHANNEL_ALIASES)
        self._log(f"  Channels resolved: {found}/{total}")

    def _validate(self) -> None:
        """Drop rows with NaN in critical channels, clip obviously bad values."""
        df = self._raw.copy()

        # Drop fully empty columns
        df.dropna(axis=1, how="all", inplace=True)

        # Clip physical limits
        speed_col = self._channel_map.get("speed_kmh")
        if speed_col:
            df[speed_col] = df[speed_col].clip(0, 400)
        rpm_col = self._channel_map.get("rpm")
        if rpm_col:
            df[rpm_col] = df[rpm_col].clip(0, 12000)

        # Forward-fill small gaps (e.g. sensor dropout < 5 samples)
        df.ffill(limit=5, inplace=True)

        self._raw = df
        self._log(f"  Validation complete. Rows remaining: {len(df):,}")

    def _resample(self) -> None:
        """Ensure uniform 100 Hz time base using linear interpolation."""
        df    = self._raw.copy()
        t_col = self._channel_map.get("time", "Time")

        if t_col not in df.columns:
            df["Time"] = np.arange(len(df)) / self.params.sample_rate_hz
        else:
            df.rename(columns={t_col: "Time"}, inplace=True)

        # Build uniform time axis
        t_start = df["Time"].iloc[0]
        t_end   = df["Time"].iloc[-1]
        dt      = 1.0 / self.params.sample_rate_hz
        t_new   = np.arange(t_start, t_end, dt)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        resampled = {"Time": t_new}
        for col in numeric_cols:
            if col == "Time":
                continue
            resampled[col] = np.interp(t_new, df["Time"].values, df[col].values)

        self._processed = pd.DataFrame(resampled)
        self._log(f"  Resampled to {self.params.sample_rate_hz:.0f} Hz — {len(self._processed):,} samples")

    def _compute_maths_channels(self) -> None:
        """
        Replicate Motec i2 maths channels from first principles.
        Each calculation is documented with the formula used.
        """
        df  = self._processed
        p   = self.params
        dt  = 1.0 / p.sample_rate_hz
        fs  = p.sample_rate_hz

        # resolve columns (fall back to zeros if not found)
        def get(canonical: str) -> np.ndarray:
            actual = self._channel_map.get(canonical)
            if actual and actual in df.columns:
                return df[actual].values.astype(float)
            # try direct match after resample
            for alias in CHANNEL_ALIASES.get(canonical, []):
                if alias in df.columns:
                    return df[alias].values.astype(float)
            return np.zeros(len(df))

        speed_ms    = get("speed_kmh") / 3.6
        steering_deg = get("steering")
        lat_acc_raw  = get("lat_acc")
        long_acc_raw = get("long_acc")

        # ── 1. Steering angle rate (deg/s) ─────────────────────────────────
        # Motec formula: d(SteeringAngle)/dt filtered at 10 Hz
        steer_rate_raw = np.gradient(steering_deg, dt)
        df["SteeringRate"] = _butter_lowpass(steer_rate_raw, cutoff_hz=10.0, fs_hz=fs)

        # ── 2. Yaw rate estimate (rad/s) ───────────────────────────────────
        # From Motec: YawRate = (LateralAcc * g) / Speed  [small angle approx]
        # More accurate than using IMU yaw directly at low speed
        with np.errstate(divide="ignore", invalid="ignore"):
            yaw_rate = np.where(
                speed_ms > 5.0,
                (lat_acc_raw * 9.81) / speed_ms,
                0.0
            )
        df["YawRateEstimate"] = _butter_lowpass(yaw_rate, cutoff_hz=15.0, fs_hz=fs)

        # ── 3. Lateral G calculated (Milliken bicycle model) ───────────────
        # LatG_calc = V² × δ / (g × L)
        # where δ = steering angle / steering ratio, L = wheelbase
        # GT3 typical steering ratio: 14:1
        steer_ratio = 14.0
        steer_rad   = np.radians(steering_deg / steer_ratio)
        lat_g_calc_raw = (speed_ms ** 2 * steer_rad) / (9.81 * p.wheelbase_m)
        df["LatG_calc"] = np.clip(
            _butter_lowpass(lat_g_calc_raw, cutoff_hz=15.0, fs_hz=fs),
            -4.0, 4.0
        )

        # ── 4. Longitudinal G calculated ───────────────────────────────────
        # dV/dt from speed channel, filtered
        lon_g_raw = np.gradient(speed_ms, dt) / 9.81
        df["LonG_calc"] = _butter_lowpass(lon_g_raw, cutoff_hz=15.0, fs_hz=fs)

        # ── 5. Oversteer metric ────────────────────────────────────────────
        # Oversteer = YawRate_measured - YawRate_neutral
        # YawRate_neutral (Ackermann) = V / (L × (1 + K × V²))
        # K = understeer gradient coefficient = (m × (a_f/C_f - a_r/C_r)) / L²
        # Simplified: use sign and magnitude of (LatG_calc - LatG_measured)
        # +ve = oversteer (rear slides), -ve = understeer (front pushes)
        df["OversteerMetric"] = df["LatG_calc"] - lat_acc_raw
        df["OversteerMetric"] = _butter_lowpass(
            df["OversteerMetric"].values, cutoff_hz=5.0, fs_hz=fs
        )

        # ── 6. Lateral load transfer — front & rear (N) ───────────────────
        # ΔFLF = (m × ay × g × h_cg) / (2 × tf)   — front
        # ΔFLR = (m × ay × g × h_cg) / (2 × tr)   — rear
        # where ay is lateral acc in g units
        ay = lat_acc_raw  # g
        df["LateralLoadTransferFront"] = (
            p.mass_kg * ay * 9.81 * p.cg_height_m
        ) / (2 * p.track_front_m)
        df["LateralLoadTransferRear"] = (
            p.mass_kg * ay * 9.81 * p.cg_height_m
        ) / (2 * p.track_rear_m)

        # ── 7. Balance index ───────────────────────────────────────────────
        # BalanceIndex = ΔLLT_front / (ΔLLT_front + ΔLLT_rear)
        # 0 = all load on rear, 1 = all load on front
        # Neutral ~0.45–0.52 for typical GT3 front-weight dist
        total_llt = (
            np.abs(df["LateralLoadTransferFront"].values) +
            np.abs(df["LateralLoadTransferRear"].values)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            df["BalanceIndex"] = np.where(
                total_llt > 1.0,
                np.abs(df["LateralLoadTransferFront"].values) / total_llt,
                p.front_weight_dist  # fallback at near-zero lateral load
            )

        # ── 8. Traction circle utilisation ────────────────────────────────
        # Combined_G = sqrt(LatG² + LonG²)
        # Utilisation = Combined_G / grip_limit_g   (dimensionless, 0–1)
        # >1.0 = over-limit (tyre slip event)
        combined_g = np.sqrt(lat_acc_raw ** 2 + long_acc_raw ** 2)
        df["TractionCircleUtil"] = combined_g / p.grip_limit_g

        self._log(
            f"  Maths channels computed: SteeringRate, YawRateEstimate, "
            f"LatG_calc, LonG_calc, OversteerMetric, LateralLoadTransfer "
            f"(front/rear), BalanceIndex, TractionCircleUtil"
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Pipeline] {msg}")
