#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================
#  PATENT NOTICE — DEMO / RESEARCH USE ONLY
# ============================================================
#  The OrganOS architecture implemented in this file includes
#  concepts covered by patent priority applications filed with
#  KIPO (Korean Intellectual Property Office):
#
#    - Failure-Density based Risk-Dose accumulation (BangAyu)
#    - Fail-Closed execution kernel (Sidecar)
#    - Bio-inspired collapse propagation modeling (CollapseOS)
#    - Adaptive state-machine intervention (AnomalyDetector)
#
#  This code is released for RESEARCH, EDUCATION, and
#  REPRODUCIBILITY purposes only.
#
#  Commercial use, derivative products, or deployment of the
#  patented architecture requires explicit written permission.
#
#  Contact : MinGi Kim — linkedin.com/in/MinGiKim
#  Patents : Filed with KIPO, 2025-2026
# ============================================================

"""
data_loader.py
--------------
Loads sensor data for the OrganOS Robot PdM simulation.

Two modes:
  1. NASA CSV mode  -- loads actual FEMTO / IMS bearing dataset files
  2. Synthetic mode -- generates data using published NASA parameter profiles
                       (used when dataset files are not available)

NASA FEMTO Dataset structure (after download):
  FEMTO/
    Bearing1_1/
      acc_00001.csv  (col0=hour, col1=min, col2=sec, col3=microsec,
                      col4=horiz_accel_g, col5=vert_accel_g)
      ...
    Bearing1_2/ ...

IMS Dataset structure:
  IMS/
    1st_test/  (4-column CSV: bearing1~4 vibration, 20480 samples/file)
    2nd_test/
    3rd_test/

Usage:
  # Synthetic (always works)
  df = load_data(mode="synthetic", profile="femto_outer", T=500, seed=42)

  # NASA FEMTO (requires downloaded files)
  df = load_data(mode="nasa_femto", data_dir="./data/FEMTO/Bearing1_1")

  # IMS (requires downloaded files)
  df = load_data(mode="nasa_ims", data_dir="./data/IMS/1st_test", bearing_col=0)
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional

from nasa_params import get_profile, BearingProfile


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic generator (NASA-parameter-based)
# ─────────────────────────────────────────────────────────────────────────────

def _logistic_deg(t: np.ndarray, onset: int, rate: float) -> np.ndarray:
    """S-shaped degradation curve matching FEMTO Bearing_1 mean profile."""
    x = t - onset
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-rate * x)), 0.0)


def generate_synthetic(
    profile_name: str = "femto_outer",
    T: int = 500,
    fault_onset: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic sensor time series using NASA-published parameter values.

    Parameters
    ----------
    profile_name : str
        One of "femto_outer", "femto_inner", "ims_wrist"
    T : int
        Number of time steps (1 step = 30 min field operation)
    fault_onset : int, optional
        Step at which degradation begins. Defaults to T//2.
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: t, M (vibration), S (position error), B (current dev), E (env stress), deg (true degradation)

    Notes
    -----
    Parameter values from:
      FEMTO: Nectoux et al. 2012, Table I (vibration RMS statistics)
      IMS:   Qiu et al. 2006, Table II (bearing vibration characteristics)
    """
    p: BearingProfile = get_profile(profile_name)
    rng = np.random.default_rng(seed)

    if fault_onset is None:
        fault_onset = T // 2

    t = np.arange(T)
    deg = _logistic_deg(t, fault_onset, p.deg_rate)

    # ── M: Vibration (g) — FEMTO horizontal channel characteristics ──
    # Normal: Gaussian noise around measured mean
    # Fault:  exponential rise to fault_max (FEMTO Bearing_1 profile)
    M_normal = p.vib_normal_mean + p.vib_normal_std * rng.standard_normal(T)
    M_fault  = (p.vib_fault_max - p.vib_normal_mean) * deg
    M_noise  = p.vib_noise_std * rng.standard_normal(T)
    M = np.clip(M_normal + M_fault + M_noise, 0.001, p.vib_fault_max * 1.1)

    # ── S: Position error (mm) — IMS encoder drift characteristics ──
    S_normal = p.pos_normal_mean + p.pos_normal_std * rng.standard_normal(T)
    S_fault  = (p.pos_fault_max - p.pos_normal_mean) * deg
    S = np.clip(S_normal + S_fault + 0.005 * rng.standard_normal(T), 0.0, p.pos_fault_max * 1.1)

    # ── B: Current deviation (A) — IMS servo analogue ──
    B_normal = p.curr_normal_mean + p.curr_normal_std * rng.standard_normal(T)
    B_fault  = (p.curr_fault_max - p.curr_normal_mean) * deg
    # Current spikes: occur with higher probability during degradation
    spike_p = np.where(t > fault_onset - 30, p.curr_spike_prob, p.curr_spike_prob * 0.1)
    spikes  = np.where(rng.random(T) < spike_p,
                       rng.uniform(0.1, p.curr_spike_mag, T), 0.0)
    B = np.clip(B_normal + B_fault + spikes, 0.0, p.curr_fault_max * 1.2)

    # ── E: Environmental stress (workload variation) ──
    E = 0.15 + 0.10 * np.sin(2 * np.pi * t / 50) + 0.04 * rng.standard_normal(T)
    E = np.clip(E, 0.0, 1.0)

    return pd.DataFrame({
        "t":       t,
        "M":       M,
        "S":       S,
        "B":       B,
        "E":       E,
        "deg":     deg,
        "source":  f"synthetic:{profile_name}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# NASA FEMTO loader (requires downloaded files)
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_femto(
    data_dir: str,
    step_minutes: float = 0.1,   # FEMTO: 0.1s windows in accelerated test
    field_scale: float = 300.0,  # ISO 281 L10 life derating to field robot
    resample_steps: int = 500,
) -> pd.DataFrame:
    """
    Load NASA FEMTO Bearing Dataset CSV files.

    Each acc_XXXXX.csv has columns:
      [hour, min, sec, microsec, horiz_accel_g, vert_accel_g]

    Parameters
    ----------
    data_dir : str
        Path to bearing folder, e.g. "./data/FEMTO/Bearing1_1"
    step_minutes : float
        Duration of each file window in minutes (FEMTO: 0.1s = 0.00167min)
    field_scale : float
        Life scaling factor from accelerated test to field robot
    resample_steps : int
        Resample to this many steps for simulation

    Returns
    -------
    pd.DataFrame with columns: t, M, S, B, E, deg, source
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "acc_*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No acc_*.csv files found in {data_dir}
"
            f"Download NASA FEMTO dataset from:
"
            f"  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/"
            f"prognostic-data-repository/
"
            f"Then place Bearing1_1/ etc. under ./data/FEMTO/"
        )

    rms_series = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, header=None,
                             names=["hour","min","sec","usec","horiz","vert"])
            rms = float(np.sqrt(np.mean(df["horiz"].values ** 2)))
            rms_series.append(rms)
        except Exception:
            continue

    if not rms_series:
        raise ValueError(f"Could not parse any CSV files in {data_dir}")

    rms_arr = np.array(rms_series)
    T_raw   = len(rms_arr)

    # Normalize to [0,1] degradation proxy
    rms_min = np.percentile(rms_arr[:max(1, T_raw // 10)], 50)
    rms_max = rms_arr.max()
    deg_raw = np.clip((rms_arr - rms_min) / max(1e-9, rms_max - rms_min), 0.0, 1.0)

    # Resample to simulation steps
    t_old = np.linspace(0, 1, T_raw)
    t_new = np.linspace(0, 1, resample_steps)
    M = np.interp(t_new, t_old, rms_arr)
    deg = np.interp(t_new, t_old, deg_raw)

    rng = np.random.default_rng(99)
    # Derive S and B from M using typical cross-correlation from IMS study
    S = np.clip(0.045 + (M - M.min()) / (M.max() - M.min() + 1e-9) * 0.85
                + 0.005 * rng.standard_normal(resample_steps), 0.0, 2.0)
    B = np.clip(0.080 + (M - M.min()) / (M.max() - M.min() + 1e-9) * 0.70
                + 0.010 * rng.standard_normal(resample_steps), 0.0, 2.0)
    E = np.clip(0.15 + 0.10 * np.sin(2 * np.pi * np.arange(resample_steps) / 50)
                + 0.04 * rng.standard_normal(resample_steps), 0.0, 1.0)

    return pd.DataFrame({
        "t":      np.arange(resample_steps),
        "M":      M,
        "S":      S,
        "B":      B,
        "E":      E,
        "deg":    deg,
        "source": f"nasa_femto:{os.path.basename(data_dir)}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# IMS loader (requires downloaded files)
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_ims(
    data_dir: str,
    bearing_col: int = 0,
    resample_steps: int = 500,
) -> pd.DataFrame:
    """
    Load IMS (University of Cincinnati) Bearing Dataset.

    Each file is a tab-separated matrix of 20480 samples x 4 bearings.
    Files are named as timestamps.

    Parameters
    ----------
    data_dir : str
        Path to test folder, e.g. "./data/IMS/1st_test"
    bearing_col : int
        Which bearing column (0-3)
    resample_steps : int

    Returns
    -------
    pd.DataFrame with columns: t, M, S, B, E, deg, source
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if not f.endswith((".csv", ".py", ".txt", ".md"))
    ])
    if not files:
        raise FileNotFoundError(
            f"No IMS data files found in {data_dir}
"
            f"Download IMS Bearing Dataset from:
"
            f"  https://www.nasa.gov/intelligent-systems-division/
"
            f"  (search: IMS Bearing Dataset Qiu 2006)
"
            f"Then place 1st_test/ etc. under ./data/IMS/"
        )

    rms_series = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            data = np.loadtxt(fpath)
            if data.ndim == 1:
                col = data
            else:
                col = data[:, bearing_col]
            rms_series.append(float(np.sqrt(np.mean(col ** 2))))
        except Exception:
            continue

    if not rms_series:
        raise ValueError(f"Could not parse IMS files in {data_dir}")

    rms_arr = np.array(rms_series)
    T_raw   = len(rms_arr)
    rms_min = np.percentile(rms_arr[:max(1, T_raw // 5)], 50)
    rms_max = rms_arr.max()
    deg_raw = np.clip((rms_arr - rms_min) / max(1e-9, rms_max - rms_min), 0.0, 1.0)

    t_old = np.linspace(0, 1, T_raw)
    t_new = np.linspace(0, 1, resample_steps)
    M   = np.interp(t_new, t_old, rms_arr)
    deg = np.interp(t_new, t_old, deg_raw)

    rng = np.random.default_rng(77)
    S = np.clip(0.035 + (M - M.min()) / (M.max() - M.min() + 1e-9) * 0.65
                + 0.004 * rng.standard_normal(resample_steps), 0.0, 1.5)
    B = np.clip(0.065 + (M - M.min()) / (M.max() - M.min() + 1e-9) * 0.55
                + 0.008 * rng.standard_normal(resample_steps), 0.0, 1.5)
    E = np.clip(0.15 + 0.08 * np.sin(2 * np.pi * np.arange(resample_steps) / 50)
                + 0.03 * rng.standard_normal(resample_steps), 0.0, 1.0)

    return pd.DataFrame({
        "t":      np.arange(resample_steps),
        "M":      M,
        "S":      S,
        "B":      B,
        "E":      E,
        "deg":    deg,
        "source": f"nasa_ims:{os.path.basename(data_dir)}_bearing{bearing_col}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    mode: str = "synthetic",
    profile: str = "femto_outer",
    data_dir: Optional[str] = None,
    bearing_col: int = 0,
    T: int = 500,
    fault_onset: Optional[int] = None,
    seed: int = 42,
    resample_steps: int = 500,
) -> pd.DataFrame:
    """
    Unified data loader.

    Parameters
    ----------
    mode : "synthetic" | "nasa_femto" | "nasa_ims"
    profile : bearing profile name (synthetic mode only)
    data_dir : path to dataset folder (nasa modes)
    bearing_col : bearing column index (ims mode)
    T : time steps (synthetic mode)
    fault_onset : fault start step (synthetic mode, default T//2)
    seed : random seed (synthetic mode)
    resample_steps : output length for nasa modes

    Returns
    -------
    pd.DataFrame with columns: t, M, S, B, E, deg, source
    """
    if mode == "synthetic":
        return generate_synthetic(profile_name=profile, T=T,
                                  fault_onset=fault_onset, seed=seed)
    elif mode == "nasa_femto":
        if data_dir is None:
            raise ValueError("data_dir required for nasa_femto mode")
        return load_nasa_femto(data_dir, resample_steps=resample_steps)
    elif mode == "nasa_ims":
        if data_dir is None:
            raise ValueError("data_dir required for nasa_ims mode")
        return load_nasa_ims(data_dir, bearing_col=bearing_col,
                             resample_steps=resample_steps)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use: synthetic | nasa_femto | nasa_ims")


if __name__ == "__main__":
    # Quick test — synthetic mode
    for prof in ["femto_outer", "femto_inner", "ims_wrist"]:
        df = load_data(mode="synthetic", profile=prof, T=500)
        print(f"[{prof}] shape={df.shape}  "
              f"M_max={df['M'].max():.3f}g  "
              f"B_max={df['B'].max():.3f}A  "
              f"source={df['source'].iloc[0]}")
