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
robot_sim.py
------------
OrganOS Robot Predictive Maintenance Simulation
CollapseOS + BangAyu + AnomalyDetector integrated pipeline

Sensor parameters derived from:
  - NASA FEMTO Bearing Dataset (Nectoux et al. 2012)
  - IMS Bearing Dataset (Qiu et al. 2006)

Usage:
  python robot_sim.py                          # all 3 profiles, synthetic
  python robot_sim.py --profile femto_inner    # single profile
  python robot_sim.py --source nasa_femto --data_dir ./data/FEMTO/Bearing1_1
  python robot_sim.py --source nasa_ims   --data_dir ./data/IMS/1st_test
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data
from nasa_params  import get_profile, PROFILES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CollapseOS: robot_step (joint-level damage)
# ─────────────────────────────────────────
def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def robot_step(st, stress, dt, maint_rate, wear_r, drift_r, batt_r):
    st2 = dict(st)
    st2["M"] = clip(st["M"] + dt*(wear_r *(0.5+stress) - maint_rate*0.6*st["M"]))
    st2["S"] = clip(st["S"] + dt*(drift_r*(0.5+stress) - maint_rate*0.5*st["S"]))
    st2["B"] = clip(st["B"] + dt*(batt_r *(0.5+stress) - maint_rate*0.4*st["B"]))
    return st2

def damage_from_state(st, weights):
    return clip(weights["M"]*st["M"] + weights["S"]*st["S"] + weights["B"]*st["B"])

def mode_update(mode, D, C, dDdt):
    if mode == "NORMAL":
        return "COLLAPSE" if D >= 0.35 and dDdt >= 0.015 else "NORMAL"
    if mode == "COLLAPSE":
        return "CONTAIN"  if D >= 0.55 or C >= 0.08 else "COLLAPSE"
    if mode == "CONTAIN":
        return "RECOVER"  if C <= 0.03 and D <= 0.55 else "CONTAIN"
    if mode == "RECOVER":
        if C <= 0.015 and dDdt <= 0.005: return "NORMAL"
        if D >= 0.35  and dDdt >= 0.015: return "COLLAPSE"
        return "RECOVER"
    return mode

# ─────────────────────────────────────────
# AnomalyDetector: 5-stage FSM
# ─────────────────────────────────────────
FSM = ["NORMAL","SOFT_GUARD","PRE_SEAL","SEAL","RECOVER"]

def fsm_step(prev, D_eff, dC, theta1, theta2, theta3, eta=0.05):
    if   D_eff < theta1: base = 0
    elif D_eff < theta2: base = 1
    elif D_eff < theta3: base = 2
    else:                base = 3
    if dC >= eta: base = min(base+1, 3)
    if prev == 3: return 4 if D_eff < theta2 else 3
    if prev == 4: return 0 if D_eff < theta1 else 4
    return base

# ─────────────────────────────────────────
# BangAyu: Risk-Dose accumulation
# ─────────────────────────────────────────
def bangayu_step(riskdose, M, S, B, p_profile, decay=0.92, gain=0.35):
    R = clip(0.35*(M/p_profile.vib_fault_max)
           + 0.35*(S/p_profile.pos_fault_max)
           + 0.30*(B/p_profile.curr_fault_max))
    riskdose = riskdose*decay + R*gain
    if riskdose >= 1.10 or R >= 0.90: mode = "L3"
    else:
        C_score = clip(0.55*R + 0.35*min(riskdose/1.1, 1.0))
        if   C_score >= 0.85: mode = "L3"
        elif C_score >= 0.60: mode = "L2"
        elif C_score >= 0.30: mode = "L1"
        else:                 mode = "L0"
    speed = {"L0":100,"L1":80,"L2":50,"L3":0}[mode]
    return riskdose, R, mode, speed

# ─────────────────────────────────────────
# Core simulation: one run
# ─────────────────────────────────────────
def run_sim(df, profile_name, use_predictive=False, dt=1.0):
    p  = get_profile(profile_name)
    T  = len(df)
    lam = 0.95

    # CollapseOS wear rates scaled to profile
    wear_r  = 0.010 * (p.vib_fault_max  / 2.5)
    drift_r = 0.008 * (p.pos_fault_max  / 0.9)
    batt_r  = 0.009 * (p.curr_fault_max / 0.85)

    maint_base   = 0.008
    maint_active = 0.008 if not use_predictive else 0.055

    st      = {"M": p.vib_normal_mean/p.vib_fault_max*0.12,
               "S": p.pos_normal_mean/p.pos_fault_max*0.10,
               "B": p.curr_normal_mean/p.curr_fault_max*0.14}
    mode_c  = "NORMAL"
    C       = 0.0
    D_hat   = 0.08
    D_dose  = 0.0
    fsm_s   = 0
    riskdose= 0.0

    # auto-theta from warmup
    warm = 60
    theta1, theta2, theta3 = 0.18, 0.32, 0.50

    recs = []
    downtime = 0
    interv   = 0

    for t in range(T):
        row = df.iloc[t]
        M_norm = row["M"] / p.vib_fault_max
        S_norm = row["S"] / p.pos_fault_max
        B_norm = row["B"] / p.curr_fault_max

        sensor_stress = clip(0.4*M_norm + 0.3*S_norm + 0.3*B_norm)
        stress = clip(0.6*D_hat + 0.35*row["E"] + 0.25*sensor_stress)

        maint_rate = maint_base
        if use_predictive and fsm_s >= 2:
            maint_rate = maint_active
            interv += 1

        prev_D = D_hat
        st = robot_step(st, stress, dt, maint_rate, wear_r, drift_r, batt_r)
        D_raw = damage_from_state(st, p.damage_weights)
        D_hat = clip(0.80*D_hat + 0.20*D_raw)
        dDdt  = (D_hat - prev_D) / max(dt, 1e-12)
        mode_c= mode_update(mode_c, D_hat, C, dDdt)

        damp = 0.65 if mode_c=="CONTAIN" else 0.18
        aC   = 0.04 if mode_c=="COLLAPSE" else 0.0
        C = clip(C + dt*(0.80*max(0,dDdt) + aC*C - damp*C))

        R_inst = clip(0.5*D_hat + 0.3*C + 0.2*sensor_stress)
        D_dose = lam*D_dose + R_inst

        riskdose, R_by, by_mode, speed = bangayu_step(
            riskdose, row["M"], row["S"], row["B"], p)

        prev_fsm = fsm_s
        dC_val = abs(C - (recs[-1]["C"] if recs else 0.0))
        fsm_s  = fsm_step(fsm_s, D_dose/20.0, dC_val, theta1, theta2, theta3)

        if fsm_s == 3: downtime += 1

        recs.append({"t":t, "M":row["M"], "S":row["S"], "B":row["B"],
                     "D_hat":D_hat, "C":C, "D_dose":D_dose,
                     "mode_collapse":mode_c, "fsm_state":fsm_s,
                     "fsm_name":FSM[fsm_s], "riskdose":riskdose,
                     "bangayu_mode":by_mode, "robot_speed":speed,
                     "intervening":int(use_predictive and fsm_s>=2),
                     "deg_truth":row["deg"]})

    rdf = pd.DataFrame(recs)
    kpi = {
        "profile":          profile_name,
        "mode":             "predictive" if use_predictive else "reactive",
        "source":           df["source"].iloc[0],
        "T":                T,
        "final_D":          round(float(rdf["D_hat"].iloc[-1]), 4),
        "peak_D":           round(float(rdf["D_hat"].max()), 4),
        "downtime_steps":   int(downtime),
        "downtime_pct":     round(downtime/T*100, 2),
        "intervention_cnt": int(interv),
        "L3_steps":         int((rdf["bangayu_mode"]=="L3").sum()),
        "seal_first":       int(rdf[rdf["fsm_state"]==3]["t"].min()) if downtime>0 else None,
    }
    return rdf, kpi

# ─────────────────────────────────────────
# Network collapse (6 robots)
# ─────────────────────────────────────────
def run_network(df, profile_name, n=6, coupling=0.25, seed=7):
    p   = get_profile(profile_name)
    T   = len(df)
    rng = np.random.default_rng(seed)
    wear_r  = 0.010*(p.vib_fault_max/2.5)
    drift_r = 0.008*(p.pos_fault_max/0.9)
    batt_r  = 0.009*(p.curr_fault_max/0.85)

    A = np.zeros((n,n))
    for i in range(n):
        A[i,(i-1)%n]=0.5; A[i,(i+1)%n]=0.5
    A *= coupling

    states = [{"M":0.05+rng.random()*0.04,
               "S":0.04+rng.random()*0.03,
               "B":0.06+rng.random()*0.04} for _ in range(n)]
    D_hat  = np.array([0.07+rng.random()*0.03 for _ in range(n)])
    C_net  = np.zeros(n)
    modes  = ["NORMAL"]*n
    recs   = []

    for t in range(T):
        row = df.iloc[t]
        s_stresses = np.array([
            clip(0.4*(row["M"]/p.vib_fault_max)
               + 0.3*(row["S"]/p.pos_fault_max)
               + 0.3*(row["B"]/p.curr_fault_max)) if i==0
            else clip(0.12+rng.standard_normal()*0.04)
            for i in range(n)
        ])
        cv = A @ C_net
        D_prev = D_hat.copy()
        for i in range(n):
            stress = clip(0.6*D_hat[i] + 0.35*row["E"] + cv[i] + 0.2*s_stresses[i])
            states[i] = robot_step(states[i], stress, 1.0, 0.010, wear_r, drift_r, batt_r)
            D_hat[i]  = clip(0.80*D_hat[i] + 0.20*damage_from_state(states[i], p.damage_weights))
        dDdt_v = D_hat - D_prev
        for i in range(n):
            modes[i] = mode_update(modes[i], D_hat[i], C_net[i], dDdt_v[i])
            damp = 0.65 if modes[i]=="CONTAIN" else 0.18
            aC   = 0.04 if modes[i]=="COLLAPSE" else 0.0
            C_net[i] = clip(C_net[i]+(0.80*max(0,dDdt_v[i])+aC*C_net[i]-damp*C_net[i]))
        recs.append({
            "t":t, "D_avg":float(D_hat.mean()), "D_max":float(D_hat.max()),
            "C_avg":float(C_net.mean()),
            "frac_collapse":float(np.mean(np.array(modes)=="COLLAPSE")),
            "frac_fail":float(np.mean(D_hat>=0.85)),
        })
    return pd.DataFrame(recs)

# ─────────────────────────────────────────
# Plot: 4-panel per profile
# ─────────────────────────────────────────
def plot_profile(df, r_df, p_df, net_df, r_kpi, p_kpi, profile_name, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    src = df["source"].iloc[0]
    fig.suptitle(
        f"OrganOS Robot PdM Simulation — Profile: {profile_name}
"
        f"Data source: {src}",
        fontsize=13, fontweight='bold', y=0.98)

    t = df["t"].values
    RC = '#e05c5c'; PC = '#3a86c8'; TC = '#888888'

    # (1) Damage
    ax = axes[0]
    ax.plot(t, r_df["D_hat"], color=RC, lw=2, label=f"D_hat Reactive  (final={r_kpi['final_D']})")
    ax.plot(t, p_df["D_hat"], color=PC, lw=2, label=f"D_hat Predictive (final={p_kpi['final_D']})")
    ax.plot(t, df["deg"]*0.85, color=TC, lw=1.5, ls='--', alpha=0.5, label="True degradation (norm.)")
    ax.axhline(0.85, color='red', lw=1, ls='--', alpha=0.5, label="Fault threshold D=0.85")
    if r_kpi["seal_first"]:
        ax.axvline(r_kpi["seal_first"], color=RC, lw=1.5, ls='--', alpha=0.7,
                   label=f"SEAL onset (reactive) t={r_kpi['seal_first']}")
    ax.fill_between(t, p_df["intervening"]*0.04, alpha=0.15, color=PC, label="PdM intervention")
    ax.set_title(f"(1) Damage Score D  |  Downtime: Reactive={r_kpi['downtime_pct']}%  "
                 f"Predictive={p_kpi['downtime_pct']}%", fontsize=10)
    ax.set_ylabel("Damage D_hat"); ax.set_ylim(0,1); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (2) BangAyu mode
    ax = axes[1]
    mc = {0:'#4caf50',1:'#ffeb3b',2:'#ff9800',3:'#f44336'}
    mm = {'L0':0,'L1':1,'L2':2,'L3':3}
    for i in range(len(t)-1):
        ax.axvspan(t[i],t[i+1],ymin=0,ymax=1,alpha=0.30,
                   color=mc[mm[p_df["bangayu_mode"].iloc[i]]])
    ax2 = ax.twinx()
    ax2.plot(t, p_df["riskdose"], color='#222', lw=1.5, label="Risk-Dose")
    ax2.axhline(1.10, color='red', lw=1, ls='--', alpha=0.7, label="Fail-Closed thr.")
    ax2.set_ylabel("Risk-Dose"); ax2.legend(fontsize=8, loc='upper right')
    ax.set_yticks([0.125,0.375,0.625,0.875])
    ax.set_yticklabels(['L0
100%','L1
80%','L2
50%','L3
Stop'], fontsize=8)
    patches = [mpatches.Patch(color=mc[i],alpha=0.5,
               label=['L0 Normal','L1 Slow','L2 Inspect','L3 Stop'][i]) for i in range(4)]
    ax.legend(handles=patches, fontsize=8, loc='upper left', ncol=4)
    ax.set_title(f"(2) BangAyu Decision Mode & Risk-Dose  |  "
                 f"L3 steps (predictive): {p_kpi['L3_steps']}", fontsize=10)
    ax.grid(alpha=0.2)

    # (3) FSM
    ax = axes[2]
    ax.fill_between(t, r_df["fsm_state"], alpha=0.4, color=RC, label="FSM Reactive")
    ax.fill_between(t, p_df["fsm_state"], alpha=0.4, color=PC, label="FSM Predictive")
    ax.plot(t, r_df["fsm_state"], color=RC, lw=1.5)
    ax.plot(t, p_df["fsm_state"], color=PC, lw=1.5)
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(FSM, fontsize=8)
    ax.set_title(f"(3) AnomalyDetector FSM  |  "
                 f"SEAL steps: Reactive={r_kpi['downtime_steps']}  "
                 f"Predictive={p_kpi['downtime_steps']}", fontsize=10)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)

    # (4) Network
    ax = axes[3]
    ax.plot(t, net_df["D_avg"], color='#333', lw=2, label="D avg (6 robots)")
    ax.plot(t, net_df["D_max"], color=RC, lw=2, label="D max (6 robots)")
    ax.plot(t, net_df["C_avg"], color='#9c27b0', lw=1.5, ls='--', label="Collapse velocity C")
    ax.fill_between(t, net_df["frac_collapse"], alpha=0.25, color='orange', label="Fraction COLLAPSE")
    ax.axhline(0.85, color='red', lw=1, ls='--', alpha=0.4)
    ax.set_title("(4) CollapseOS Network Propagation — 6 Robots, Ring Topology", fontsize=10)
    ax.set_ylabel("Damage / Fraction"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Time Step  (1 step = 30 min field operation)", fontsize=9)
        ax.set_xlim(0, len(t))

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] {out_path}")

# ─────────────────────────────────────────
# Comparison plot: 3 profiles side by side
# ─────────────────────────────────────────
def plot_comparison(all_results, out_path):
    profiles = list(all_results.keys())
    n = len(profiles)
    fig, axes = plt.subplots(3, n, figsize=(6*n, 15))
    fig.suptitle("OrganOS Robot PdM — Profile Comparison
"
                 "(NASA FEMTO outer/inner race + IMS wrist joint)",
                 fontsize=13, fontweight='bold')

    for col, prof in enumerate(profiles):
        df, r_df, p_df, r_kpi, p_kpi = all_results[prof]
        t = df["t"].values
        RC='#e05c5c'; PC='#3a86c8'

        # Row 0: Damage
        ax = axes[0][col]
        ax.plot(t, r_df["D_hat"], color=RC, lw=2, label="Reactive")
        ax.plot(t, p_df["D_hat"], color=PC, lw=2, label="Predictive")
        ax.axhline(0.85, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_title(f"{prof}
Damage D_hat", fontsize=9)
        ax.set_ylim(0,1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel("Step (30min/step)", fontsize=8)

        # Row 1: FSM state
        ax = axes[1][col]
        ax.fill_between(t, r_df["fsm_state"], alpha=0.4, color=RC, label="Reactive")
        ax.fill_between(t, p_df["fsm_state"], alpha=0.4, color=PC, label="Predictive")
        ax.set_yticks([0,1,2,3,4]); ax.set_yticklabels(FSM, fontsize=6)
        ax.set_title("FSM State", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel("Step (30min/step)", fontsize=8)

        # Row 2: KPI bar
        ax = axes[2][col]
        metrics = ['final_D','peak_D','downtime_pct']
        labels  = ['Final D','Peak D','Downtime %']
        x = np.arange(len(metrics))
        r_vals = [r_kpi[m]/100 if m=='downtime_pct' else r_kpi[m] for m in metrics]
        p_vals = [p_kpi[m]/100 if m=='downtime_pct' else p_kpi[m] for m in metrics]
        ax.bar(x-0.2, r_vals, 0.35, color=RC, alpha=0.8, label="Reactive")
        ax.bar(x+0.2, p_vals, 0.35, color=PC, alpha=0.8, label="Predictive")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0,1.05)
        for i,(rv,pv) in enumerate(zip(r_vals,p_vals)):
            ax.text(i-0.2, rv+0.02, f"{rv:.2f}", ha='center', fontsize=7, color=RC)
            ax.text(i+0.2, pv+0.02, f"{pv:.2f}", ha='center', fontsize=7, color=PC)
        ax.set_title("KPI Comparison", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [comparison plot] {out_path}")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile",  default="all",
                    help="femto_outer | femto_inner | ims_wrist | all")
    ap.add_argument("--source",   default="synthetic",
                    help="synthetic | nasa_femto | nasa_ims")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--bearing_col", type=int, default=0)
    ap.add_argument("--T",        type=int, default=500)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    run_profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    all_results  = {}
    all_kpis     = []

    print("=" * 65)
    print("OrganOS Robot PdM Simulation")
    print(f"  Source  : {args.source}")
    print(f"  Profiles: {run_profiles}")
    print("=" * 65)

    for prof in run_profiles:
        print(f"
[{prof}]")

        if args.source == "synthetic":
            df = load_data(mode="synthetic", profile=prof, T=args.T, seed=args.seed)
        elif args.source == "nasa_femto":
            df = load_data(mode="nasa_femto", data_dir=args.data_dir,
                           resample_steps=args.T)
        else:
            df = load_data(mode="nasa_ims", data_dir=args.data_dir,
                           bearing_col=args.bearing_col, resample_steps=args.T)
        print(f"  Data: {df['source'].iloc[0]}  shape={df.shape}")

        r_df, r_kpi = run_sim(df, prof, use_predictive=False)
        p_df, p_kpi = run_sim(df, prof, use_predictive=True)
        net_df      = run_network(df, prof)

        d_improve   = (1 - p_kpi["final_D"] / max(r_kpi["final_D"],1e-6)) * 100
        dt_improve  = (1 - p_kpi["downtime_pct"] / max(r_kpi["downtime_pct"],1e-6)) * 100 \
                      if r_kpi["downtime_pct"] > 0 else 0.0

        print(f"  Reactive  : D={r_kpi['final_D']}  downtime={r_kpi['downtime_pct']}%")
        print(f"  Predictive: D={p_kpi['final_D']}  downtime={p_kpi['downtime_pct']}%")
        print(f"  Improvement: damage={d_improve:.1f}%  downtime={dt_improve:.1f}%")

        r_kpi["damage_improvement_pct"]   = round(d_improve, 1)
        r_kpi["downtime_improvement_pct"] = round(dt_improve, 1)

        # Per-profile plot
        plot_profile(df, r_df, p_df, net_df, r_kpi, p_kpi, prof,
                     os.path.join(OUT_DIR, f"sim_{prof}.png"))

        # Save CSV
        r_df.to_csv(os.path.join(OUT_DIR, f"ts_{prof}_reactive.csv"),   index=False)
        p_df.to_csv(os.path.join(OUT_DIR, f"ts_{prof}_predictive.csv"), index=False)
        net_df.to_csv(os.path.join(OUT_DIR, f"net_{prof}.csv"),         index=False)

        all_results[prof] = (df, r_df, p_df, r_kpi, p_kpi)
        all_kpis.extend([r_kpi, p_kpi])

    # Comparison plot
    if len(run_profiles) > 1:
        plot_comparison(all_results, os.path.join(OUT_DIR, "comparison_all_profiles.png"))

    # KPI summary CSV
    kpi_df = pd.DataFrame(all_kpis)
    kpi_df.to_csv(os.path.join(OUT_DIR, "kpi_summary.csv"), index=False)
    print(f"
[KPI Summary]")
    print(kpi_df[["profile","mode","final_D","peak_D","downtime_pct",
                  "intervention_cnt","L3_steps"]].to_string(index=False))
    print(f"
[Done] results -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
