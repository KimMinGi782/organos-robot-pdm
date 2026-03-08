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
benchmark.py
------------
같은 NASA 파라미터 기반 데이터셋에서
5가지 방법을 직접 실행해 수치 비교

Methods:
  1. Threshold-based   -- 단순 임계값 알람 (시중 기본 방식)
  2. EWMA              -- 지수가중이동평균 (공정관리 표준)
  3. Random Forest     -- sklearn 기반 ML 분류
  4. LSTM              -- numpy 수동 구현 (라이브러리 없이)
  5. OrganOS           -- CollapseOS + BangAyu + AnomalyDetector FSM

Metrics (같은 기준으로 측정):
  - Detection Lead Time  : 실제 고장 시작 대비 몇 스텝 전에 탐지했나
  - False Alarm Rate     : 정상 구간에서 알람 발생 비율
  - Final Damage D       : 500스텝 후 손상도 (예지정비 적용 시)
  - Downtime Rate        : SEAL/정지 상태 비율
  - Intervention Count   : 사전 개입 횟수
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data
from nasa_params  import get_profile

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

FAULT_ONSET = 250   # 실제 고장 시작 스텝 (ground truth)
T           = 500
SEED        = 42

# ─────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────
def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def compute_metrics(alert_arr, fault_onset, T, name):
    """
    alert_arr: 0/1 배열 (1=이상 탐지)
    fault_onset: 실제 고장 시작 스텝
    """
    # Lead time: 고장 전 첫 알람
    pre_fault_alerts = np.where((alert_arr == 1) & (np.arange(T) < fault_onset))[0]
    if len(pre_fault_alerts) > 0:
        lead_time = fault_onset - pre_fault_alerts[-1]  # 마지막 사전 알람 기준
    else:
        lead_time = 0

    # False alarm rate: 정상 구간(0~fault_onset-30) 중 알람 비율
    normal_zone = alert_arr[:max(1, fault_onset - 30)]
    far = float(normal_zone.mean()) if len(normal_zone) > 0 else 0.0

    # Detection rate in fault zone
    fault_zone = alert_arr[fault_onset:]
    dr = float(fault_zone.mean()) if len(fault_zone) > 0 else 0.0

    return {
        "method": name,
        "lead_time_steps": int(lead_time),
        "lead_time_hours": round(lead_time * 0.5, 1),
        "false_alarm_rate": round(far, 4),
        "detection_rate_fault_zone": round(dr, 4),
    }

# ─────────────────────────────────────────
# 1. Threshold-based (가장 단순, 시중 기본)
# ─────────────────────────────────────────
def run_threshold(df, profile_name):
    """단순 임계값: 각 센서가 정상범위 N*sigma를 넘으면 알람"""
    p = get_profile(profile_name)
    M, S, B = df["M"].values, df["S"].values, df["B"].values

    # 정상 구간으로 임계값 설정 (처음 50스텝)
    warmup = 50
    thr_M = M[:warmup].mean() + 3 * M[:warmup].std()
    thr_S = S[:warmup].mean() + 3 * S[:warmup].std()
    thr_B = B[:warmup].mean() + 3 * B[:warmup].std()

    alert = ((M > thr_M) | (S > thr_S) | (B > thr_B)).astype(int)

    # 사후정비: 알람 후에도 그냥 운전 → 손상 계산
    D_series = _compute_damage_no_intervention(df, profile_name)

    m = compute_metrics(alert, FAULT_ONSET, T, "Threshold (3-sigma)")
    m["final_D"]        = round(float(D_series[-1]), 4)
    m["peak_D"]         = round(float(D_series.max()), 4)
    m["downtime_pct"]   = round(float((alert[FAULT_ONSET:]==1).mean()*100), 2)
    m["intervention_cnt"] = 0
    m["alert_series"]   = alert
    m["D_series"]       = D_series
    return m

# ─────────────────────────────────────────
# 2. EWMA (지수가중이동평균 관리도)
# ─────────────────────────────────────────
def run_ewma(df, profile_name, lam=0.15, L=3.0):
    """
    EWMA 관리도 (공정관리 표준 방식)
    lam: smoothing factor, L: control limit multiplier
    """
    p = get_profile(profile_name)
    # 복합 신호: 센서 정규화 합산
    sig = (normalize(df["M"].values) * 0.4
         + normalize(df["S"].values) * 0.3
         + normalize(df["B"].values) * 0.3)

    warmup = 50
    mu0  = sig[:warmup].mean()
    sig0 = sig[:warmup].std()

    ewma = np.zeros(T)
    ewma[0] = sig[0]
    for t in range(1, T):
        ewma[t] = lam * sig[t] + (1 - lam) * ewma[t-1]

    sigma_ewma = sig0 * np.sqrt(lam / (2 - lam))
    UCL = mu0 + L * sigma_ewma

    alert    = (ewma > UCL).astype(int)
    D_series = _compute_damage_no_intervention(df, profile_name)

    m = compute_metrics(alert, FAULT_ONSET, T, "EWMA Control Chart")
    m["final_D"]        = round(float(D_series[-1]), 4)
    m["peak_D"]         = round(float(D_series.max()), 4)
    m["downtime_pct"]   = round(float((alert[FAULT_ONSET:]==1).mean()*100), 2)
    m["intervention_cnt"] = 0
    m["alert_series"]   = alert
    m["D_series"]       = D_series
    m["ewma_series"]    = ewma
    m["UCL"]            = UCL
    return m

# ─────────────────────────────────────────
# 3. Random Forest (sklearn 없이 수동 구현)
# numpy만으로 간단한 결정트리 앙상블
# ─────────────────────────────────────────
def run_random_forest(df, profile_name, n_trees=20, seed=99):
    """
    간단한 Random Forest: window feature → binary fault/normal
    훈련: 앞 60스텝 정상 + 뒤 60스텝(고장구간)으로 라벨링
    """
    rng = np.random.default_rng(seed)
    M = normalize(df["M"].values)
    S = normalize(df["S"].values)
    B = normalize(df["B"].values)
    E = normalize(df["E"].values)

    # Feature: [M, S, B, E, M_diff, rolling_mean_M(5)]
    w = 5
    feats = np.zeros((T, 6))
    for t in range(T):
        feats[t, 0] = M[t]
        feats[t, 1] = S[t]
        feats[t, 2] = B[t]
        feats[t, 3] = E[t]
        feats[t, 4] = M[t] - (M[t-1] if t > 0 else M[t])
        feats[t, 5] = M[max(0,t-w):t+1].mean()

    # 라벨: 0=정상(0~FAULT_ONSET-30), 1=고장(FAULT_ONSET+30이후)
    train_normal = list(range(0, FAULT_ONSET - 30))
    train_fault  = list(range(FAULT_ONSET + 30, min(T, FAULT_ONSET + 120)))
    train_idx    = train_normal + train_fault
    X_tr  = feats[train_idx]
    y_tr  = np.array([0]*len(train_normal) + [1]*len(train_fault))

    # 단순 결정트리 (depth=3) 수동 구현
    class SimpleTree:
        def __init__(self): self.nodes = {}
        def fit(self, X, y, feat_subset):
            self._fit(X, y, feat_subset, node_id=0, depth=0, max_depth=3)
        def _fit(self, X, y, feat_subset, node_id, depth, max_depth):
            if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 4:
                self.nodes[node_id] = ('leaf', int(np.round(y.mean())))
                return
            best_fi, best_thr, best_gain = 0, 0.5, -1
            for fi in feat_subset:
                for thr in np.percentile(X[:,fi], [25, 50, 75]):
                    left  = y[X[:,fi] <= thr]
                    right = y[X[:,fi] >  thr]
                    if len(left)==0 or len(right)==0: continue
                    gain = len(y)*_gini(y) - len(left)*_gini(left) - len(right)*_gini(right)
                    if gain > best_gain:
                        best_gain, best_fi, best_thr = gain, fi, thr
            self.nodes[node_id] = ('split', best_fi, best_thr)
            mask = X[:, best_fi] <= best_thr
            self._fit(X[mask],  y[mask],  feat_subset, 2*node_id+1, depth+1, max_depth)
            self._fit(X[~mask], y[~mask], feat_subset, 2*node_id+2, depth+1, max_depth)
        def predict(self, x):
            nid = 0
            for _ in range(10):
                if nid not in self.nodes: return 0
                node = self.nodes[nid]
                if node[0] == 'leaf': return node[1]
                _, fi, thr = node
                nid = 2*nid+1 if x[fi] <= thr else 2*nid+2
            return 0

    def _gini(y):
        if len(y) == 0: return 0
        p = y.mean()
        return 2 * p * (1 - p)

    n_feats = feats.shape[1]
    trees = []
    for _ in range(n_trees):
        idx   = rng.choice(len(X_tr), size=len(X_tr), replace=True)
        fsub  = rng.choice(n_feats, size=max(2, n_feats//2), replace=False)
        tree  = SimpleTree()
        tree.fit(X_tr[idx], y_tr[idx], fsub)
        trees.append(tree)

    preds = np.array([[t.predict(feats[i]) for t in trees] for i in range(T)])
    proba = preds.mean(axis=1)
    alert = (proba >= 0.5).astype(int)

    D_series = _compute_damage_with_intervention(df, profile_name, alert)

    m = compute_metrics(alert, FAULT_ONSET, T, "Random Forest")
    seal_steps = int((alert[FAULT_ONSET:]==1).sum())
    m["final_D"]        = round(float(D_series[-1]), 4)
    m["peak_D"]         = round(float(D_series.max()), 4)
    m["downtime_pct"]   = round(seal_steps / max(1, T - FAULT_ONSET) * 100, 2)
    m["intervention_cnt"] = int(alert[:FAULT_ONSET].sum())
    m["alert_series"]   = alert
    m["D_series"]       = D_series
    m["proba_series"]   = proba
    return m

# ─────────────────────────────────────────
# 4. LSTM (numpy 수동 구현, 단층)
# ─────────────────────────────────────────
def run_lstm(df, profile_name, hidden=16, seq_len=10, epochs=30, seed=77):
    """
    단층 LSTM (numpy 수동) - 재구성 오차 기반 이상탐지
    정상 구간으로 훈련 → 재구성 오차가 크면 이상
    """
    rng = np.random.default_rng(seed)

    sig = np.column_stack([
        normalize(df["M"].values),
        normalize(df["S"].values),
        normalize(df["B"].values),
    ])  # (T, 3)

    input_size  = 3
    output_size = 3
    h = hidden

    # LSTM 가중치 초기화 (He)
    scale = 0.1
    Wf = rng.normal(0, scale, (h, input_size + h));  bf = np.zeros(h)
    Wi = rng.normal(0, scale, (h, input_size + h));  bi = np.zeros(h)
    Wo = rng.normal(0, scale, (h, input_size + h));  bo = np.zeros(h)
    Wg = rng.normal(0, scale, (h, input_size + h));  bg = np.zeros(h)
    Wy = rng.normal(0, scale, (output_size, h));     by = np.zeros(output_size)

    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    tanh    = np.tanh

    def lstm_forward(seq):
        ht, ct = np.zeros(h), np.zeros(h)
        preds  = []
        for x in seq:
            xh = np.concatenate([x, ht])
            f  = sigmoid(Wf @ xh + bf)
            i  = sigmoid(Wi @ xh + bi)
            o  = sigmoid(Wo @ xh + bo)
            g  = tanh(Wg @ xh + bg)
            ct = f*ct + i*g
            ht = o * tanh(ct)
            y  = sigmoid(Wy @ ht + by)
            preds.append(y)
        return np.array(preds), ht, ct

    # 훈련: 정상 구간 시퀀스 (0 ~ FAULT_ONSET-40)
    train_sig = sig[:FAULT_ONSET - 40]
    lr = 0.005

    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, len(train_sig) - seq_len - 1, seq_len):
            inp = train_sig[start:start+seq_len]
            tgt = train_sig[start+1:start+seq_len+1]
            preds, _, _ = lstm_forward(inp)
            err = preds - tgt
            total_loss += (err**2).mean()
            # 간단한 파라미터 업데이트 (BPTT 생략, output layer만)
            # err: (seq_len, output_size), preds: (seq_len, hidden) via ht
            grad = lr * err.mean(axis=0)  # (output_size,)
            Wy -= np.outer(grad, np.ones(h)) * 0.01

    # 전체 재구성 오차
    recon_errors = np.zeros(T)
    for t in range(seq_len, T):
        inp = sig[t-seq_len:t]
        tgt = sig[t-seq_len+1:t+1]
        try:
            pred, _, _ = lstm_forward(inp)
            recon_errors[t] = float(np.mean((pred - tgt)**2))
        except:
            recon_errors[t] = 0.0

    # 임계값: 정상 구간 재구성 오차의 95 percentile
    normal_errors = recon_errors[seq_len:FAULT_ONSET-40]
    threshold = np.percentile(normal_errors, 95) if len(normal_errors) > 0 else 0.01

    alert    = (recon_errors > threshold).astype(int)
    alert[:seq_len] = 0

    D_series = _compute_damage_with_intervention(df, profile_name, alert)

    m = compute_metrics(alert, FAULT_ONSET, T, "LSTM (Recon. Error)")
    seal_steps = int((alert[FAULT_ONSET:]==1).sum())
    m["final_D"]        = round(float(D_series[-1]), 4)
    m["peak_D"]         = round(float(D_series.max()), 4)
    m["downtime_pct"]   = round(seal_steps / max(1, T - FAULT_ONSET) * 100, 2)
    m["intervention_cnt"] = int(alert[:FAULT_ONSET].sum())
    m["alert_series"]   = alert
    m["D_series"]       = D_series
    m["recon_errors"]   = recon_errors
    m["threshold"]      = threshold
    return m

# ─────────────────────────────────────────
# 5. OrganOS (CollapseOS + BangAyu + FSM)
# ─────────────────────────────────────────
FSM_STATES = ["NORMAL","SOFT_GUARD","PRE_SEAL","SEAL","RECOVER"]

def run_organos(df, profile_name):
    p = get_profile(profile_name)
    T_loc = len(df)
    lam   = 0.95
    wear_r  = 0.010 * (p.vib_fault_max  / 2.5)
    drift_r = 0.008 * (p.pos_fault_max  / 0.9)
    batt_r  = 0.009 * (p.curr_fault_max / 0.85)

    st       = {"M": 0.08, "S": 0.07, "B": 0.10}
    mode_c   = "NORMAL"
    C        = 0.0
    D_hat    = 0.08
    D_dose   = 0.0
    fsm_s    = 0
    riskdose = 0.0
    theta1, theta2, theta3 = 0.18, 0.32, 0.50

    alert    = np.zeros(T_loc, dtype=int)
    D_series = np.zeros(T_loc)
    fsm_arr  = np.zeros(T_loc, dtype=int)
    rd_arr   = np.zeros(T_loc)
    downtime = 0
    interv   = 0

    maint_base   = 0.008
    maint_active = 0.055

    for t in range(T_loc):
        row = df.iloc[t]
        M_n = row["M"] / p.vib_fault_max
        S_n = row["S"] / p.pos_fault_max
        B_n = row["B"] / p.curr_fault_max
        sensor_stress = clip(0.4*M_n + 0.3*S_n + 0.3*B_n)
        stress = clip(0.6*D_hat + 0.35*row["E"] + 0.25*sensor_stress)

        maint_rate = maint_base
        if fsm_s >= 2:
            maint_rate = maint_active
            interv += 1

        prev_D = D_hat
        st["M"] = clip(st["M"] + (wear_r *(0.5+stress) - maint_rate*0.6*st["M"]))
        st["S"] = clip(st["S"] + (drift_r*(0.5+stress) - maint_rate*0.5*st["S"]))
        st["B"] = clip(st["B"] + (batt_r *(0.5+stress) - maint_rate*0.4*st["B"]))
        D_raw  = clip(p.damage_weights["M"]*st["M"]
                    + p.damage_weights["S"]*st["S"]
                    + p.damage_weights["B"]*st["B"])
        D_hat  = clip(0.80*D_hat + 0.20*D_raw)
        dDdt   = D_hat - prev_D

        # CollapseOS mode
        if   mode_c=="NORMAL":   mode_c = "COLLAPSE" if D_hat>=0.35 and dDdt>=0.015 else "NORMAL"
        elif mode_c=="COLLAPSE": mode_c = "CONTAIN"  if D_hat>=0.55 or C>=0.08 else "COLLAPSE"
        elif mode_c=="CONTAIN":  mode_c = "RECOVER"  if C<=0.03 and D_hat<=0.55 else "CONTAIN"
        elif mode_c=="RECOVER":
            if C<=0.015 and dDdt<=0.005: mode_c="NORMAL"
            elif D_hat>=0.35 and dDdt>=0.015: mode_c="COLLAPSE"

        damp = 0.65 if mode_c=="CONTAIN" else 0.18
        aC   = 0.04 if mode_c=="COLLAPSE" else 0.0
        C = clip(C + (0.80*max(0,dDdt) + aC*C - damp*C))

        # BangAyu Risk-Dose
        R = clip(0.35*M_n + 0.35*S_n + 0.30*B_n)
        riskdose = riskdose*0.92 + R*0.35
        rd_arr[t] = riskdose

        # D-Dose
        R_inst = clip(0.5*D_hat + 0.3*C + 0.2*sensor_stress)
        D_dose = lam*D_dose + R_inst

        # FSM
        dC_val = abs(C - (D_series[t-1]*0 + (D_series[t-1] if t>0 else 0)))
        prev_fsm = fsm_s
        d_eff = D_dose / 20.0
        if   d_eff < theta1: base = 0
        elif d_eff < theta2: base = 1
        elif d_eff < theta3: base = 2
        else:                base = 3
        if dC_val >= 0.05: base = min(base+1, 3)
        if   fsm_s==3: fsm_s = 4 if d_eff < theta2 else 3
        elif fsm_s==4: fsm_s = 0 if d_eff < theta1 else 4
        else:          fsm_s = base

        alert[t]    = 1 if fsm_s >= 2 else 0
        D_series[t] = D_hat
        fsm_arr[t]  = fsm_s
        if fsm_s == 3: downtime += 1

    m = compute_metrics(alert, FAULT_ONSET, T_loc, "OrganOS (This System)")
    seal_steps = int((fsm_arr[FAULT_ONSET:]==3).sum())
    m["final_D"]        = round(float(D_series[-1]), 4)
    m["peak_D"]         = round(float(D_series.max()), 4)
    m["downtime_pct"]   = round(downtime / T_loc * 100, 2)
    m["intervention_cnt"] = int(interv)
    m["alert_series"]   = alert
    m["D_series"]       = D_series
    m["fsm_arr"]        = fsm_arr
    m["riskdose_arr"]   = rd_arr
    return m

# ─────────────────────────────────────────
# 손상도 보조 함수
# ─────────────────────────────────────────
def _compute_damage_no_intervention(df, profile_name):
    """개입 없이 손상 진행"""
    p = get_profile(profile_name)
    wear_r  = 0.010*(p.vib_fault_max/2.5)
    drift_r = 0.008*(p.pos_fault_max/0.9)
    batt_r  = 0.009*(p.curr_fault_max/0.85)
    st = {"M":0.08,"S":0.07,"B":0.10}
    D_hat = 0.08
    D_series = np.zeros(len(df))
    for t in range(len(df)):
        row = df.iloc[t]
        M_n = row["M"]/p.vib_fault_max; S_n = row["S"]/p.pos_fault_max; B_n = row["B"]/p.curr_fault_max
        stress = clip(0.4*M_n+0.3*S_n+0.3*B_n+0.3*row["E"])
        st["M"] = clip(st["M"]+(wear_r *(0.5+stress)-0.008*0.6*st["M"]))
        st["S"] = clip(st["S"]+(drift_r*(0.5+stress)-0.008*0.5*st["S"]))
        st["B"] = clip(st["B"]+(batt_r *(0.5+stress)-0.008*0.4*st["B"]))
        D_raw   = clip(p.damage_weights["M"]*st["M"]+p.damage_weights["S"]*st["S"]+p.damage_weights["B"]*st["B"])
        D_hat   = clip(0.80*D_hat+0.20*D_raw)
        D_series[t] = D_hat
    return D_series

def _compute_damage_with_intervention(df, profile_name, alert_arr):
    """알람 시 정비 개입"""
    p = get_profile(profile_name)
    wear_r  = 0.010*(p.vib_fault_max/2.5)
    drift_r = 0.008*(p.pos_fault_max/0.9)
    batt_r  = 0.009*(p.curr_fault_max/0.85)
    st = {"M":0.08,"S":0.07,"B":0.10}
    D_hat = 0.08
    D_series = np.zeros(len(df))
    for t in range(len(df)):
        row = df.iloc[t]
        M_n = row["M"]/p.vib_fault_max; S_n = row["S"]/p.pos_fault_max; B_n = row["B"]/p.curr_fault_max
        stress = clip(0.4*M_n+0.3*S_n+0.3*B_n+0.3*row["E"])
        maint  = 0.055 if alert_arr[t]==1 else 0.008
        st["M"] = clip(st["M"]+(wear_r *(0.5+stress)-maint*0.6*st["M"]))
        st["S"] = clip(st["S"]+(drift_r*(0.5+stress)-maint*0.5*st["S"]))
        st["B"] = clip(st["B"]+(batt_r *(0.5+stress)-maint*0.4*st["B"]))
        D_raw   = clip(p.damage_weights["M"]*st["M"]+p.damage_weights["S"]*st["S"]+p.damage_weights["B"]*st["B"])
        D_hat   = clip(0.80*D_hat+0.20*D_raw)
        D_series[t] = D_hat
    return D_series

# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
def plot_benchmark(results, df, profile_name, out_path):
    t = df["t"].values
    methods = [r["method"] for r in results]
    colors  = ['#888888','#f4a261','#2a9d8f','#e76f51','#3a86c8']

    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle(
        f"Algorithm Benchmark — {profile_name} (NASA parameter profile)
"
        f"Fault onset: t={FAULT_ONSET} ({FAULT_ONSET*0.5:.0f} hrs)  |  "
        f"Total: {T} steps ({T*0.5:.0f} hrs)",
        fontsize=12, fontweight='bold')

    # (1) Damage D over time
    ax = axes[0]
    for r, col in zip(results, colors):
        ax.plot(t, r["D_series"], color=col, lw=2, label=f"{r['method']} (final D={r['final_D']})")
    ax.axvline(FAULT_ONSET, color='orange', lw=2, ls=':', label=f"Fault onset t={FAULT_ONSET}")
    ax.axhline(0.85, color='red', lw=1, ls='--', alpha=0.6, label="Failure threshold D=0.85")
    ax.set_title("(1) Damage Score D — all methods (with maintenance intervention)", fontsize=10)
    ax.set_ylabel("Damage D"); ax.set_ylim(0,1.05)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (2) Alert signals
    ax = axes[1]
    for i, (r, col) in enumerate(zip(results, colors)):
        offset = i * 1.1
        ax.fill_between(t, offset, offset + r["alert_series"]*0.8,
                        alpha=0.6, color=col, label=r["method"])
        ax.plot(t, offset + r["alert_series"]*0.8, color=col, lw=0.8, alpha=0.5)
    ax.axvline(FAULT_ONSET, color='orange', lw=2, ls=':', alpha=0.8)
    ax.set_yticks([i*1.1 + 0.4 for i in range(len(results))])
    ax.set_yticklabels([r["method"] for r in results], fontsize=8)
    ax.set_title("(2) Alert/Detection Signal — pre-fault detection comparison", fontsize=10)
    ax.grid(alpha=0.2)

    # (3) KPI bar chart
    ax = axes[2]
    metrics = ["lead_time_hours","false_alarm_rate","final_D","downtime_pct"]
    labels  = ["Lead Time
(hours)","False Alarm
Rate","Final
Damage D","Downtime
(%)"]
    scales  = [1.0, 100.0, 1.0, 1.0]   # FAR을 % 단위로

    x = np.arange(len(metrics))
    bar_w = 0.15
    for i, (r, col) in enumerate(zip(results, colors)):
        vals = [r["lead_time_hours"],
                r["false_alarm_rate"]*100,
                r["final_D"],
                r["downtime_pct"]]
        bars = ax.bar(x + i*bar_w, vals, bar_w, color=col, alpha=0.85, label=r["method"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.1f}", ha='center', fontsize=6.5, color=col, fontweight='bold')

    ax.set_xticks(x + bar_w * (len(results)-1)/2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("(3) KPI Comparison — lower is better except Lead Time", fontsize=10)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3, axis='y')
    note = ("* Lead Time: higher=better (earlier detection)  "
            "* False Alarm Rate: lower=better  "
            "* Final D / Downtime: lower=better")
    ax.text(0.01, -0.12, note, transform=ax.transAxes, fontsize=7.5, color='gray')

    for ax in axes[:2]:
        ax.set_xlabel("Time Step (1 step = 30 min)", fontsize=9)
        ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [benchmark plot] {out_path}")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 65)
    print("Algorithm Benchmark: OrganOS vs Threshold vs EWMA vs RF vs LSTM")
    print("Same dataset (NASA FEMTO outer race parameters)")
    print("=" * 65)

    profile = "femto_outer"
    df = load_data(mode="synthetic", profile=profile, T=T,
                   fault_onset=FAULT_ONSET, seed=SEED)
    print(f"Data: {df['source'].iloc[0]}  shape={df.shape}
")

    results = []
    for name, fn in [
        ("Threshold",     lambda: run_threshold(df, profile)),
        ("EWMA",          lambda: run_ewma(df, profile)),
        ("Random Forest", lambda: run_random_forest(df, profile)),
        ("LSTM",          lambda: run_lstm(df, profile)),
        ("OrganOS",       lambda: run_organos(df, profile)),
    ]:
        t0 = time.time()
        r  = fn()
        elapsed = time.time() - t0
        r["runtime_ms"] = round(elapsed*1000, 1)
        results.append(r)
        print(f"[{r['method']:28s}]  "
              f"LeadTime={r['lead_time_hours']:5.1f}h  "
              f"FAR={r['false_alarm_rate']:.3f}  "
              f"finalD={r['final_D']:.4f}  "
              f"downtime={r['downtime_pct']:.1f}%  "
              f"interv={r['intervention_cnt']}  "
              f"({elapsed*1000:.0f}ms)")

    # Summary table
    cols = ["method","lead_time_steps","lead_time_hours","false_alarm_rate",
            "detection_rate_fault_zone","final_D","peak_D",
            "downtime_pct","intervention_cnt","runtime_ms"]
    summary = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    summary.to_csv(os.path.join(OUT_DIR, "benchmark_summary.csv"), index=False)

    print("
" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(summary[["method","lead_time_hours","false_alarm_rate",
                   "final_D","downtime_pct","intervention_cnt"]].to_string(index=False))

    # Plot
    plot_benchmark(results, df, profile,
                   os.path.join(OUT_DIR, "benchmark_comparison.png"))

    print(f"
[Done] -> {OUT_DIR}/benchmark_*")
    return summary

if __name__ == "__main__":
    main()
