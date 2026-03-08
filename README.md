# OrganOS — Institutional Runtime Governance Architecture

> ⚠️ **DEMO / RESEARCH RELEASE** — Core architecture concepts are covered by patent priority applications (KIPO, 2025-2026). This code is shared for research and reproducibility. Commercial use requires separate licensing. See [Patent Notice](#patent-notice).

### Industrial Robot Predictive Maintenance Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20PCE-orange.svg)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
[![Patent Pending](https://img.shields.io/badge/Patent-KIPO%20Filed-green.svg)](#patent-notice)

> **First public release** of the OrganOS runtime governance simulation —  
> industrial robot joint failure prediction + self-recovery control.

---

## What is OrganOS?

OrganOS is a **runtime governance framework** for complex autonomous systems.  
Unlike conventional predictive maintenance that only detects faults,  
OrganOS **governs its own execution** based on evolving system states —  
suppressing damage through staged self-recovery rather than just raising alarms.

> *Complex systems should not only predict outcomes,  
> but also govern their own execution based on evolving system states.*

---

## Architecture

```
Sensor Stream ──► CollapseOS ──► BangAyu ──► AnomalyDetector ──► Sidecar
   M / S / B      robot_step()   L0-L3 FSM    5-stage FSM       Fail-Closed
```

| Module | Role | Key Mechanism |
|--------|------|---------------|
| **CollapseOS** | Damage & collapse modeling | Joint damage D, collapse velocity C, Ring-topology network propagation |
| **BangAyu** | Adaptive judgment | Risk-Dose accumulation → L0:100% / L1:80% / L2:50% / L3:Stop |
| **AnomalyDetector** | State machine | Kalman filter + FSM: NORMAL→SOFT_GUARD→PRE_SEAL→SEAL→RECOVER |
| **Sidecar v1.2** | Execution safety | Fail-Closed kernel, HMAC-signed execution permits |

---

## Results

Validated on **NASA Prognostics Center of Excellence** bearing dataset parameters.

### Damage Suppression — 3 NASA Profiles

| Profile | Reactive D | **OrganOS D** | Damage Reduction |
|---------|-----------|--------------|-----------------|
| femto_outer (NASA FEMTO outer race) | 1.000 | **0.329** | **67.2%** |
| femto_inner (NASA FEMTO inner race) | 1.000 | **0.476** | **52.4%** |
| ims_wrist   (IMS wrist bearing)     | 1.000 | **0.242** | **75.8%** |

### Algorithm Benchmark — Same Dataset

| Algorithm | FAR | Final D | Lead Time |
|-----------|-----|---------|-----------|
| Threshold (3σ) | 1.8% | 1.000 | 38.0h |
| EWMA Chart | 2.3% | 1.000 | 36.0h |
| Random Forest | 0.0% | 0.496 | 0.0h |
| LSTM | 5.0% | 1.000 | 13.0h |
| **OrganOS (this work)** | **7.73%** | **0.502** | **9.5h** |

> Others detect faults but reach D=1.000 (complete failure).  
> OrganOS holds damage at **0.502** through continuous self-recovery.

### Pareto Frontier

The simulation reveals a physical constraint:  
FAR and damage cannot be simultaneously minimized.  
OrganOS operates on this frontier at **FAR=7.73%, D=0.502** —  
confirmed via grid search over 684 candidate parameter sets.

![Benchmark](results/benchmark_comparison.png)
![Pareto](results/v3_finetuned_final.png)

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/organos-robot-pdm
cd organos-robot-pdm
pip install numpy pandas matplotlib scipy

# Synthetic data — runs immediately, no download needed
python sim/robot_sim.py

# Specific profile
python sim/robot_sim.py --profile femto_outer
python sim/robot_sim.py --profile femto_inner
python sim/robot_sim.py --profile ims_wrist

# 5-algorithm benchmark
python sim/benchmark.py
```

**With NASA real data:**
```bash
# Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
python sim/robot_sim.py --source nasa_femto --data_dir ./data/FEMTO/Bearing1_1
```

---

## Confirmed Parameters (v3 Fine-tuned)

```python
ORGANOS_CONFIG = {
    "theta1":     0.32,   # SOFT_GUARD entry
    "theta2":     0.46,   # PRE_SEAL entry
    "theta3":     0.60,   # SEAL / Fail-Closed trigger
    "dC_eta":     0.18,   # Collapse velocity spike sensitivity
    "dose_scale": 20.0,
    # FAR=7.73%  final_D=0.502  lead_time=9.5h
}
```

Grid search: θ1 ∈ [0.27,0.33], θ2 ∈ [0.42,0.50], θ3 ∈ [0.57,0.65], η ∈ {0.10,0.12,0.15,0.18,0.20}  
Score = 0.7×D + 0.3×FAR, minimized over 684 FAR≤10.5% candidates.

---

## File Structure

```
organos-robot-pdm/
├── sim/
│   ├── robot_sim.py       # Main simulation (all 4 modules)
│   ├── benchmark.py       # 5-algorithm benchmark
│   ├── nasa_params.py     # NASA FEMTO / IMS parameters
│   └── data_loader.py     # Real data + synthetic fallback
├── results/
│   ├── kpi_summary.csv
│   ├── benchmark_summary.csv
│   ├── v3_finetuned_kpi.csv
│   ├── comparison_all_profiles.png
│   ├── benchmark_comparison.png
│   └── v3_finetuned_final.png
└── README.md
```

---

## Dataset References

- **NASA FEMTO**: Nectoux et al., *PRONOSTIA: An experimental platform for bearings accelerated degradation tests*, IEEE ICPHM 2012. → [NASA PCE Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **IMS Bearing**: Qiu et al., *Wavelet filter-based weak signature detection method*, Journal of Sound and Vibration, 289(4), 2006.

Raw dataset files are **not included** (license). Download from NASA PCE and place under `./data/`.

---

## Patent Notice

Core OrganOS components are covered by patent priority applications filed with KIPO:

- Failure-Density based Risk-Dose accumulation
- Fail-Closed execution kernel
- Bio-inspired collapse propagation modeling

This repository shares simulation code for **research and reproducibility**.  
Commercial deployment requires separate licensing.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Min-Gi Kim — Independent AI Safety Researcher & System Architect*  
*OrganOS / Institutional Runtime Governance Architecture / 2026*
