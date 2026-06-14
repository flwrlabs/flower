# fldsme: Federated Learning over IEEE 802.15.4e DSME IoT Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-1.31.0-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Baseline-blueviolet?style=flat-square)
![Venue](https://img.shields.io/badge/Venue-IEEE-00629B?style=flat-square&logo=ieee&logoColor=white)

**Author:** Sonali Anand

</div>

---

## Overview

This Flower Baseline simulates **federated learning over a realistic IEEE 802.15.4e DSME
(Deterministic and Synchronous Multi-channel Extension) cluster-tree IoT network**.

Each FL client represents an IoT end device whose participation is governed by physical
MAC-layer constraints derived from two published papers:

| Paper | Contribution to this baseline |
|-------|------------------------------|
| **PSO-DSME** — Anand et al., 2024 | Power consumption model (Eq. 3): computes energy cost per FL round per client |
| **SeCAP** — Anand et al., 2025 | Adaptive CAP model: determines GTS slot availability and bandwidth fraction per round |

This is the **first Flower Baseline to incorporate a physical MAC-layer model for FL
client selection and gradient compression**.

---

## What Makes This Baseline Novel

Standard Flower baselines assume every client can always participate and always transmit
its full gradient update. In real IoT deployments, neither is true:

- A device with a depleted battery **cannot train** that round
- A device with only 4 GTS slots allocated **can only transmit ~57% of its gradient**

This baseline models both constraints explicitly.

### 1. Energy-Gated Client Participation (PSO-DSME paper, Eq. 3)

Before training, each client computes its energy cost for this round:
P = 2^(MO-SO) x { Ptx.Ttx + Prx.Trx + Pidle.Tidle } / TMD

where Ptx=255 mW, Prx=135 mW, Pidle=1.3 mW (Table I, PSO paper).

If energy_budget < cost, the client returns num-examples=0 and is excluded
from aggregation. This is the MAC-layer gate for FL participation.

### 2. GTS Bandwidth Masking (SeCAP paper, CFP model)

After training, each client applies top-k sparsification to its gradient,
retaining only the fraction transmittable within its allocated GTS slots:

```python
n_keep = max(1, int(n_params * bandwidth_fraction))  # ~0.75-0.89 per round
```

### 3. SeCAP NCR Mode Energy Saving

When a cluster switches to NCR mode (SeCAP active), more frequent CAP slots
reduce idle waiting time, cutting radio energy cost by up to **61%**
(51.8 mJ -> 20.1 mJ).

### 4. DSMEFedAvg: Handling All-Depleted Rounds

When all sampled clients are energy-depleted, standard FedAvg raises
ZeroDivisionError. DSMEFedAvg detects this and holds the global model
unchanged — correctly modelling a DSME beacon interval with no transmissions.

---

## About the Papers

### Paper 1: PSO-DSME

> Anand, S., Choudhury, N., Ojha, T., Hazarika, A., Dave, J.
> "Improving Network Efficiency in Clustered Tree Topology through PSO
> Optimization in IEEE 802.15.4-DSME based IoT Networks." 2024.

Proposes PSO-based adaptive multi-superframe parameter tuning (BO, MO, SO).
The power model (Eq. 3) is the direct source of the energy gate in dsme_model.py.

### Paper 2: SeCAP

> Anand, S., Gorrela, A., Rahman, R., Choudhury, N., Hazarika, A.,
> Choudhury, D., Ojha, T.
> "Delay-Bounded Adaptive MAC for IEEE 802.15.4e DSME Networks: Enhancing Resilience under Bursty and Dynamic IoT Traffic" 2025.

Proposes Selective CAP Preservation: a delay-aware adaptive MAC algorithm
that toggles between CR and NCR modes based on M/M/1 queue occupancy.
The NCR energy model and GTS slot availability model drive the bandwidth
masking in client_app.py.

---

## DSME Cluster-Tree Topology
PAN Coordinator (PANC)

|-- Cluster 0  (Cluster Head + End Devices)

|-- Cluster 1  (Cluster Head + End Devices)

|-- Cluster 2  (Cluster Head + End Devices)  <- faster depletion rate

|-- Cluster 3  (Cluster Head + End Devices)
Multi-superframe (BO=6, MO=5, SO=3):
Superframes per multi-SF : 2^(MO-SO) = 4
Slots per superframe     : 16
CAP slots                : 8   (CSMA/CA contention)
CFP / GTS slots          : 7   (scheduled, deterministic)
Superframe duration      : ~122.9 ms
Multi-SF duration        : ~491.5 ms

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| FL algorithm | DSMEFedAvg (extends FedAvg) |
| Rounds | 20 |
| Clients | 10 IoT end devices |
| Clusters | 4 (BO=6, MO=5, SO=3) |
| Fraction train | 0.5 (5 clients sampled per round) |
| Local epochs | 1 |
| Learning rate | 0.1 (SGD, momentum=0.9) |
| Batch size | 32 |
| Dataset | CIFAR-10 (IID partition) |
| Model | Lightweight CNN (~62k params, ~241 KB) |
| Energy budget | 60 mJ per client per round |
| Base bandwidth fraction | 0.8 |
| Hardware | Apple M1 (CPU only) |

### Model Architecture

Input (3x32x32)

-> Conv2d(3, 6, k=5) -> ReLU -> MaxPool(2)

-> Conv2d(6, 16, k=5) -> ReLU -> MaxPool(2)

-> Flatten -> FC(400->120) -> ReLU

-> FC(120->84) -> ReLU

-> FC(84->10)

Total parameters: ~61,770 (~241 KB)

---

## Results

20 rounds, 10 clients, CIFAR-10 IID, energy budget 60 mJ:

| Round | Train Loss | Eval Acc | Avg Energy (mJ) | Avg BW Frac | Notes |
|-------|-----------|----------|----------------|-------------|-------|
| 1  | 2.176 | 11.0% | 51.8 | 0.798 | All active, CR mode |
| 2  | 2.107 | 15.5% | 51.8 | 0.747 | |
| 3  | 2.061 | 17.1% | 39.1 | 0.779 | NCR clients: 39.1 mJ |
| 4  | 2.055 | 23.6% | 51.8 | 0.831 | |
| 5  | 2.033 | 23.7% | 51.8 | 0.781 | |
| 6  | 2.013 | 22.7% | 32.7 | 0.844 | SeCAP NCR active |
| 7  | 1.994 | 22.1% | 51.8 | 0.769 | |
| 8  | 2.028 | 20.3% | 51.8 | 0.802 | |
| 9  | 2.029 | 24.8% | 20.1 | 0.888 | Full NCR: 61% energy saving |
| 10 | 2.009 | 26.4% | 51.8 | 0.763 | |
| 11 | 2.016 | 18.6% | 51.8 | 0.795 | |
| 12 | 2.027 | 23.1% | 20.1 | 0.882 | NCR clients active |
| 13 | --    | 23.1% | --   | --    | All depleted: model held |
| 14 | --    | 23.1% | --   | --    | All depleted: model held |
| 15 | 1.994 | 28.0% | 27.9 | 0.877 | Peak accuracy |
| 16 | --    | 28.0% | --   | --    | All depleted: model held |
| 17 | --    | 28.0% | --   | --    | All depleted: model held |
| 18 | 1.967 | 25.8% | 20.1 | 0.880 | NCR: 61% energy saving |
| 19 | --    | 25.8% | --   | --    | All depleted: model held |
| 20 | --    | 25.8% | --   | --    | All depleted: model held |

**Key observations:**

- Train loss decreased from 2.18 to 1.97 across active training rounds
- Eval accuracy climbed from 11.0% to 28.0% (peak, round 15), settling at 25.8%
- Rounds 13-14, 16-17, 19-20: all sampled clients energy-depleted. DSMEFedAvg
  held the global model unchanged, modelling a DSME beacon interval with no
  successful transmissions
- SeCAP NCR mode reduced energy from 51.8 mJ to 20.1 mJ (61% saving) in
  rounds 9, 12, 18 — showing how adaptive CAP management directly benefits
  FL participation rates
- GTS bandwidth fraction ranged 0.75-0.89 across all active rounds

---

## Environment Setup

```bash
git clone https://github.com/adap/flower.git
cd flower/baselines/fldsme

python3.12 -m venv venv
source venv/bin/activate

pip install -e .
```

---

## Running the Baseline

```bash
# Run with default config (20 rounds, 10 clients)
flwr run .

# Stream logs in real time
flwr run . --stream

# Override parameters at runtime
flwr run . --run-config "num-server-rounds=10 energy-budget-mj=80.0"
```

---

## Configuration Reference

```toml
[tool.flwr.app.config]
num-server-rounds = 20
fraction-train = 0.5
local-epochs = 1

# DSME cluster-tree topology
bo = 6
mo = 5
so = 3
num-clusters = 4

# MAC-layer energy budget per client per FL round (mJ)
energy-budget-mj = 60.0

# GTS bandwidth constraint
bandwidth-fraction = 0.8
```

---

## File Structure
```
fldsme/

|-- fldsme/
|   |-- init.py
|   |-- client_app.py    # Energy gate + GTS bandwidth mask
|   |-- server_app.py    # DSMEFedAvg: handles all-depleted rounds
|   |-- dsme_model.py    # MAC-layer model: PSO Eq.3 + SeCAP bandwidth
|   |-- model.py         # Lightweight CNN for CIFAR-10
|   |-- dataset.py       # CIFAR-10 loading and partitioning
|   |-- strategy.py      # Reserved for future extensions
|   +-- utils.py
|-- pyproject.toml
+-- README.md
```

---

## Citation

```bibtex
@inproceedings{anand2024psodsme,
  author    = {Anand, Sonali and Choudhury, Nikumani and Ojha, Tamoghna
               and Hazarika, Anakhi and Dave, Jay},
  title     = {Improving Network Efficiency in Clustered Tree Topology
               through {PSO} Optimization in {IEEE} 802.15.4-{DSME}
               based {IoT} Networks},
  year      = {2024},
  note      = {Supported by DST-SERB Grant SRG/2023/002016}
}

@inproceedings{anand2025secap,
  author    = {Anand, Sonali and Gorrela, Alekhya and Rahman, Raziur
               and Choudhury, Nikumani and Hazarika, Anakhi
               and Choudhury, Dipamani and Ojha, Tamoghna},
  title     = {Delay-Bounded Adaptive {MAC} for {IEEE} 802.15.4e {DSME}
               Networks: Enhancing Resilience under Bursty and Dynamic
               {IoT} Traffic},
  year      = {2025},
  note      = {Supported by DST-SERB Grant SRG/2023/002016}
}
```

---

## Author
**Sonali Anand**
sonalianand2406@gmail.com

IEEE 802.15.4e · DSME · Federated Learning · IoT · MAC Optimisation · Energy-Aware FL · Flower Baseline
