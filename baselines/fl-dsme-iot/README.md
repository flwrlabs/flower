# fl-dsme-iot: Federated Learning over IEEE 802.15.4e DSME IoT Networks

## Overview

This Flower Baseline simulates federated learning over a realistic IEEE 802.15.4e DSME (Deterministic and Synchronous Multi-channel Extension) cluster-tree IoT network. 
Each FL client is an IoT end device whose participation is governed by 
physical MAC-layer constraints derived from two published papers:

1. **PSO-DSME** (Anand et al., 2024) — provides the power consumption model(Eq. 3) used to compute each client's energy cost per FL round
2. **SeCAP** (Anand et al., 2025) — provides the adaptive CAP model that determines GTS slot availability and bandwidth fraction per round

This is the first Flower Baseline to incorporate a physical MAC-layer model for FL client selection and gradient compression.

## Novel Contributions

- **Energy-gated client participation**: before training, each client checks whether its remaining energy budget covers the DSME radio + compute cost. Depleted clients 
return `num-examples=0` and are excluded from aggregation.
- **GTS bandwidth masking**: after training, each client applies top-k sparsification to its gradient update, retaining only the fraction transmittable within its allocated 
GTS slots in the CFP.
- **SeCAP NCR mode**: when queue length exceeds the delay threshold, the coordinator switches to NCR mode (CAP in every superframe), roughly halving the radio energy cost 
and increasing GTS availability.
- **DSMEFedAvg**: extends FedAvg to handle rounds where all sampled clients are energy-depleted, keeping the global model unchanged (modelling a DSME beacon interval with 
no successful transmissions).

## DSME Topology

Cluster-tree (BO=6, MO=5, SO=3):

4 multi-superframe duration = 2^(MO-SO) = 4 superframes
CAP slots per superframe: 8
CFP (GTS) slots per superframe: 7
4 clusters, 10 IoT end devices total

## About this Baseline

**Papers:**
- Anand, S., Choudhury, N., Ojha, T., Hazarika, A., Dave, J.
  "Improving Network Efficiency in Clustered Tree Topology through PSO Optimization in IEEE 802.15.4-DSME based IoT Networks." 2024.
- Anand, S., Gorrela, A., Rahman, R., Choudhury, N., Hazarika, A.,Choudhury, D., Ojha, T. "Delay-Bounded Adaptive MAC for IEEE 802.15.4e DSME Networks: Enhancing Resilience under Bursty and Dynamic IoT Traffic" 2025.

**Task:** Image classification (CIFAR-10 as proxy for IoT sensor data)

**Model:** Lightweight CNN (~62k parameters, ~241 KB)

Conv2d(3,6,5) -> MaxPool -> Conv2d(6,16,5) -> MaxPool -> FC(400,120) -> FC(120,84) -> FC(84,10)

**Dataset:** CIFAR-10, IID partition across 10 clients

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| FL rounds | 20 |
| Clients | 10 IoT end devices |
| Clusters | 4 (BO=6, MO=5, SO=3) |
| Fraction train | 0.5 (5 clients sampled per round) |
| Local epochs | 1 |
| Learning rate | 0.1 |
| Batch size | 32 |
| Energy budget | 60 mJ per client |
| Base bandwidth fraction | 0.8 |

## Environment

Python 3.12

flwr[simulation] >= 1.24.0

torch == 2.8.0

torchvision == 0.23.0

Tested on Apple M1 (CPU only).

## Quickstart

```bash
git clone https://github.com/adap/flower.git
cd flower/baselines/fl-dsme-iot
pip install -e .
flwr run .
```

## Results

20 rounds, 10 clients, CIFAR-10 IID, energy budget 60 mJ:

| Round | Train Loss | Eval Acc | Avg Energy (mJ) | Avg BW Frac | Notes |
|-------|-----------|----------|----------------|-------------|-------|
| 1  | 2.176 | 11.0% | 51.8 | 0.798 | All active (CR mode) |
| 3  | 2.061 | 17.1% | 39.1 | 0.779 | NCR clients: 39.1 mJ |
| 5  | 2.033 | 23.7% | 51.8 | 0.781 | |
| 9  | 2.029 | 24.8% | 20.1 | 0.888 | Full NCR: 20.1 mJ |
| 10 | 2.009 | 26.4% | 51.8 | 0.763 | |
| 12 | 2.027 | 23.1% | 20.1 | 0.882 | NCR clients active |
| 13 | — | 23.1% | — | — | All depleted: model held |
| 15 | 1.994 | 28.0% | 27.9 | 0.877 | Best eval accuracy |
| 18 | 1.967 | 25.8% | 20.1 | 0.880 | NCR saves 61% energy |
| 20 | — | 25.8% | — | — | All depleted: model held |

**Key observations:**
- Train loss decreased from **2.18 → 1.97** over active rounds
- Eval accuracy peaked at **28.0%** (round 15), final **25.8%** (round 20)
- Rounds 13, 14, 16, 17, 19, 20: all sampled clients energy-depleted —
  DSMEFedAvg held the global model unchanged (DSME beacon skip)
- SeCAP NCR mode reduced energy cost from 51.8 mJ → 20.1 mJ (**61% saving**)
- GTS bandwidth fraction ranged **0.75–0.89** per round

## Configuration

All hyperparameters are set in `pyproject.toml` under `[tool.flwr.app.config]`:

```toml
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

## File Structure
fl-dsme-iot/

├── fl_dsme_iot/

│   ├── client_app.py    # DSME-gated FL client with energy check + BW mask

│   ├── server_app.py    # DSMEFedAvg: handles all-depleted rounds gracefully

│   ├── dsme_model.py    # MAC-layer energy/bandwidth model (PSO + SeCAP)

│   ├── model.py         # Lightweight CNN for CIFAR-10

│   ├── dataset.py       # CIFAR-10 data loading and partitioning

│   └── strategy.py      # (reserved for future custom strategy extensions)

├── pyproject.toml

└── README.md
\## Citation

If you use this baseline, please cite:

```bibtex
@inproceedings{anand2024psodsme,
  author    = {Anand, Sonali and Choudhury, Nikumani and Ojha, Tamoghna
               and Hazarika, Anakhi and Dave, Jay},
  title     = {Improving Network Efficiency in Clustered Tree Topology
               through {PSO} Optimization in {IEEE} 802.15.4-{DSME}
               based {IoT} Networks},
  year      = {2024}
}

@inproceedings{anand2025secap,
  author    = {Anand, Sonali and Gorrela, Alekhya and Rahman, Raziur
               and Choudhury, Nikumani and Hazarika, Anakhi
               and Choudhury, Dipamani and Ojha, Tamoghna},
  title     = {Delay-Bounded Adaptive {MAC} for {IEEE} 802.15.4e {DSME}
               Networks: Enhancing Resilience under Bursty and Dynamic
               {IoT} Traffic},
  year      = {2025}
}
```
