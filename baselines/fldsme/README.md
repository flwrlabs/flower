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

This Flower Baseline simulates **federated learning over an IEEE 802.15.4e DSME
(Deterministic and Synchronous Multi-channel Extension) cluster-tree IoT network**.

Each FL client represents an IoT end device whose participation is governed by
MAC-layer constraints derived from two published papers:

| Paper | Contribution to this baseline |
|-------|------------------------------|
| **PSO-DSME** — Anand et al., 2024 | Power consumption model (Eq. 3): energy cost per FL round per client |
| **SeCAP** — Anand et al., 2025 | Adaptive CAP model: GTS slot availability and bandwidth fraction per round |

As far as we are aware, this is the first Flower Baseline in which client
participation and gradient compression are driven by a physical MAC-layer model
rather than by a synthetic dropout probability.

---

## What makes this baseline different

Standard FL baselines assume every client can always participate and always
transmit its full update. In battery-powered IoT deployments neither holds:

- a device with a depleted budget **cannot train** that round
- a device allocated 4 of 7 GTS slots **can only transmit part of its update**

Both constraints are modelled explicitly, and — importantly — both can be
switched off, so the cost they impose is measurable rather than assumed.

### 1. Energy-gated participation (PSO-DSME, Eq. 3)

Before training, each client computes its cost for the round:

```
P = 2^(MO-SO) x { Ptx·Ttx + Prx·Trx + Pidle·Tidle } / TMD
```

with Ptx = 255 mW, Prx = 135 mW, Pidle = 1.3 mW (Table I, PSO paper). If the
residual budget is below the cost, the client returns `num-examples=0` and is
excluded from aggregation.

### 2. GTS bandwidth masking (SeCAP, CFP model)

After training, each client top-k sparsifies its **update delta** — not its
absolute weights — keeping the fraction transmittable in its allocated slots,
and transmits `global + sparse_update`:

```python
n_keep = max(1, int(n_params * bandwidth_fraction))   # ~0.75-0.89 per round
```

Sparsifying the delta rather than the weights matters: masking absolute weights
zeroes real parameters and corrupts the global model during aggregation.

### 3. SeCAP NCR mode

Switching a cluster to NCR mode shortens idle waiting, cutting per-round cost
from 51.78 mJ to 20.05 mJ. Under the budget used here this is not only an
efficiency gain — see [Results](#results).

### 4. DSMEFedAvg

Extends `FedAvg` with two behaviours standard FedAvg lacks:

- **All-depleted rounds.** When every sampled client returns `num-examples=0`,
  FedAvg raises `ZeroDivisionError`. `DSMEFedAvg` holds the global model
  unchanged, modelling a beacon interval with no successful transmissions.
- **Correct metric accounting.** FedAvg weights metrics by `num-examples`, so
  skipped clients have zero weight and vanish from the round record.
  `dsme_metrics_aggregation` counts active and skipped clients separately.

### 5. Control arm (`NullMACModel`)

Setting `dsme-enabled=false` substitutes a MAC model that returns "eligible,
zero cost, full bandwidth" for every client. The control arm therefore runs the
**identical code path** with the constraints set to no-op values, rather than
taking a separate branch. This is what makes the comparison in the results below
a measurement rather than an assertion.

---

## Results

Five seeds (0–4), 20 rounds, 10 clients, `fraction-train = 0.5`. Mean ± std.

| Arm | Final acc (%) | Peak acc (%) | Total energy (mJ) | Rounds with skipped clients |
|---|---|---|---|---|
| FedAvg (no DSME gate) | 31.2 ± 1.9 | 31.2 | 0.0 | 0 / 20 |
| DSME (MAC-gated) | 24.8 ± 2.5 | 26.6 | 553.8 | 17 / 20 |

### The MAC layer costs 6.4 accuracy points

31.2% → 24.8% over 20 rounds. The two arms differ only in whether the
MAC-layer model is active.

For scale: repeating the DSME arm under a different RNG stream, five seeds,
identical code, gave 24.1 ± 4.3 — a **0.7-point** spread. The 6.4-point gap is
roughly nine times the measured run-to-run noise. That 0.7 figure is also the
bar any future client-selection strategy must clear; a 1-point improvement in
this setup is not distinguishable from noise.

### Accuracy declines after the feasibility horizon

The DSME arm peaks at 26.6% and finishes at 24.8%. The control arm peaks at its
final round. The 1.8-point decline follows from *which* clients remain eligible.

Per-round cost is 51.78 mJ in CR mode against a 60 mJ starting budget that
drains at 1.5–3.1 mJ per round by cluster, with 5 mJ recharged every five
rounds. Solving for when a client can still afford a round:

| Mode | Cost/round | Contiguous horizon | Last reachable round |
|---|---|---|---|
| CR | 51.78 mJ | round 8 | round 15 |
| NCR | 20.05 mJ | round 20 | round 76 |

**Uninterrupted CR-mode training stops after round 8.** Rounds 10, 15 and 20
train only because a harvest event lifts clients back above the threshold —
participation past round 8 is scheduled by the 5-round recharge period, not by
the energy budget.

The clients that survive longest are those in the slowest-draining clusters, so late-round updates come from a shrinking, non-randomly selected subset. Because partitions here are IID, this is a variance effect rather than a data-distribution bias: fewer contributing clients per round means noisier aggregates, and the same few clients contribute repeatedly. Under non-IID partitioning the same mechanism would additionally skew the global model toward whichever data those clusters hold — a stronger effect that this setup does not measure.

### NCR mode as an enabling condition

Past round 15 no client can afford a CR-mode round at any point in the remaining
schedule. Every round that trains from there on is an NCR round: `ncr_fraction`
is non-zero at rounds 3, 6, 9, 12, 15 and 18 and zero elsewhere, tracking the
`fl_round % 3 == 0` condition that puts clusters 0 and 2 into NCR mode.

Extending the calculation to 100 rounds, the CR horizon stays at 15 while the
NCR horizon reaches 76. Under this budget, adaptive CAP is better described as
the condition under which the federation remains trainable than as a 61%
energy saving.

`feasibility.py` computes these horizons from the MAC model alone, without
running training:

```bash
python feasibility.py --rounds 20
python feasibility.py --sweep-size --rounds 200
```

### Figures

| | |
|---|---|
| `docs/accuracy_vs_round.png` | Both arms, mean ± std bands |
| `docs/participation.png` | Active vs energy-depleted clients per round |
| `docs/energy_per_round.png` | CR vs NCR energy draw |

Reproduce:

```bash
python run_experiments.py --seeds 0 1 2 3 4 --rounds 20
python plot_results.py --results results --out docs
```

---

## Limitations

**Five seeds.** The DSME arm's standard deviation is 2.5 points across seeds and
0.7 across identical repeats. That is enough to separate the 6.4-point control
gap from noise and not much more.

**Node sampling is not pinned.** `configure_train` samples 5 of 10 nodes using
node IDs assigned fresh by the SuperLink each session, so the same seed does not
reproduce the same subset across runs. Model initialisation, DataLoader shuffle,
and the MAC bandwidth draw are all seeded; the sampled subset is not. Results
are averaged over seeds rather than exactly reproducible run to run.

**CIFAR-10 is not an IoT workload.** A 241 KB update over a 250 kbps 802.15.4
radio is not a realistic deployment. The dataset and architecture come from the
standard Flower baseline template so the FL side stays comparable with other
baselines; the contribution here is the MAC-layer model. A sensor-native dataset
(UCI HAR, WISDM, keyword spotting) with a TinyML-scale model would make the
absolute energy figures meaningful rather than illustrative.

**The energy model is analytical, not trace-driven.** Costs come from the
closed-form power model in PSO-DSME and the service-rate model in SeCAP. There
is no packet-level simulation, so collisions, retransmissions, beacon scheduling
and channel hopping are not represented. Validation against openDSME or an
OMNeT++ trace is the obvious next step.

**Bandwidth is modelled, not scheduled.** The GTS allocation is a per-client
fraction drawn from a seeded RNG, not the output of a slot-scheduling algorithm.

**Depletion is deterministic.** Drain is a fixed function of cluster ID and
recharge is a fixed 5 mJ every 5 rounds. Real harvesting is stochastic.

---

## About the papers

### Paper 1: PSO-DSME

> Anand, S., Choudhury, N., Ojha, T., Hazarika, A., Dave, J.
> "Improving Network Efficiency in Clustered Tree Topology through PSO
> Optimization in IEEE 802.15.4-DSME based IoT Networks." 2024.

PSO-based adaptive multi-superframe parameter tuning (BO, MO, SO). The power
model (Eq. 3) is the direct source of the energy gate in `dsme_model.py`.

### Paper 2: SeCAP

> Anand, S., Gorrela, A., Rahman, R., Choudhury, N., Hazarika, A.,
> Choudhury, D., Ojha, T.
> "Delay-Bounded Adaptive MAC for IEEE 802.15.4e DSME Networks: Enhancing
> Resilience under Bursty and Dynamic IoT Traffic." 2025.

Selective CAP Preservation: a delay-aware adaptive MAC algorithm toggling
between CR and NCR modes on M/M/1 queue occupancy. The NCR energy model and GTS
availability model drive the bandwidth masking in `client_app.py`.

---

## DSME cluster-tree topology

```
PAN Coordinator (PANC)
  |-- Cluster 0  (Cluster Head + End Devices)   drain 1.5 mJ/round
  |-- Cluster 1  (Cluster Head + End Devices)   drain 2.3 mJ/round
  |-- Cluster 2  (Cluster Head + End Devices)   drain 3.1 mJ/round
  +-- Cluster 3  (Cluster Head + End Devices)   drain 1.5 mJ/round
```

Multi-superframe (BO=6, MO=5, SO=3):

```
Superframes per multi-SF : 2^(MO-SO) = 4
Slots per superframe     : 16
CAP slots                : 8   (CSMA/CA contention)
CFP / GTS slots          : 7   (scheduled, deterministic)
Superframe duration      : ~122.9 ms
Multi-SF duration        : ~491.5 ms
```

---

## Experimental setup

| Parameter | Value |
|-----------|-------|
| FL algorithm | DSMEFedAvg (extends FedAvg) |
| Rounds | 20 |
| Seeds | 5 (0–4) |
| Clients | 10 IoT end devices |
| Clusters | 4 (BO=6, MO=5, SO=3) |
| Fraction train | 0.5 (5 clients sampled per round) |
| Local epochs | 1 |
| Learning rate | 0.1 (SGD, momentum 0.9) |
| Batch size | 32 |
| Dataset | CIFAR-10 (IID partition) |
| Model | Lightweight CNN (~61,770 params, ~241 KB) |
| Initial energy budget | 60 mJ per client, depleting over rounds |
| Recharge | 5 mJ every 5 rounds |
| Base bandwidth fraction | 0.8 |
| Hardware | Apple M1 (MPS) |

Note that the energy budget is a **per-client reserve that depletes across
rounds**, not a per-round allowance. This is what produces the feasibility
horizon discussed above.

### Model architecture

```
Input (3x32x32)
  -> Conv2d(3, 6, k=5)  -> ReLU -> MaxPool(2)
  -> Conv2d(6, 16, k=5) -> ReLU -> MaxPool(2)
  -> Flatten -> FC(400->120) -> ReLU
  -> FC(120->84) -> ReLU
  -> FC(84->10)
```

Total parameters: ~61,770 (~241 KB at 4 bytes/param).

---

## Environment setup

```bash
git clone https://github.com/flwrlabs/flower.git
cd flower/baselines/fldsme

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install matplotlib          # for plot_results.py only
```

---

## Running the baseline

```bash
# default config, 20 rounds, 10 clients
flwr run . --stream

# control arm: no MAC-layer constraints
flwr run . --run-config 'dsme-enabled=false' --stream

# override parameters (string values must be quoted)
flwr run . --run-config 'num-server-rounds=10 energy-budget-mj=80.0' --stream

# full ablation, 5 seeds
python run_experiments.py --seeds 0 1 2 3 4 --rounds 20
python plot_results.py --results results --out docs

# feasibility horizon, no training required
python feasibility.py --rounds 20
```

`--stream` is required for scripted runs: without it `flwr run` submits the run
and returns immediately.

---

## Configuration reference

```toml
[tool.flwr.app.config]
num-server-rounds = 20
fraction-train = 0.5
local-epochs = 1
seed = 0

# DSME cluster-tree topology
bo = 6
mo = 5
so = 3
num-clusters = 4

# MAC-layer energy reserve per client (mJ), depletes across rounds
energy-budget-mj = 60.0

# GTS bandwidth constraint
bandwidth-fraction = 0.8

# Control arm: false substitutes NullMACModel (no gate, full bandwidth)
dsme-enabled = true

# Where to write per-round metrics as JSON ("" disables)
results-path = ""
```

---

## File structure

```
fldsme/
|-- fldsme/
|   |-- __init__.py
|   |-- client_app.py       # energy gate + GTS bandwidth mask
|   |-- server_app.py       # DSMEFedAvg, metric aggregation, JSON dump
|   |-- dsme_model.py       # MAC-layer model: PSO Eq.3 + SeCAP bandwidth
|   |-- null_mac_model.py   # control arm: no-op MAC model
|   |-- model.py            # lightweight CNN, device selection
|   |-- dataset.py          # CIFAR-10 loading and partitioning
|   |-- strategy.py         # reserved for future extensions
|   +-- utils.py
|-- docs/                   # figures and summary table
|-- feasibility.py          # feasibility-horizon calculator
|-- run_experiments.py      # multi-seed ablation runner
|-- plot_results.py         # figures with mean ± std bands
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

**Sonali Anand** — sonalianand2406@gmail.com

IEEE 802.15.4e · DSME · Federated Learning · IoT · MAC Optimisation ·
Energy-Aware FL · Flower Baseline