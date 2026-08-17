---
title: Discovering Unified Sparse Subnetworks at Initialization for Efficient Federated Learning
url: https://openreview.net/forum?id=kUZ6LhUB26
labels: [sparse federated learning, communication efficiency, non-iid, image classification]
dataset: [CIFAR-10, CIFAR-100]
---
# SSFL: Discovering Unified Sparse Subnetworks at Initialization for Efficient Federated Learning

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [openreview.net/forum?id=kUZ6LhUB26](https://openreview.net/forum?id=kUZ6LhUB26)

**Authors:** Riyasat Ohib, Bishal Thapaliya, Gintare Karolina Dziugaite, Jingyu Liu, Vince D. Calhoun, Sergey Plis

**Abstract:** Federated Learning (FL) enables collaborative model training across distributed clients without sharing raw data. Communication cost remains a bottleneck, especially for overparameterized networks. SSFL discovers a unified sparse subnetwork at initialization from client-local saliency scores, then trains and communicates only the active parameters inside that fixed shared subspace.

## About this baseline

**What's implemented:** Static SSFL on CIFAR-10 and CIFAR-100 with ResNet-18: client saliency discovery, server-side global mask aggregation, and sparse FedAvg in the fixed subspace. Sparse ArrayRecord packing is used after the mask is installed (`transport="sparse"`).

**Datasets:** CIFAR-10, CIFAR-100 (via Flower Datasets)

**Hardware Setup:** Demo and smoke runs work on CPU. Paper-scale runs (100 clients, 999 rounds) were executed on a Linux GPU node with Flower 1.33 simulation. CIFAR-10 took ~4 h; CIFAR-100 ~3.3 h.

**Contributors:** Riyasat Ohib

## Experimental Setup

**Task:** Image classification

**Model:** ResNet-18 (sparse, density `0.5`)

**Dataset:** CIFAR-10 / CIFAR-100 partitioned across 100 clients with a balanced Dirichlet partitioner (`α = 0.3`) matching the original SSFL code.

| Dataset | #classes | #rounds | #partitions | partitioning | density |
| :------ | :------: | :-----: | :---------: | :----------: | :-----: |
| CIFAR-10 | 10 | 999 | 100 | balanced Dirichlet `α=0.3` | 0.5 |
| CIFAR-100 | 100 | 999 | 100 | balanced Dirichlet `α=0.3` | 0.5 |

**Training Hyperparameters (defaults in `pyproject.toml`):**

| Description | Default (demo) |
| ----------- | -------------- |
| total clients | 4 |
| clients per round | 50% |
| number of rounds | 3 |
| local epochs | 1 |
| optimizer | SGD (`lr=0.1`, `wd=5e-4`) |
| transport | sparse |

Paper profiles differ: CIFAR-10 uses round decay `lr * 0.998^round`, momentum 0.0, 5 local epochs, batch 16. CIFAR-100 uses cosine annealing, momentum 0.9, 10 local epochs, batch 128.

## Environment Setup

Python 3.12 is recommended.

```bash
cd baselines/ssfl
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
# optional: pip install -e ".[dev]"
```

## Running the Experiments

```bash
# Demo (4 clients, 3 rounds, CPU)
flwr run .

# Evaluated smoke
flwr run . --run-config conf/cifar10_eval_smoke.toml

# Short 100-client validation (needs GPU federation)
flwr run . gpu-simulation --run-config conf/cifar10_short.toml

# Paper-scale CIFAR-10 / CIFAR-100
flwr run . gpu-simulation --run-config conf/cifar10_paper.toml
flwr run . gpu-simulation --run-config conf/cifar100_paper.toml
```

String overrides need quotes, e.g. `flwr run . --run-config 'dataset="cifar100"'`.

## Expected results

Single-seed Flower runs (`seed=550`) versus the paper 5-run mean. Eval is reported at round 990.

| Dataset   | Config | This baseline (seed 550) | Paper 5-run mean |
|-----------|--------|-------------------------:|-----------------:|
| CIFAR-10  | `conf/cifar10_paper.toml` | 89.56% | ~88.29% |
| CIFAR-100 | `conf/cifar100_paper.toml` | 59.75% | ~61.37% |

These are single-seed results, not a 5-seed sweep. CIFAR-10 landed slightly above the published mean; CIFAR-100 slightly below.

v1 covers static SSFL on CIFAR-10/100 with ResNet-18 at density 0.5. Tiny-ImageNet and random / client-wise masks are out of scope.

## License

This Flower baseline is contributed under Apache-2.0. The original SSFL implementation is MIT: https://github.com/riohib/SSFL
