---
title: Federated Nonconvex Sparse Learning
url: https://arxiv.org/abs/2101.00052
labels: [sparse learning, hard thresholding, non-IID, linear regression, logistic regression]
dataset: [Simulation I, Simulation II, MNIST, E2006-tfidf, RCV1]
---

# FedHT: Federated Nonconvex Sparse Learning

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2101.00052](https://arxiv.org/abs/2101.00052)

**Authors:** Qianqian Tong, Guannan Liang, Tan Zhu, Jinbo Bi (University of Connecticut)

**Abstract:** Nonconvex sparse learning plays an essential role in many areas, such as signal processing and deep network compression. Iterative hard thresholding (IHT) methods are the state-of-the-art for nonconvex sparse learning due to their capability of recovering true support and scalability with large datasets. Theoretical analysis of IHT is currently based on centralized IID data. In realistic large-scale situations, however, data are distributed, hardly IID, and private to local edge computing devices. In this paper, we propose two IHT methods: Federated Hard Thresholding (Fed-HT) and Federated Iterative Hard Thresholding (FedIter-HT). We prove that both algorithms enjoy a linear convergence rate and have strong guarantees to recover the optimal sparse estimator, similar to traditional IHT methods, but now with decentralized non-IID data. Empirical results demonstrate that the Fed-HT and FedIter-HT outperform their competitor, a distributed IHT, in terms of decreasing the objective values with lower requirements on communication rounds and bandwidth.

## About this baseline

**What is implemented:** The code in this directory replicates the simulation experiments in *Federated Nonconvex Sparse Learning* (Tong et al., 2021), which proposed the Fed-HT and FedIter-HT algorithms. Both algorithms extend iterative hard thresholding to federated settings with non-IID data. The baseline replicates the two key synthetic experiments from the paper (Simulation I: sparse linear regression, Simulation II: sparse logistic regression) and supports MNIST for the softmax regression experiment. The Distributed-IHT baseline (K=1) is included for comparison.

**Datasets:** Simulation I (synthetic), Simulation II (synthetic), MNIST

**Hardware Setup:** These experiments were run on a MacBook with Apple Silicon (ARM). Any machine with 4 CPU cores should reproduce results in reasonable time. The simulation experiments use 100 clients with 100 samples each. MNIST uses 100 clients with 600 samples each.

**Contributors:** Harshal Manerikar

## Experimental Setup

**Task:** Sparse parameter estimation under a cardinality constraint

**Model:** Linear models with no hidden layers. The cardinality constraint is enforced externally by the strategy via hard thresholding rather than inside the model.

| Experiment | Model | Loss |
| :--- | :--- | :--- |
| Simulation I | Sparse linear regression | Mean squared error |
| Simulation II | Sparse logistic regression | Binary cross-entropy |
| MNIST | Sparse softmax regression | Cross-entropy |

**Algorithms:**

Both algorithms follow the same outer loop structure (T communication rounds, N clients, K local SGD steps each). The difference is where the hard thresholding operator H_tau is applied:

| Algorithm | Local update | Server aggregation |
| :--- | :--- | :--- |
| Fed-HT | Plain SGD, no thresholding | H_tau applied after weighted average |
| FedIter-HT | H_tau applied after each SGD step | H_tau applied after weighted average |
| Distributed-IHT (baseline) | K=1, communicates every step | H_tau applied after weighted average |

**Dataset partitioning:**

| Experiment | Clients | Samples per client | Partition method |
| :--- | :---: | :---: | :--- |
| Simulation I | 100 | 100 | Synthetic generation with alpha=0.1, beta=0.1 |
| Simulation II | 100 | 1000 (binary thresholded) | Synthetic generation with alpha=1.0, beta=1.0 |
| MNIST | 100 | 600 | Pathological (each client holds 2 of 10 digit classes) |

**Training hyperparameters (defaults):**

| Description | Default value |
| :--- | :--- |
| Total clients | 100 |
| Fraction sampled per round | 1.0 (all clients) |
| Number of rounds | 100 |
| Local steps K | 5 |
| Learning rate | 0.001 |
| Sparsity tau | 200 (simulations), 500 (MNIST) |
| Initialization | Zero (x_0 = 0 as in the paper) |
| Client resources | 2 CPUs, 0 GPUs |

**Hyperparameter search ranges (from the paper):**

The paper uses grid search to select the best K and learning rate per experiment. The search ranges are:

| Parameter | Search range |
| :--- | :--- |
| K (local steps) | {3, 5, 8, 10} |
| Learning rate | {10, 1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01, 0.001} |

## Environment Setup

The Flower venv must be placed **outside** the project directory. PyTorch contains files nested more than 10 directories deep, and `flwr run .` will reject the project if it finds such paths inside the project tree.

```bash
# Create a Python 3.12 environment outside the project
# On macOS with Homebrew Python you may need the DYLD fix below
DYLD_LIBRARY_PATH=$(brew --prefix expat)/lib python3.12 -m venv ~/fedht-venv

# Activate
source ~/fedht-venv/bin/activate

# Install dependencies
pip install -e /path/to/baselines/fedht
```

**macOS note:** Homebrew Python 3.12 may fail with a `pyexpat` symbol error on older macOS versions. Running with `DYLD_LIBRARY_PATH=$(brew --prefix expat)/lib` before any Python or `flwr` command resolves this. Install Homebrew expat first with `brew install expat`.

**Federation setup:** These experiments simulate 100 clients. The `local-simulation` federation must be configured with `num-supernodes = 100`. Add the following to `~/.flwr/config.toml` (create the file if it does not exist):

```toml
[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 100
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0
```

## Running the Experiments

Make sure the venv is active, then from inside the `fedht` directory:

```bash
# Simulation I: sparse linear regression with Fed-HT (default)
flwr run . local-simulation

# Switch to FedIter-HT
flwr run . local-simulation --run-config 'algorithm.name="FedIterHT"'

# Simulation II: sparse logistic regression
flwr run . local-simulation --run-config 'dataset.name="simulation_II" model.input-dim=1000 dataset.batch-size=20'

# MNIST: sparse softmax regression
flwr run . local-simulation --run-config 'dataset.name="mnist" dataset.batch-size=64 algorithm.tau=500 model.input-dim=784 model.num-classes=10'

# Override K and learning rate for grid search
flwr run . local-simulation --run-config "algorithm.local-steps=10 algorithm.learning-rate=0.01"

# Reduce rounds for a quick smoke test (client count is set by the federation num-supernodes)
flwr run . local-simulation --run-config "algorithm.num-server-rounds=10"
```

## Expected Results

The key result from the paper is that Fed-HT and FedIter-HT reach the same objective value as Distributed-IHT using significantly fewer communication rounds:

| Experiment | Algorithm | Rounds to match Distributed-IHT |
| :--- | :--- | :--- |
| Simulation I (linear) | Fed-HT (K=3, lr=0.003) | ~28 rounds vs 100 for Distributed-IHT (3.5x fewer) |
| Simulation I (linear) | FedIter-HT (K=3) | TBD |
| Simulation II (logistic) | All algorithms | TBD |

Plots of objective value vs communication rounds for each experiment will be added to `_static/` after grid search is complete. See `docs/EXTENDED_README.md` for detailed per-experiment results and plots.
