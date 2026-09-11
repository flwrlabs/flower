---
title: "Verifiable Aggregation Workflow"
url: https://github.com/rwilliamspbg-ops/Sovereign-Mohawk-Proto
labels: [verification, aggregation, reproducibility, message-api, synthetic-data]
dataset: [synthetic]
---

## Verifiable Aggregation Workflow

> Note: If you use this baseline in your work, please cite Flower and any upstream work that inspired your implementation.

**Paper/Reference:** [Sovereign-Mohawk-Proto](https://github.com/rwilliamspbg-ops/Sovereign-Mohawk-Proto)

**Authors:** Community contribution by rwilliamspbg-ops

**Abstract:** This baseline demonstrates a reproducible federated learning workflow in Flower where standard FedAvg aggregation is augmented with optional server-side verification hooks. At each round, the server recomputes the weighted aggregate from raw client updates, compares it to the strategy output under a configurable tolerance, and records deterministic hashes and verification outcomes in a JSON report.

## About this baseline

**What is implemented:** A minimal Message API Flower baseline with deterministic synthetic data, optional verification checks around aggregation outputs, and benchmark-friendly reporting scripts.

**Datasets:** Fully deterministic synthetic binary classification data generated per client partition.

**Hardware Setup:** CPU-only runs are supported. Default configuration (8 clients, 5 rounds) typically finishes in under a minute on a laptop-class CPU.

**Contributors:** rwilliamspbg-ops, Flower community maintainers

## Experimental Setup

**Task:** Binary classification.

**Model:** Small MLP with two linear layers and one ReLU.

**Dataset:**

| Property | Value |
| --- | --- |
| Source | Generated on the fly (no downloads) |
| Features | 10 float features |
| Labels | Binary (0/1) |
| Clients | 8 by default |
| Local train examples/client | 128 |
| Local val examples/client | 64 |
| Partitioning | Deterministic client-specific distribution shift |

**Training Hyperparameters (default):**

| Hyperparameter | Value |
| --- | --- |
| num-server-rounds | 5 |
| fraction-train | 1.0 |
| fraction-evaluate | 1.0 |
| local-epochs | 1 |
| learning-rate | 0.05 |
| batch-size | 32 |
| random-seed | 2026 |
| verify-aggregation | true |
| verification-tolerance | 1e-6 |

The number of simulated clients is controlled by the federation setting
`options.num-supernodes` (default: 8), not by `--run-config`.

## Environment Setup

```bash
# Create the virtual environment
pyenv virtualenv 3.12.12 verifiableagg

# Activate it
pyenv activate verifiableagg

# Install baseline
pip install -e .

# If you are contributing changes and want to run lint/type checks
pip install -e ".[dev]"
```

For contributor checks used in Flower baselines CI:

```bash
cd ..
./dev/test-baseline-structure.sh verifiableagg
./dev/test-baseline.sh verifiableagg
```

## Running the Experiments

```bash
# Run with defaults from pyproject.toml
flwr run .

# Override selected values from the CLI
flwr run . --run-config "num-server-rounds=10 verify-aggregation=true random-seed=2026"

# Run benchmark helper script (train + report check)
bash run_benchmark.sh
```

## Verification Outputs and Reproducibility

After each run, artifacts are written to the directory set by artifacts-dir (default: artifacts):

- artifacts/final_model.pt
- artifacts/report.json

The report includes:

- Effective run configuration
- Per-round aggregated train/eval metrics
- Per-round verification status (pass/fail)
- Maximum absolute replay difference
- Deterministic SHA256 hash of aggregated parameters per round

To summarize and validate verification outcomes:

```bash
python benchmark_report.py --report-path artifacts/report.json
```

This command exits non-zero if any round fails verification, which makes it suitable for CI or benchmark automation.

## Expected Results

With default settings, all rounds should pass verification with very small numerical replay error (typically near machine precision). Example benchmark output:

```text
round   num_replies      max_abs_diff    passed
1       8                0.00000000e+00  1
2       8                0.00000000e+00  1
3       8                0.00000000e+00  1
4       8                0.00000000e+00  1
5       8                0.00000000e+00  1
All rounds verified. Max observed absolute difference: 0.00000000e+00
```
