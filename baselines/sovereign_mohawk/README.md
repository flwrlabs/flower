---
title: Sovereign Mohawk Inspired Verifiable Aggregation Starter
url: https://github.com/rwilliamspbg-ops/Sovereign-Mohawk-Proto
labels: [verifiable-aggregation, byzantine-resilience, simulation, cifar10]
dataset: [CIFAR10]
---

<!-- markdownlint-disable MD025 -->

Sovereign Mohawk Inspired Verifiable Aggregation Starter
========================================================

> Note: If you use this baseline in your work, please also cite the Flower paper.

**Paper/Reference:** [Sovereign-Mohawk-Proto](https://github.com/rwilliamspbg-ops/Sovereign-Mohawk-Proto)

**Authors:** Ryan Williams (reference project), Flower community baseline adaptation by @rwilliamspbg-ops

**Abstract:** This baseline starter introduces a reproducible Flower simulation scaffold designed for future experimentation with verifiable aggregation ideas. The current implementation is intentionally minimal and runs a standard federated training loop on CIFAR10. It includes explicit extension points to add proof or verification hooks around model update aggregation in follow-up commits.

About This Baseline
-------------------

**What’s implemented:** A runnable baseline starter with ServerApp and ClientApp components, CIFAR10 partition loading, and FedAvg orchestration.

**Datasets:** CIFAR10 via Flower Datasets.

**Hardware Setup:** CPU-only execution supported for starter runs.

**Contributors:** @rwilliamspbg-ops

Experimental Setup
------------------

**Task:** Image classification

**Model:** Small CNN adapted from Flower's baseline template.

**Dataset:** IID partitioning of CIFAR10 with per-client 80/20 local train/test split.

**Training Hyperparameters:**

| Description | Default Value |
| --- | --- |
| num-server-rounds | 3 |
| fraction-train | 0.5 |
| local-epochs | 1 |
| num-supernodes | 10 |

Environment Setup
-----------------

```bash
# Create and activate env
pyenv virtualenv 3.12.12 sovereign-mohawk
pyenv activate sovereign-mohawk

# Install baseline (from this directory)
pip install -e ".[dev]"
```

Running The Experiments
-----------------------

From this baseline directory:

```bash
# Run default starter simulation
flwr run .

# Override rounds and local epochs
flwr run . --run-config num-server-rounds=5,local-epochs=2
```

Smoke Test
----------

From the `baselines/` directory, run:

```bash
./dev/format-baseline.sh sovereign_mohawk
./dev/test-baseline.sh sovereign_mohawk
```

Then run a short baseline simulation from `baselines/sovereign_mohawk/`:

```bash
flwr run . --run-config num-server-rounds=1,fraction-train=0.5,local-epochs=1,enable-verification-hooks=true
```

Expected Results (Smoke Run)
----------------------------

| Check | Expected Signal |
| --- | --- |
| Python lint/type checks | `All Python checks passed` in `./dev/test-baseline.sh` output |
| Simulation completion | Run exits without exception |
| Verification hook status | JSON log contains `"status": "passed"` |
| Model artifact | `final_model.pt` is created |
| Verification artifact | `verification_report.json` is created |

Next Implementation Steps
-------------------------

1. Extend verification checks from finite-value validation to configurable policy checks.
2. Add experiment config variants and expected result tables aligned with the final issue scope.
3. Add plotting/report utilities to persist comparable metrics across repeated runs.
