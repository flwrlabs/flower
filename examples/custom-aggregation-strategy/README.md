---
tags: [strategy, custom aggregation, robustness, model poisoning]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Custom Aggregation: Trust-Weighted FedAvg (Flower / PyTorch)

This example shows **how to implement a custom aggregation strategy** on Flower's
message-based API by subclassing `FedAvg` and overriding a single method,
`aggregate_train`. It is meant as a minimal, readable template for anyone who wants
to change *how client updates are combined* (rather than just adding logging or
checkpointing on top of FedAvg).

As a concrete, testable motivation, the custom strategy implements a simple
**norm-based robustness filter**: client updates whose L2 norm is far above the
median are treated as outliers (e.g. poisoned models) and down-weighted before
averaging. The example ships with an optional attack so you can see the difference
between vanilla FedAvg and the trust-aware variant under model poisoning.

> This is a *pedagogical* example of writing custom aggregation. Flower already
> ships production robust strategies (`Krum`, `MultiKrum`, `Bulyan`, `FedMedian`,
> `FedTrimmedAvg`) in `flwr.serverapp.strategy` — reach for those for real use.

## The idea in one method

`TrustAwareFedAvg` (see [`customagg/strategy.py`](customagg/strategy.py)) overrides
only `aggregate_train`:

1. Compute each client's update norm `‖client_arrays − global_arrays‖₂`.
2. Score how much of an outlier it is using robust statistics — a `median + MAD`
   z-score. Updates within a `trust-z` dead zone of the median are fully trusted
   (`1.0`); only updates beyond it are down-weighted, decaying as `exp(-beta · excess)`.
   Using the MAD (median absolute deviation) keeps honest clients near the median
   from being penalized just for being slightly above it.
3. Multiply that trust score into the existing `num-examples` weighting key, then
   reuse `FedAvg`'s weighted aggregation unchanged.

> Assumes an honest majority (the median/MAD must reflect benign clients). This is
> the standard assumption behind robust-aggregation defenses.

**Design note.** `aggregate_train` only receives the client replies, *not* the
current global model — so the strategy stashes the global `ArrayRecord` in
`configure_train` (`self._global = arrays`) to compute the true *update* norm. This
is the one non-obvious part of writing norm/distance-based aggregation on this API.

## Install

```bash
pip install -e .
```

## Run

By default this runs `TrustAwareFedAvg` with 3 (of 10) clients poisoning their
updates:

```bash
flwr run . --stream
```

Reproduce the three-way comparison below by overriding the config:

```bash
# (1) Vanilla FedAvg, no attackers — clean baseline
flwr run . --run-config "strategy='fedavg' num-malicious=0" --stream

# (2) Vanilla FedAvg, 3 attackers — training collapses
flwr run . --run-config "strategy='fedavg' num-malicious=3" --stream

# (3) Trust-aware FedAvg, 3 attackers — recovers
flwr run . --run-config "strategy='trust' num-malicious=3" --stream
```

## Results

CIFAR-10, 10 clients, 5 rounds, 1 local epoch (final global accuracy on the
server-side test set):

| Setting                         | Attackers | Final accuracy |
| ------------------------------- | :-------: | :------------: |
| FedAvg                          |     0     |     20.2 %     |
| FedAvg                          |     3     |  10.1 % 💀     |
| **TrustAwareFedAvg**            |     3     |  **20.8 %** ✅ |

With 3 poisoning clients, vanilla FedAvg collapses to random-guess accuracy, while
the trust-aware variant recovers to the clean baseline. The strategy logs which
clients it down-weighted each round, e.g.:

```
TrustAware: median_norm=8.054 | down-weighted 3/10 clients
            #2(norm=250.1,trust=0.00) #6(norm=249.6,trust=0.00) #8(norm=248.7,trust=0.00)
```

(Honest update norms ≈ 8, poisoned ≈ 250 — cleanly separable.)

## Configuration

| Key               | Default | Meaning                                              |
| ----------------- | :-----: | ---------------------------------------------------- |
| `strategy`        | `trust` | `trust` (TrustAwareFedAvg) or `fedavg` (vanilla)     |
| `num-malicious`   | `3`     | Number of clients that poison their update (0 = off) |
| `attack-scale`    | `1.0`   | Std of Gaussian noise added by malicious clients     |
| `trust-z`         | `3.5`   | Dead-zone width (robust-std/MAD units); inside it, trust = 1.0 |
| `trust-beta`      | `5.0`   | Decay rate of trust past the dead zone               |
| `num-server-rounds` | `5`   | Federated rounds                                     |

## Running on GPU

The dependencies install the CPU build of PyTorch by default. To use a GPU,
install a CUDA build of `torch`/`torchvision`, then run with the `gpu-simulation`
federation defined in `pyproject.toml` (which allocates 0.1 GPU per client):

```bash
flwr run . gpu-simulation --stream
```
