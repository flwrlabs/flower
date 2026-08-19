---
title: Discovering Unified Sparse Subnetworks at Initialization for Efficient Federated Learning
url: https://openreview.net/forum?id=kUZ6LhUB26
labels: [sparse federated learning, communication efficiency, non-iid, image classification]
dataset: [CIFAR-10, CIFAR-100]
---

# SSFL: Discovering Unified Sparse Subnetworks at Initialization for Efficient Federated Learning

> If you use this baseline, please cite both the original SSFL paper and Flower.

**Paper:** [Transactions on Machine Learning Research / OpenReview](https://openreview.net/forum?id=kUZ6LhUB26)

**Reference implementation:** [riohib/SSFL](https://github.com/riohib/SSFL)

**Authors:** Riyasat Ohib, Bishal Thapaliya, Gintare Karolina Dziugaite, Jingyu Liu, Vince D. Calhoun, Sergey Plis

## About this baseline

SSFL (Salient Sparse Federated Learning) discovers one unified sparse
subnetwork before federated training. Each client computes parameter saliency
scores from a local minibatch, the server averages those scores and creates a
global mask, and all clients subsequently train in that fixed shared
subspace. The paper provides early evidence that a unified subspace can offer
a better accuracy-sparsity trade-off than methods that repeatedly change the
sparse topology during training.

This Flower baseline implements the complete static SSFL pipeline:

1. all clients compute `abs(gradient * weight)` saliency scores;
2. the ServerApp averages the scores and creates a global mask;
3. the mask is installed on every ClientApp;
4. sampled clients perform masked local SGD and sample-weighted FedAvg;
5. only active parameters are packed into Flower `ArrayRecord` payloads after
   discovery when `transport="sparse"`.

The baseline uses the Flower Message API, Flower Datasets, PyTorch, and a
balanced Dirichlet partitioner matching the original SSFL implementation.

**Implemented experiments:** static SSFL with ResNet-18 on CIFAR-10 and
CIFAR-100 at density `0.5`.

**Contributors:** Riyasat Ohib

## Experimental setup

Both paper profiles use 100 simulated clients, 10% participation per round,
balanced Dirichlet partitioning with `alpha=0.3`, seed 550, and 999 training
rounds. The original runner configured 1000 communication rounds but trained
over `range(1, 1000)`, so this baseline uses 999 rounds for parity.

| Setting | CIFAR-10 | CIFAR-100 |
| --- | ---: | ---: |
| Model | ResNet-18 | ResNet-18 |
| Number of classes | 10 | 100 |
| Global density | 0.5 | 0.5 |
| Total clients | 100 | 100 |
| Clients per round | 10 | 10 |
| Training rounds | 999 | 999 |
| Local epochs | 5 | 10 |
| Batch size | 16 | 128 |
| Optimizer | SGD | SGD |
| Initial learning rate | 0.1 | 0.1 |
| Scheduler | `0.1 * 0.998^round` | cosine warm restarts (`T_0=10`, `eta_min=1e-5`) |
| Momentum | 0.0 | 0.9 |
| Weight decay | 0.0005 | 0.0005 |
| Gradient clipping | 10.0 | 10.0 |
| Evaluation interval | 10 rounds | 10 rounds |

The global mask is discovered from one saliency minibatch per client and uses
uniform score averaging. Balanced partitions contain equal numbers of
examples, so uniform and sample-weighted saliency averages coincide here.

## Environment setup

Python 3.12 is recommended.

```bash
git clone https://github.com/flwrlabs/flower.git
cd flower/baselines/ssfl

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If your Python installation does not provide `venv`, use
[uv](https://docs.astral.sh/uv/) instead:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

No dataset download step is required. Flower Datasets downloads CIFAR from
Hugging Face on first use.

## Running the Flower baseline

The portable launcher configures Flower simulation resources on each command.
It contains no Slurm or site-specific cluster settings.

### Quick validation

The two-client demo and four-client evaluated smoke cap each partition at 512
examples for a quick functional check. Paper profiles use every example.

```bash
# Simplest Flower invocation (uses the local Simulation Runtime)
flwr run . --stream

# Two-client, three-round CPU demo
./run_experiments.sh demo

# Four-client, three-round CPU run with centralized evaluation
./run_experiments.sh eval-smoke

# Short 100-client, three-round GPU validation
./run_experiments.sh short
```

### Paper-scale CIFAR-10

```bash
./run_experiments.sh cifar10
```

Equivalent Flower command:

```bash
flwr run . --stream \
  --federation-config \
  "num-supernodes=100 client-resources-num-cpus=1 client-resources-num-gpus=0.125" \
  --run-config conf/cifar10_paper.toml
```

### Paper-scale CIFAR-100

```bash
./run_experiments.sh cifar100
```

Equivalent Flower command:

```bash
flwr run . --stream \
  --federation-config \
  "num-supernodes=100 client-resources-num-cpus=1 client-resources-num-gpus=0.125" \
  --run-config conf/cifar100_paper.toml
```

The default GPU allocation is `0.125` GPU per concurrent ClientApp (up to
eight concurrent ClientApps per GPU). Adjust it for available GPU memory:

```bash
SSFL_CLIENT_GPUS=0.25 ./run_experiments.sh cifar100
```

Increasing the fraction reduces concurrency and GPU memory pressure. The 100
clients are simulated sequentially in batches; 100 physical GPUs are not
required.

## Outputs and experiment tracking

Paper runs write:

- `outputs/<profile>/summary.json`: final accuracy/loss, resolved config, mask
  digest, and communication totals;
- `outputs/<profile>/metrics.jsonl`: per-evaluation and per-training metrics
  for the current run (replaced if the same `checkpoint-dir` is reused);
- `outputs/<profile>/*.pt`: initial, periodic, and final checkpoints;
- `wandb/`: offline W&B runs when `wandb-mode` is `offline` or `online`.

The paper profiles use W&B offline mode, so no account is required. Local
`summary.json` and `metrics.jsonl` are always written. To log to your own
W&B account:

```bash
flwr run . --stream \
  --federation-config \
  "num-supernodes=100 client-resources-num-cpus=1 client-resources-num-gpus=0.125" \
  --run-config conf/cifar100_paper.toml \
  --run-config 'wandb-mode="online" wandb-entity="<your-entity>"'
```

Communication values are Flower `ArrayRecord` payload bytes; they exclude
Flower Message framing and network-layer overhead.

## Expected results

The following are single-seed Flower runs (`seed=550`) compared with the
paper's five-run means. Because evaluation runs every ten rounds, the final
reported evaluation is at round 990.

| Dataset | Profile | This baseline (seed 550) | Paper 5-run mean |
| --- | --- | ---: | ---: |
| CIFAR-10 | `conf/cifar10_paper.toml` | 89.56% | ~88.29% |
| CIFAR-100 | `conf/cifar100_paper.toml` | 59.75% | ~61.37% |

These reference runs were completed with one CUDA GPU. Runtime depends on GPU,
CPU concurrency, storage, and download speed; observed runtimes were
approximately four hours for CIFAR-10 and 3.3 hours for CIFAR-100.

## Scope

This contribution covers static SSFL, CIFAR-10/100, ResNet-18, and density
0.5. Tiny-ImageNet, dynamic/random masks, and client-specific sparse
topologies are not included.

## Citation

```bibtex
@article{
ohib2026ssfl,
title={{SSFL}: Discovering Sparse Unified Subnetworks at Initialization for Efficient Federated Learning},
author={Riyasat Ohib and Bishal Thapaliya and Gintare Karolina Dziugaite and Jingyu Liu and Vince D. Calhoun and Sergey Plis},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=kUZ6LhUB26}
}
```

## License

This Flower baseline is contributed under Apache-2.0. The original SSFL
implementation is MIT licensed.
