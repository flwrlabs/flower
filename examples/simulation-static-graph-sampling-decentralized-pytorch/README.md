# Simulation + Graph Static + Sampling (PyTorch)

This example demonstrates a **decentralized Flower simulation** in PyTorch with:

- a **static communication graph** (CSR format)
- **neighbor sampling** enabled
- a **NodeApp** that performs local training and evaluation

The goal is to show how to configure a scenario where nodes exchange updates on a fixed topology while a sampling algorithm controls the effective network view over time.

## Prerequisites (recommended for first run)

- Python 3.10+ available in your shell
- Run from this example directory: `examples/simulation-static-graph-sampling-decentralized-pytorch`
- Use a virtual environment (recommended)
- Install this example with `pip install -e .` so `flower-super-dnode` is available

If the script is not executable on your machine, run it with:

- `bash ./scripts/run.sh`

## Install

```bash
cd examples/simulation-static-graph-sampling-decentralized-pytorch
pip install -e .
```

## Run

### Fastest path for beginners

```bash
./scripts/run.sh
```

This is the recommended first run path (single command, no extra arguments).

This starts `flower-super-dnode` in simulation mode with:

- static graph generation (`--network-config-mode csr`)
- sampling enabled (`--enable-sampling`)

using [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml).

## What this example does

When you run [scripts/run.sh](scripts/run.sh), `flower-super-dnode` starts in `simulation` mode with:

- [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml) as the main simulation config
- `--network-config-mode csr` to enable static CSR topology mode
- `--enable-sampling` to enable network sampling
- `--nodeapps-pyproject pyproject.toml` to load the NodeApp declared in [pyproject.toml](pyproject.toml)

The loaded NodeApp is `quickstart_decentralized_pytorch.node_apps:app` (see [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py)).

## PyTorch NodeApp details

The NodeApp implements a simple decentralized FL loop:

- **Model**: a small `TinyNet` MLP (16-dim input, hidden size 32, 2 output classes)
- **Data**: synthetic partitioned datasets generated per `partition-id`
- **Train step**: local SGD + CrossEntropy, then send weights and metrics (`train_loss`, `round`, `num-examples`)
- **Eval step**: compute and return `eval_loss` and `eval_acc`

Default FL run configuration in the NodeApp:

- `rounds=5`
- `n_aggregation_steps=2`
- `local-epochs=1`
- `lr=0.05`

## Static topology (CSR) and sampling

Network behavior is controlled by [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml) and [config_sampling.json](config_sampling.json).

- `nb_nodes: 6` for six simulated nodes
- `topology.kind: ring` as the base topology
- `enable_sampling: true` + `sampling.algorithm: brahams`
- `sampling_period: 1000` ms for periodic sampling updates
- sampling parameters (`view_size`, `sampler_size`, `alpha`, `beta`, `delay`) in YAML and JSON

[config_sampling.json](config_sampling.json) also contains the CSR arrays (`rows`, `cols`) describing static graph connectivity.

## Useful simulation parameters

Key settings in [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml):

- `max_sim_time: 25` and `time_step_ms: 100` for simulation progression
- `base_latency_ms: 35` and `jitter_factor: 0.08` for network latency modeling
- `failure_probability: 0.0` (no injected failures in this scenario)

## Quick customization

- Change `nb_nodes` to scale the simulation
- Tune `sampling.view_size` / `sampler_size` to change connectivity behavior
- Modify `train_config` in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py) (`local-epochs`, `lr`, `batch-size`) to test different learning dynamics

## How to verify it is running correctly

After startup, you should see simulation logs (no immediate shell return) and ongoing activity from simulated nodes.

Good signs:

- no `command not found` error for `flower-super-dnode`
- no config parsing error for [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml)
- no CSR/sampling option error for `--network-config-mode csr` and `--enable-sampling`

## Troubleshooting

- `flower-super-dnode: command not found` → run `pip install -e .` again in this folder
- Script exits with config error → verify [configs/simulation_static_graph_sampling.yaml](configs/simulation_static_graph_sampling.yaml) and [config_sampling.json](config_sampling.json)
- Another local simulation is still active → stop it, then rerun
