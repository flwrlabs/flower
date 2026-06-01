# Simulation + Graph Dynamic (PyTorch)

This example demonstrates a **decentralized Flower simulation** in PyTorch using:

- a **dynamic random topology**
- **sampling enabled** (GBPS)
- a PyTorch **NodeApp** that trains and evaluates locally on synthetic partitioned data

## Prerequisites (recommended for first run)

- Python 3.10+ available in your shell
- Run from this example directory: `examples/simulation-dynamic-graph-decentralized-pytorch`
- Use a virtual environment (recommended)
- Install this example with `pip install -e .` so `flower-super-dnode` is available

If the script is not executable on your machine, run it with:

- `bash ./scripts/run.sh`

## Install

```bash
cd examples/simulation-dynamic-graph-decentralized-pytorch
pip install -e .
```

## Run

### Fastest path for beginners

```bash
./scripts/run.sh
```

This command is the recommended first run path (single command, no extra arguments).

This starts `flower-super-dnode` in simulation mode using a dynamic random graph
with sampling enabled, from
[configs/simulation_dynamic_graph.yaml](configs/simulation_dynamic_graph.yaml).

## What this example does

When you run [scripts/run.sh](scripts/run.sh), `flower-super-dnode` starts in `simulation` mode with:

- [configs/simulation_dynamic_graph.yaml](configs/simulation_dynamic_graph.yaml) as the simulation config
- dynamic network behavior configured via `network.topology.kind: random`
- NodeApp loading from `--nodeapps-pyproject pyproject.toml`

The NodeApp is `quickstart_decentralized_pytorch.node_apps:app` (see [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py)).

## PyTorch NodeApp details

The NodeApp implements a lightweight decentralized FL loop:

- **Model**: `TinyNet` MLP (16-dim input → 32 hidden units → 2 classes)
- **Data**: synthetic data generated per partition (`partition-id`)
- **Train**: local SGD + CrossEntropy, returns model arrays and metrics (`train_loss`, `round`, `num-examples`)
- **Evaluate**: local `eval_loss` and `eval_acc`

Default run configuration:

- `rounds=5`
- `n_aggregation_steps=2`
- `local-epochs=1`
- `lr=0.05`

## Dynamic graph + sampling configuration

The main network-related settings in [configs/simulation_dynamic_graph.yaml](configs/simulation_dynamic_graph.yaml) are:

- `nb_nodes: 4`
- `topology.kind: random` with `random.mode: exact`, `send_to: 2`, `receive_from: 2`
- `enable_sampling: true`
- `sampling.algorithm: gbps`
- `sampling_period: 1000` ms

GBPS sampling parameters include `view_size`, `heal`, `swap`, `selection_policy`, `propagation_policy`, `delay`, and `age`.

## Useful simulation parameters

- `max_sim_time: 20` and `time_step_ms: 100`
- `base_latency_ms: 30` and `jitter_factor: 0.05`
- `failure_probability: 0.0` (no disconnections injected)

## Quick customization

- Increase `nb_nodes` to scale up the simulation
- Modify `send_to`/`receive_from` to change dynamic graph density
- Tune GBPS sampling parameters in the `sampling` section
- Update `train_config` in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py) to test different local optimization settings

## How to verify it is running correctly

After startup, you should see simulation logs (no immediate shell return) and periodic activity from simulated nodes.

Good signs:

- no `command not found` error for `flower-super-dnode`
- no config parsing error for [configs/simulation_dynamic_graph.yaml](configs/simulation_dynamic_graph.yaml)
- logs continue until simulation completion (`max_sim_time`) or manual stop

## Troubleshooting

- `flower-super-dnode: command not found` → run `pip install -e .` again in this folder
- Script exits immediately with config error → verify path and YAML formatting in [configs/simulation_dynamic_graph.yaml](configs/simulation_dynamic_graph.yaml)
- Port/network-related errors from another local run → stop other simulations before retrying
