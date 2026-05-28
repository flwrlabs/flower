# Simulation PyTorch Example

This example runs a **decentralized discrete-event simulation** with `flower-super-dnode`
and a simple PyTorch model (`TinyNet`) trained on deterministic synthetic partitions.
It includes an explicit sampling/network configuration file: `config_sampling.json`.

## Install

```bash
cd examples/simulation-pytorch
pip install -e .
```

## Run (recommended)

```bash
python -m simulation_pytorch.run --nb-nodes 4 --max-sim-time 25 --verbose-sim
```

Or, if the script entrypoint is available:

```bash
run-simulation-pytorch --nb-nodes 4 --max-sim-time 25 --verbose-sim
```

## File-based config (YAML/TOML)

You can provide a simulation config file (same spirit as deploy), and keep CLI as overrides:

```bash
python -m simulation_pytorch.run --sim-config /path/to/simulation.yaml
```

When `network.enable_sampling=true`, a sampling configuration is generated.
When `network.enable_sampling=false`, a CSR topology configuration is generated.

Example `simulation.yaml`:

```yaml
simulation:
  nb_nodes: 4
  timeout: 120
  max_sim_time: 25
  time_step_ms: 100
  real_time_factor: 1.0
  multi_thread: false
  verbose: false

network:
  enable_sampling: true
  sampling_period: 1000
  sampling:
    config_file: /tmp/config_sampling.json
    algorithm: gbps   # gbps | brahams | basalt
    view_size: 4
    heal: 0
    swap: 0
    selection_policy: old
    propagation_policy: pushpull
    delay: 2
    age: 1
    sampler_size: 8   # brahams only
    alpha: 0.5        # brahams only
    beta: 0.5         # brahams only
    refresh: 1        # basalt only
  topology:
    kind: random
    seed: 42
    random:
      mode: exact   # exact | range
      send_to: 2
      receive_from: 2

latency:
  base_latency_ms: 30
  jitter_factor: 0.05

disconnection:
  failure_probability: 0.0
  recovery_time: 10

synchronization:
  sync_node_count: 0
  sync_interval_ms: 500
  max_drift_ms: 0
```

The launcher forwards extra simulation CLI flags, for example:

```bash
python -m simulation_pytorch.run \
  --nb-nodes 8 \
  --no-enable-sampling \
  --topology-kind ring \
  --topology-seed 42
```

Random topology with exact random degrees:

```bash
python -m simulation_pytorch.run \
  --no-enable-sampling \
  --topology-kind random \
  --random-mode exact \
  --random-send-to 2 \
  --random-receive-from 2
```

Random topology with range constraints:

```bash
python -m simulation_pytorch.run \
  --no-enable-sampling \
  --topology-kind random \
  --random-mode range \
  --random-min-send-to 1 \
  --random-max-send-to 3 \
  --random-min-receive-from 1 \
  --random-max-receive-from 3

Sampling examples:

```bash
python -m simulation_pytorch.run \
  --enable-sampling \
  --sampling-algorithm gbps

python -m simulation_pytorch.run \
  --enable-sampling \
  --sampling-algorithm brahams \
  --sampling-sampler-size 8 \
  --sampling-alpha 0.5 \
  --sampling-beta 0.5

python -m simulation_pytorch.run \
  --enable-sampling \
  --sampling-algorithm basalt \
  --sampling-refresh 1
```

## Run (direct CLI)

```bash
flower-super-dnode \
  --execution-mode simulation \
  --nodeapps-pyproject pyproject.toml \
  --nb-nodes 4 \
  --max-sim-time 25 \
  --sim-timeout 120 \
  --base-latency-ms 30 \
  --jitter-factor 0.05 \
  --verbose-sim
```

## What this example does

- Declares one `NodeApp` subject: `trainer-pytorch-sim`
- Creates one virtual app instance per node via simulation mode
- Trains a tiny MLP for binary classification locally on each partition
- Returns `arrays` and local metrics during train, and evaluation metrics during evaluate

## Main files

- `simulation_pytorch/node_apps.py`: model, local train/eval, NodeApp
- `simulation_pytorch/run.py`: convenience runner for simulation mode
- `pyproject.toml`: dependencies and NodeApp autoload mapping
- `config_sampling.json`: auto-generated/updated peer-sampling or CSR network config
