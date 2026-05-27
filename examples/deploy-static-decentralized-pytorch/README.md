# Deploy + Topology Static (PyTorch)

This example shows how to run Flower in **deploy mode** with a **static topology** using a PyTorch NodeApp.

- In single-node mode, each process is started manually with a node name and port.
- In cluster mode, a helper script generates a static ring topology and launches all nodes locally.

## Prerequisites (recommended for first run)

- Python 3.10+ available in your shell
- Run from this example directory: `examples/deploy-static-decentralized-pytorch`
- Use a virtual environment (recommended)
- Install this example with `pip install -e .` so `flower-super-dnode` is available

If scripts are not executable on your machine, run them with `bash`:

- `bash ./scripts/run.sh ...`
- `bash ./scripts/run_cluster.sh ...`

## Install

```bash
cd examples/deploy-static-decentralized-pytorch
pip install -e .
```

## Run

### Fastest path for beginners

Use the cluster script first (single command):

```bash
./scripts/run_cluster.sh
```

This starts 4 local nodes (`node_1`..`node_4`) on ports `9200..9203`.

### Single node (manual multi-terminal)

```bash
# terminal 1
./scripts/run.sh node_1 9201
# terminal 2
./scripts/run.sh node_2 9202
# terminal 3
./scripts/run.sh node_3 9203
# terminal 4
./scripts/run.sh node_4 9204
```

Starts `flower-super-dnode` with static ring topology
([configs/deploy_static.yaml](configs/deploy_static.yaml)).
Node names must match entries in the generated topology (`node_1`..`node_4`).

Use this manual mode only if you want full control over each process.

In this mode, `scripts/run.sh` starts one process with:

- `--execution-mode deploy`
- `--config configs/deploy_static.yaml`
- `--node-name <node_x>` and `--port <port>`
- `--node-data-config-json` for partition metadata

### Multi-node cluster (N nodes on localhost, one command)

```bash
./scripts/run_cluster.sh                  # 4 nodes, ports 9200-9203
./scripts/run_cluster.sh 6                # 6 nodes, ports 9200-9205
./scripts/run_cluster.sh 8 9300           # 8 nodes, ports 9300-9307
```

What the cluster script does:
1. Generates a ring topology YAML for N nodes (`node_1`..`node_N`) via
	 `flwr.decentralized.common.graph.api.generate_deploy_topology_yaml`.
2. Starts all N nodes in parallel, each on `BASE_PORT + (i-1)`.
- `Ctrl-C` stops the entire cluster.
- Per-node logs are written to a temp directory and tailed to stdout.
- Uses [configs/deploy_static_cluster.yaml](configs/deploy_static_cluster.yaml) which
	loads the pre-generated topology directly (no race-condition with `generate` at runtime).

## How to verify it is running correctly

After startup, you should see lines similar to:

- `Starting static cluster: ...`
- `Cluster running (...)`
- per-node log output from `node_1.log`, `node_2.log`, etc.

If one process exits, the script returns and prints the log directory path.

## Topology and runtime configuration

[configs/deploy_static.yaml](configs/deploy_static.yaml) uses:

- `topology.mode: static`
- `topology.generate.kind: ring`
- `topology.generate.node_count: 4` (default)

For cluster mode, [configs/deploy_static_cluster.yaml](configs/deploy_static_cluster.yaml) uses:

- `topology.mode: static`
- `topology.file: ./configs/generated_static_topology.yaml`

This means the topology is generated once by the script, then reused by all launched nodes.

## NodeApp behavior

The NodeApp loaded from `pyproject.toml` (`quickstart_decentralized_pytorch.node_apps:app`) performs:

- local training on synthetic partitioned data
- local evaluation (`eval_loss`, `eval_acc`)
- periodic decentralized model exchange using the configured topology

Default training/evaluation behavior is implemented in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py).

## Quick customization

- Change ring size with `./scripts/run_cluster.sh <N> [BASE_PORT]`
- Update topology kind in the generation step (if you want a different static graph)
- Tune train settings (`local-epochs`, `lr`, `batch-size`) in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py)

## Troubleshooting

- `flower-super-dnode: command not found` → run `pip install -e .` again in this folder
- `Address already in use` → change base port (for example `./scripts/run_cluster.sh 4 9300`)
- No logs appear → ensure you started from this example directory
- Immediate crash after start → check the printed temp log directory and inspect per-node logs
