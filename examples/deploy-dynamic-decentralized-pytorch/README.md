# Deploy + Topology Dynamic (PyTorch)

This example runs Flower in **deploy mode** with a **dynamic topology** and a PyTorch NodeApp.

- Nodes are started as independent `flower-super-dnode` processes.
- Topology is dynamic (`topology.mode: dynamic`), so peer relationships are not fixed in a static file.
- Sampling is configured with the GBPS algorithm.

## Prerequisites (recommended for first run)

- Python 3.10+ available in your shell
- Run from this example directory: `examples/deploy-dynamic-decentralized-pytorch`
- Use a virtual environment (recommended)
- Install this example with `pip install -e .` so `flower-super-dnode` is available

If scripts are not executable on your machine, run them with `bash`:

- `bash ./scripts/run.sh ...`
- `bash ./scripts/run_cluster.sh ...`

## Install

```bash
cd examples/deploy-dynamic-decentralized-pytorch
pip install -e .
```

## Run

### Fastest path for beginners

Run a local cluster in one command:

```bash
./scripts/run_cluster.sh
```

This starts 4 local nodes on ports `9100..9103`.

### Single node

```bash
./scripts/run.sh 9100
```

This starts `flower-super-dnode` in deploy mode with dynamic topology using
[configs/deploy_dynamic.yaml](configs/deploy_dynamic.yaml).

The script runs with:

- `--execution-mode deploy`
- `--config configs/deploy_dynamic.yaml`
- `--nodeapps-pyproject pyproject.toml`
- `--port <PORT>`
- `--node-data-config-json` for partition metadata

Use single-node mode mainly for debugging process startup and local config.

### Multi-node cluster (N nodes on localhost)

```bash
./scripts/run_cluster.sh                  # 4 nodes starting at port 9100
./scripts/run_cluster.sh 8                # 8 nodes starting at port 9100
./scripts/run_cluster.sh 6 9200           # 6 nodes starting at port 9200
```

What the cluster script does:
- Starts **node_0** on `BASE_PORT` first.
- Starts **nodes 1..N-1** on `BASE_PORT+i`.
- All processes run in the background; `Ctrl-C` stops the entire cluster.
- Per-node logs are written to a temporary directory and tailed to stdout.
NodeApp autoload is read from `pyproject.toml`:

```toml
[tool.flwr.app.components]
nodeapp1 = "quickstart_decentralized_pytorch.node_apps:app"
```

## How to verify it is running correctly

After startup, you should see:

- `Starting dynamic cluster: ...`
- `Cluster running (...)`
- continuous per-node log output (`node_0.log`, `node_1.log`, ...)

Use `Ctrl-C` to stop all nodes together.

## Dynamic topology and sampling configuration

Main settings in [configs/deploy_dynamic.yaml](configs/deploy_dynamic.yaml):

- `topology.mode: dynamic`
- `sampling.algorithm: gbps`
- `sampling.config_file: ./config_sampling.json`
- GBPS params: `view_size`, `heal`, `swap`, `selection_policy`, `propagation_policy`, `delay`, `age`

## NodeApp behavior

The NodeApp (`quickstart_decentralized_pytorch.node_apps:app`) does:

- local training on synthetic partitioned data
- local evaluation (`eval_loss`, `eval_acc`)
- decentralized parameter exchange between dynamically connected nodes

Implementation details are in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py).

## Quick customization

- Scale the local cluster with `./scripts/run_cluster.sh <N> [BASE_PORT]`
- Tune sampling behavior in [configs/deploy_dynamic.yaml](configs/deploy_dynamic.yaml)
- Adjust optimization settings in [quickstart_decentralized_pytorch/node_apps.py](quickstart_decentralized_pytorch/node_apps.py) (`local-epochs`, `lr`, `batch-size`)

## Troubleshooting

- `flower-super-dnode: command not found` → run `pip install -e .` again in this folder
- `Address already in use` → choose another base port (for example `./scripts/run_cluster.sh 4 9200`)
- Nodes start but no interaction appears → wait a few seconds for cluster formation and check tailed logs
- Early exit of one node → inspect the printed temp log directory for the failing node
