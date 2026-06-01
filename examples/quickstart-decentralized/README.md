# Quickstart: Flower Decentralized (DNode)

This example shows three ways to launch a Flower decentralized node (`DNode`):

| Script | Method |
|---|---|
| `run_dynamic` | YAML config file + dynamic (gossip) topology |
| `run_static` | YAML config file + static (pre-defined) topology |
| `run_programmatic` | Pure Python, no config file |

---

## NodeApp autoload from `pyproject.toml`

When you start `flower-super-dnode` in this folder, NodeApps are auto-loaded
from `pyproject.toml` using `[tool.flwr.app.components]`.

This quickstart defines two subjects:
- `trainer`
- `analytics`

with handlers mapped in
[quickstart_decentralized/node_apps.py](quickstart_decentralized/node_apps.py).

A PyTorch variant with data partitioning (IID partitions, CIFAR-10) is also available in
[quickstart_decentralized/node_apps_pytorch.py](quickstart_decentralized/node_apps_pytorch.py).

Example mapping:

```toml
[tool.flwr.app.components]
nodeapp1 = "quickstart_decentralized.node_apps:app1"
nodeapp2 = "quickstart_decentralized.node_apps:app2"
```

And user-defined NodeApps in Python:

```python
from flwr.common import Context, Message
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.configrecord import ConfigRecord

DATA_CONFIG = ConfigRecord({
    PARTITION_ID_KEY: 0,
    NUM_PARTITIONS_KEY: 1,
})

app1 = NodeApp(
    subject="trainer",
    initial_arrays=ArrayRecord(),
    data_config=ConfigRecord(DATA_CONFIG),
    train_config=ConfigRecord({"local-epochs": 2, "lr": 0.05}),
    eval_config=ConfigRecord({"metric-name": "accuracy"}),
)
app2 = NodeApp(
    subject="analytics",
    initial_arrays=ArrayRecord(),
    data_config=ConfigRecord(DATA_CONFIG),
    train_config=ConfigRecord({"window": 25}),
    eval_config=ConfigRecord(),
)

@app1.train()
def train_app1(message: Message, context: Context) -> Message:
    ...
    return message

@app1.evaluate()
def evaluate_app1(message: Message, context: Context) -> Message:
    ...
    return message

@app2.train()
def train_app2(message: Message, context: Context) -> Message:
    ...
    return message

@app2.evaluate()
def evaluate_app2(message: Message, context: Context) -> Message:
    ...
    return message
```

Disable autoload if needed:

```bash
flower-super-dnode --config configs/node_dynamic.yaml --disable-nodeapps-autoload
```

---

## Install

```bash
cd examples/quickstart-decentralized
pip install -e .
```

For the PyTorch variant, install optional dependencies:

```bash
pip install -e .[pytorch]
```

To use the PyTorch NodeApps, point component entries in `pyproject.toml` to:

```toml
[tool.flwr.app.components]
nodeapp1 = "quickstart_decentralized.node_apps_pytorch:app1"
nodeapp2 = "quickstart_decentralized.node_apps_pytorch:app2"
```

---

## 1 — Dynamic topology (YAML config)

Start a first node on port 9100:

```bash
python -m quickstart_decentralized.run_dynamic \
    --config configs/node_dynamic.yaml \
    --port 9100
```

Start a second node on port 9101 that will discover the first via mDNS:

```bash
python -m quickstart_decentralized.run_dynamic \
    --config configs/node_dynamic.yaml \
    --port 9101
```

Or use explicit bootnodes when mDNS is unavailable:

```bash
python -m quickstart_decentralized.run_dynamic \
    --config configs/node_dynamic.yaml \
    --port 9101 \
    --bootnodes 127.0.0.1:9100
```

### Override any field without editing the file

```bash
python -m quickstart_decentralized.run_dynamic \
    --config configs/node_dynamic.yaml \
    --context my-experiment \
    --port 9200
```

### TOML alternative

```bash
python -m quickstart_decentralized.run_dynamic \
    --config configs/node_dynamic.toml
```

---

## 2 — Static topology (YAML config, auto-generated)

The static config auto-generates a `ring` topology for 4 nodes.
Each node must be launched with a unique `--node-name`.

Terminal 1:
```bash
python -m quickstart_decentralized.run_static \
    --config configs/node_static.yaml \
    --node-name node_1
```

Terminal 2:
```bash
python -m quickstart_decentralized.run_static \
    --config configs/node_static.yaml \
    --node-name node_2
```

…and so on for `node_2`, `node_3`.

Use an existing topology YAML instead of auto-generating:

```bash
python -m quickstart_decentralized.run_static \
    --config configs/node_static.yaml \
    --topology-file /path/to/my_topology.yaml \
    --node-name node_0
```

---

## 3 — Programmatic (pure Python)

```bash
python -m quickstart_decentralized.run_programmatic
```

See [quickstart_decentralized/run_programmatic.py](quickstart_decentralized/run_programmatic.py)
for a detailed example of building `RuntimeNode` entirely in code.

---

## 4 — Launch `n` nodes with one command

Use the helper script to avoid opening multiple terminals:

```bash
chmod +x scripts/run_n_nodes.sh
./scripts/run_n_nodes.sh 3 9100 configs/node_dynamic.yaml
```

This starts 3 nodes on ports `9100`, `9101`, `9102` and writes logs to
`logs/run_n_nodes/`.

If `flower-super-dnode` is not in your PATH, override command resolution:

```bash
SUPER_DNODE_CMD="python -m flwr.decentralized.superdnode.cli.flower_super_dnode" \
./scripts/run_n_nodes.sh 3
```

---

## Config file reference

### YAML (`node_dynamic.yaml`)

```yaml
context: quickstart        # shared by all nodes that should communicate
address: 0.0.0.0
port: 9100
tcp: true
udp: false
bootnodes:                 # optional
  - "192.168.1.5:9100"

topology:
  mode: dynamic            # dynamic | static

sampling:                  # required for dynamic mode
  algorithm: gbps          # gbps | brahams | basalt
  config_file: /tmp/sampling.json
  params:
    view_size: 10
    heal: 2
    swap: 3
    selection_policy: rand
    propagation_policy: pushpull
    delay: 5
    age: 1

network:                   # all fields optional
  idle_connection_timeout_secs: 60
  enable_mdns: true
  enable_kad: true
```

### TOML alternative

See [configs/node_dynamic.toml](configs/node_dynamic.toml).

### CLI priority

```
CLI flag  >  config file value  >  built-in default
```

---

## Running the tests

From the repository root:

```bash
pytest framework/py/flwr/decentralized/common/args_test.py -v
```
