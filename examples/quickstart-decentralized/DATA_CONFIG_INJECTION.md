# Data Config Injection via CLI

**New Feature**: Inject `data_config` (partition-id, num-partitions) at runtime via the `--node-data-config-json` CLI flag to `flower-super-dnode run`.

## Motivation

Previously, partition IDs and partition counts were hardcoded in `NodeApp` constructors. This made it difficult to:
- Run multiple nodes with different partitions without code changes
- Scale to arbitrary numbers of partitions
- Test federated scenarios easily

Now you can:
- Launch N nodes on the fly with distinct partitions
- Pass partition config directly from the CLI or orchestration scripts

## Design

### Architecture

1. **NodeApp changes**:
   - `data_config` is now **optional** in `__init__` (defaults to empty `ConfigRecord`)
   - New method `set_data_config(override)` merges provided config into the app
   - Validation deferred: only validates when config is explicitly set or already provided

2. **CLI changes**:
   - New flag: `--node-data-config-json <json_string>`
   - Applied to all loaded NodeApps on that node
   - Parsed and validated before registration

3. **Deployment script** (`run_n_nodes.sh`):
   - Auto-generates `partition-id=$i` for node `i`
   - All nodes share `num-partitions=$N`

### Priority (what takes precedence)

1. **CLI override** (highest) — `--node-data-config-json`
2. **pyproject.toml definition** — in app constructor
3. **Default in code** — empty `ConfigRecord()` (must be set before use)

## Usage

### Manual (single node)

```bash
# Start one node as partition 2 out of 10
flower-super-dnode run \
  --config configs/node_dynamic.yaml \
  --nodeapps-pyproject pyproject.toml \
  --port 9100 \
  --node-name node_0 \
  --node-data-config-json '{"partition-id": 2, "num-partitions": 10}'
```

### Scripted (multiple nodes)

```bash
# Launch 4 nodes, each with a unique partition
./scripts/run_n_nodes.sh 4 9100 configs/node_dynamic.yaml
```

This automatically sets:
- Node 0: `partition-id: 0, num-partitions: 4`
- Node 1: `partition-id: 1, num-partitions: 4`
- Node 2: `partition-id: 2, num-partitions: 4`
- Node 3: `partition-id: 3, num-partitions: 4`

### Programmatic (Python)

```python
from flwr.decentralized.nodeapp import create_nodeapps_from_pyproject
from pathlib import Path

apps = create_nodeapps_from_pyproject(Path("pyproject.toml"))
for app in apps.values():
    app.set_data_config({
        "partition-id": 1,
        "num-partitions": 10,
    })
    # Now app.data_config is populated; safe to use
```

## Backward Compatibility

- Apps with pre-configured `data_config` in code still work (checked before CLI is applied)
- Apps without `data_config` must get one via CLI or `set_data_config()` before runtime
- Tests and single-node examples remain unchanged

## Example with PyTorch partitioned data

The `node_apps_pytorch.py` example now supports this pattern:

```bash
# 3 nodes, each training on 1/3 of CIFAR-10
./scripts/run_n_nodes.sh 3 9100 configs/node_dynamic.yaml
```

Each node loads its partition via `FederatedDataset + IidPartitioner`, keyed on `partition-id` and `num-partitions`.

## Validation

Data config is validated when:
1. Explicitly set via `set_data_config()` → raises if missing required keys
2. Loaded from pyproject → raises on init if invalid
3. At registration time → app must have valid config before use

Required keys: `partition-id`, `num-partitions` (both integers).

## Logging

When data_config is injected:
```
INFO: Parsed CLI data_config override: {'partition-id': 2, 'num-partitions': 10}
INFO: Applied data_config override to NodeApp 'trainer-pytorch'
INFO: Applied data_config override to NodeApp 'analytics-pytorch'
```
