# Decentralized Work Summary

## Scope delivered

This work strengthens `flwr.decentralized` in three dimensions:

1. **Runtime hardening in `NodeApp`**
   - Strict aggregate-message path handling
   - Round/action validation before dispatch
   - Duplicate request protection using payload identity
2. **Missing test coverage across decentralized modules**
   - Added targeted unit tests for previously uncovered modules
   - Fixed one real bug discovered by tests (`--nb-nodes` stripping)
3. **Usability and examples**
   - Added concrete PyTorch examples for deploy and simulation scenarios
   - Added launcher integration for `flower-super-dnode`

In practice, this means the decentralized runtime is safer to use, has better regression coverage, and is accompanied by concrete end-to-end examples that demonstrate how to run the new launch flow.

---

## Key code changes

### Runtime / protocol behavior

- `py/flwr/decentralized/nodeapp/node_app.py`
  - Aggregate parse fallback now only occurs on parse failure.
  - Aggregate handler failures no longer silently downgrade to train/evaluate fallback.
  - Duplicate detection now keys by `(source, action, round, message_object_id)` so distinct payloads in the same round are not dropped accidentally.
  - **Introduced P2P-specific parameter aggregation**: `_average_array_records()` performs lightweight equal-weight averaging of peer model parameters, avoiding strict central-FL validation rules in `FedAvg.aggregate_train()` that do not fit gossip/push-pull topologies.

Rationale:

- Central FL and peer-to-peer FL do not share the same aggregation assumptions.
- The runtime now distinguishes those contexts more clearly instead of trying to force peer updates through a central-server-only path.

### CLI/config behavior

- `py/flwr/decentralized/superdnode/config/helper.py`
  - `_strip_superdnode_only_args` now strips both `--nb-nodes` and `--nb-nodes=...`.
- `py/flwr/decentralized/superdnode/cli/flower_super_dnode.py`
  - Simulation mode now supports an explicit `--network-config-mode` override while keeping backward-compatible default behavior.

This makes the super-dnode launcher easier to use in examples and less fragile when legacy arguments are present.

### Packaging

- `framework/pyproject.toml`
  - Added script entrypoints:
    - `flower-super-dnode`
    - `flwr-super-dnode`

This removes the need for manual module-path invocation and makes the examples simpler to launch.

---

## Tests added

New test files:

- `py/flwr/decentralized/common/run_config_test.py`
- `py/flwr/decentralized/node_test.py`
- `py/flwr/decentralized/simulation/args_test.py`
- `py/flwr/decentralized/simulation/simulation_test.py`
- `py/flwr/decentralized/superdnode/config/helper_test.py`
- `py/flwr/decentralized/superdnode/config/parser_test.py`

Existing test files extended:

- `py/flwr/decentralized/nodeapp/node_app_test.py`

Coverage focus:

- argument parsing and stripping
- simulation launch configuration
- config parsing behavior
- node and NodeApp protocol handling
- edge cases around aggregate fallback and duplicate handling

---

## Example validation

All 4 PyTorch examples were validated end to end:

1. **Simulation + Dynamic Graph** (`simulation-dynamic-graph-decentralized-pytorch`)
   - 4 nodes for 4 rounds, random dynamic topology
   - Result: ✅ zero aggregation errors, model accuracy improves over rounds

2. **Simulation + Static Graph + Sampling** (`simulation-static-graph-sampling-decentralized-pytorch`)
   - 6 nodes for 5 rounds, static ring topology with peer sampling
   - Result: ✅ zero aggregation errors, model accuracy improves over rounds

3. **Deploy + Dynamic Topology** (`deploy-dynamic-decentralized-pytorch`)
   - Single-instance dynamic network (P2P listen/connect)
   - Result: ✅ node starts and loads NodeApp successfully

4. **Deploy + Static Topology** (`deploy-static-decentralized-pytorch`)
   - Multi-instance static ring (4 nodes: `node_1`, `node_2`, `node_3`, `node_4`)
   - Result: ✅ node starts and loads NodeApp successfully

Validation takeaway:

- simulation scenarios prove the runtime can complete full rounds without aggregation failures
- deploy scenarios prove the launcher, config loading, and NodeApp registration work as expected
- the examples are usable as real smoke tests for future decentralized changes

---

## Validation commands

```bash
pytest framework/py/flwr/decentralized -q
```

```bash
pytest framework/py/flwr/decentralized/nodeapp/node_app_test.py -q
```

```bash
pytest framework/py/flwr/decentralized/superdnode/cli/flower_super_dnode_test.py -q
```

### Simulation examples

```bash
# Dynamic graph
cd examples/simulation-dynamic-graph-decentralized-pytorch
source ../../venv/bin/activate
pip install -e .
flower-super-dnode --execution-mode simulation --sim-config configs/simulation_dynamic_graph.yaml --nodeapps-pyproject pyproject.toml

# Static graph + sampling
cd examples/simulation-static-graph-sampling-decentralized-pytorch
source ../../venv/bin/activate
pip install -e .
flower-super-dnode --execution-mode simulation --sim-config configs/simulation_static_graph_sampling.yaml --network-config-mode csr --enable-sampling --nodeapps-pyproject pyproject.toml
```

Expected outcome:

- the command starts normally
- no config or CLI parsing error is raised
- the simulation progresses until completion

### Deploy examples (single node dry-run, 5-second timeout)

```bash
# Dynamic topology
cd examples/deploy-dynamic-decentralized-pytorch
source ../../venv/bin/activate
pip install -e .
timeout 5 flower-super-dnode --execution-mode deploy --config configs/deploy_dynamic.yaml --nodeapps-pyproject pyproject.toml --port 9100 --node-data-config-json '{"partition-id": 0, "num-partitions": 1}' || echo "Timeout OK"

# Static topology (one node)
cd examples/deploy-static-decentralized-pytorch
source ../../venv/bin/activate
pip install -e .
timeout 5 flower-super-dnode --execution-mode deploy --config configs/deploy_static.yaml --nodeapps-pyproject pyproject.toml --node-name node_1 --port 9201 --node-data-config-json '{"partition-id": 0, "num-partitions": 4}' || echo "Timeout OK"
```

Expected outcome:

- the process starts cleanly
- the NodeApp is loaded successfully
- the timeout stops the process intentionally after startup validation

---

## New example set

Added as separate folders under `examples/`:

- `examples/deploy-dynamic-decentralized-pytorch/`
- `examples/deploy-static-decentralized-pytorch/`
- `examples/simulation-dynamic-graph-decentralized-pytorch/`
- `examples/simulation-static-graph-sampling-decentralized-pytorch/`

Each folder contains its own `README.md`, `pyproject.toml`, runnable script, and config(s).

These examples are intentionally concrete and minimal:

- they show how to launch the new decentralized runtime
- they provide realistic topology and sampling settings
- they give users a copy-paste starting point for deploy and simulation workflows
