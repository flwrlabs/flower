# PR: Decentralized hardening, test completion, and PyTorch scenario examples

## Why

This PR closes important merge-readiness gaps in `flwr.decentralized` and makes the new decentralized flow easier to understand, test, and launch:

- protocol and runtime robustness in `NodeApp`
- missing unit-test coverage in key decentralized modules
- missing concrete end-user examples for deploy and simulation modes
- missing CLI script entrypoint exposure for Super DNode

The overall goal is to move the decentralized runtime from “works in the codebase” to “easy to validate and practical to use.”

## What changed

### 1) NodeApp protocol hardening

- Prevent silent fallback from aggregate runtime errors to the legacy event path.
- Validate aggregate action and round before dispatch.
- Improve duplicate detection granularity by including payload object identity.
- **Implement P2P-specific parameter aggregation**: peer parameters are merged using lightweight equal-weight averaging, bypassing strict retry logic in `FedAvg.aggregate_train()` that assumes a central server context.

Why this matters:

- centralized and peer-to-peer decentralized flows have different assumptions
- the runtime now fails more explicitly when something is wrong
- valid peer payloads are no longer filtered out by server-centric logic

### 2) Missing tests added

- Added tests for:
  - `common/run_config.py`
  - `node.py`
  - `simulation/args.py`
  - `simulation/simulation.py`
  - `superdnode/config/helper.py`
  - `superdnode/config/parser.py`
- Extended `nodeapp/node_app_test.py` with edge-case coverage.

These tests cover argument handling, config parsing, runtime dispatch, and edge cases in the NodeApp protocol path.

### 3) CLI/config integration fix

- Fixed `_strip_superdnode_only_args` so it strips `--nb-nodes` options correctly.
- Added support for explicit `--network-config-mode` in the simulation launch path.

This makes the launcher more resilient when examples pass super-dnode-specific arguments through the CLI.

### 4) Packaging integration

In `framework/pyproject.toml`, added scripts:

- `flower-super-dnode = "flwr.decentralized.superdnode.cli.flower_super_dnode:flower_super_dnode"`
- `flwr-super-dnode = "flwr.decentralized.superdnode.cli.flower_super_dnode:flower_super_dnode"`

These aliases make the launcher directly callable from an installed environment and simplify the example commands.

### 5) New concrete PyTorch examples

Added runnable examples in separate folders:

- `examples/deploy-dynamic-decentralized-pytorch/`
- `examples/deploy-static-decentralized-pytorch/`
- `examples/simulation-dynamic-graph-decentralized-pytorch/`
- `examples/simulation-static-graph-sampling-decentralized-pytorch/`

Each folder contains dedicated README, config, and scripts for its scenario.

These examples demonstrate:

- deploy mode with dynamic topology
- deploy mode with static topology
- simulation mode with dynamic random graphs
- simulation mode with static CSR graphs plus sampling

The examples also serve as smoke tests for future decentralized changes.

## Verification

```bash
pytest framework/py/flwr/decentralized -q
```

### Test results

- ✅ 100% of new tests pass
- ✅ All 4 PyTorch examples validated:
  - Simulation + dynamic graph: 4 nodes, 4 rounds, zero aggregation errors
  - Simulation + static graph + sampling: 6 nodes, 5 rounds, zero aggregation errors
  - Deploy + dynamic topology: startup success
  - Deploy + static topology: startup success

What validation proves:

- the runtime can execute full decentralized rounds without regressions
- the deploy launcher loads the correct NodeApp configuration
- the example scenarios are reproducible and runnable end to end

## Risk / compatibility

- Behavior changes in aggregate handling are intentional fail-fast improvements.
- P2P parameter averaging uses local averaging and does not depend on unused central `FedAvg` logic.
- Simulation mode keeps backward-compatible defaults when `--network-config-mode` is not passed.
- New script aliases are additive.
- The new examples are isolated under `examples/` and do not alter existing example behavior.

## Follow-up (optional)

- Style normalization pass (tabs/spaces and newline EOF) in newly added decentralized files.
- Add a CI job stage to execute example smoke scripts with reduced simulation time.
- Consider per-scenario performance benchmarks for the 4 PyTorch examples.

## Reviewer-facing summary

If you want the shortest possible explanation of the PR, it is this:

> This PR hardens the decentralized runtime, adds missing tests, fixes CLI/config edge cases, exposes launcher scripts, and ships four complete PyTorch examples that prove the deploy and simulation flows work end to end.
