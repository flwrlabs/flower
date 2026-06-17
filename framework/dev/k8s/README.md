# Local k8s Launch-Path Harness

This dev-only harness builds local Flower runtime images, configures a k3d
cluster, starts SuperLink and SuperExec, seeds one ServerApp run through the
Control API, and verifies that the Kubernetes executor creates a TaskExecutor
Pod that reaches `Succeeded`.

Run the full local smoke test from the repository root:

```bash
./framework/dev/k8s/test-real-launch-path.sh
```

To reuse previously built images:

```bash
./framework/dev/k8s/test-real-launch-path.sh --skip-build
```

The wrapper deletes the test namespace by default. To inspect resources after a
run:

```bash
./framework/dev/k8s/test-real-launch-path.sh --skip-cleanup
```

## Defaults

| Setting | Default |
| --- | --- |
| Cluster | `flower-local-k8s` |
| Namespace | `flower-local-k8s` |
| Seed Job | `flower-local-k8s-seed-run` |
| Executor ConfigMap | `flower-local-k8s-executor-config` |
| Result | `local-k8s-launch-path` |
| ServerApp marker | `K8s launch probe ServerApp ran` |

## Output

The wrapper prints each stage, then the verifier prints a concise report:

```text
=== local k8s launch-path verification ===
Status: passed
Result: local-k8s-launch-path
TaskExecutor Pods: 1
TaskExecutor phases: Succeeded
Cleanup required: true
Verification: PASSED
```

Evidence is written under the selected output directory:

| Path | Purpose |
| --- | --- |
| `summary.json` | Machine-readable result and details. |
| `events.jsonl` | Ordered harness events. |
| `objects/real-launch.yaml` | Rendered SuperLink, executor config, and SuperExec objects. |
| `objects/seed-job.yaml` | Rendered seed ConfigMap and Job. |
| `objects/pods.json` | Observed TaskExecutor Pod list and phases. |
| `diagnostics/commands.txt` | Planned or executed host commands. |
| `diagnostics/taskexecutor-logs.txt` | Captured TaskExecutor logs. |
| `diagnostics/cleanup.txt` | Cleanup defaults and the namespace delete command. |

## What Is Tested

| Area | Tested | Notes |
| --- | --- | --- |
| Local runtime image build/import | Yes | The wrapper builds images unless `--skip-build`; the harness inspects and imports them into k3d. |
| k3d cluster setup | Yes | The harness creates the named cluster when needed. |
| Namespace/RBAC apply | Yes | Applies namespace and SuperExec Pod/Secret Role, then checks expected `kubectl auth can-i` results. |
| SuperLink/SuperExec startup | Yes | Waits for both Pods to become Ready. |
| Control API seed run | Yes | A seed Job builds the probe FAB from mounted files and calls `StartRun`. |
| TaskExecutor Pod creation | Yes | Polls for a Pod matching the run selector before failing. |
| TaskExecutor terminal phase | Yes | Waits for observed TaskExecutor Pods to reach `Succeeded`. |
| ServerApp execution marker | Yes | Verifies `K8s launch probe ServerApp ran` in TaskExecutor logs. |
| Wrapper cleanup | Yes | Default wrapper behavior deletes the namespace and verifies cleanup evidence. |

## Out Of Scope

| Area | Tested | Notes |
| --- | --- | --- |
| Capacity waiting | No | No capacity queue or resource-pool wait behavior is asserted. |
| Sweeper cleanup | No | No reconciler or orphan cleanup loop is validated. |
| Executor-owned Pod deletion | No | Namespace cleanup removes resources; executor deletion behavior is not proven. |
| Executor-owned Secret deletion | No | Secret RBAC is checked, but per-task Secret lifecycle is not asserted. |
| AppIo result completion semantics | No | This slice observes launch and Pod success, not full result semantics. |
| ClientApp execution | No | The probe includes a minimal ClientApp file only because the FAB schema expects it. |
| TLS, CNI/NetworkPolicy, production RBAC | No | This is local/dev-only and uses insecure local AppIo. |
| Concurrency, retry, failure behavior | No | The harness starts one deterministic run. |

## Useful Commands

Inspect resources after `--skip-cleanup`:

```bash
kubectl --context k3d-flower-local-k8s get pods -n flower-local-k8s
kubectl --context k3d-flower-local-k8s logs pod/flower-superlink -n flower-local-k8s
kubectl --context k3d-flower-local-k8s logs pod/flower-superexec -n flower-local-k8s
```

Remove the namespace manually:

```bash
kubectl --context k3d-flower-local-k8s delete namespace flower-local-k8s \
  --ignore-not-found=true --wait=true
```
