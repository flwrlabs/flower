# F7c Kubernetes Executor k3d Real Launch-Path Test

This document explains how to run and interpret the local F7c Kubernetes
executor harness. The test is a real k3d smoke test for one ServerApp task. It
builds local Flower runtime images, runs SuperLink and SuperExec in Kubernetes,
seeds one run through the Control API, observes the TaskExecutor Pod created by
the Kubernetes executor, verifies the evidence bundle, and cleans up by default.

## Quick Start

From the repository root:

```bash
./framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh
```

The script prints build progress, harness progress, and a final verification
summary. A successful run ends with output like:

```text
=== F7c k3d verification ===
Status: passed
Result: real-launch-path
TaskExecutor Pods: 1
  - flwr-taskexecutor-... phase=Succeeded
TaskExecutor log captures: 1
TaskExecutor phases: Succeeded
Cleanup: returncode=0
Verification: PASSED

F7c k3d test passed. Evidence is available at ...
```

## Prerequisites

The wrapper expects these commands on `PATH`:

- `docker`
- `k3d`
- `kubectl`
- `uv`
- `python`

The first run can take longer because it builds a wheel and local Docker images.
The wrapper can create the k3d cluster if it is missing.

## What The Wrapper Does

The one-command wrapper is:

```bash
framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh
```

It performs these steps:

1. Checks that required commands are installed.
2. Builds a framework wheel with `uv build --wheel`.
3. Builds local runtime images:
   - `flwr/base:dev`
   - `flwr/superlink:dev`
   - `flwr/superexec:dev`
4. Verifies the image entrypoints and TaskExecutor commands:
   - `flower-superlink`
   - `flower-superexec`
   - `flwr-serverapp`
   - `flwr-clientapp`
   - `python -c "import kubernetes"`
5. Creates or reuses the `flower-f7` k3d cluster.
6. Imports the local runtime images into k3d.
7. Applies the harness namespace and RBAC objects.
8. Runs SuperLink and SuperExec in the harness namespace.
9. Starts a seed Job that builds a tiny probe FAB and calls Control API
   `StartRun`.
10. Observes the KubernetesExecutor-created TaskExecutor Pod.
11. Waits for the TaskExecutor Pod to reach `Succeeded`.
12. Captures TaskExecutor logs and verifies the probe marker:
    `F7 probe ServerApp ran`.
13. Verifies the evidence bundle with
    `framework/dev/verify-kubernetes-executor-harness.py`.
14. Deletes the harness namespace by default.

The SuperExec image is intentionally also used as the TaskExecutor runtime image
for this slice. The Kubernetes executor overrides the container command to run
`flwr-serverapp`, so a separate TaskExecutor image is not required for this test.

## Useful Options

```bash
./framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh \
  --output-dir /tmp/my-f7c-evidence
```

Write the evidence bundle to a known location.

```bash
./framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh --skip-build
```

Reuse existing local images. The harness still inspects the images before
running and imports them into k3d.

```bash
./framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh --skip-cleanup
```

Leave the namespace and Pods in place for manual inspection. A later run is
reentrant: the harness prunes the prior SuperLink/SuperExec Pods and seed Job
before applying fresh manifests.

```bash
./framework/dev/test-kubernetes-executor-k3d-real-launch-path.sh \
  --tag my-test \
  --cluster-name flower-f7 \
  --namespace flower-f7
```

Use custom image tags, cluster names, or namespaces.

## How To Read The Output

The wrapper has three main output sections:

- `Building local runtime images`: wheel and Docker image build output.
- `Running F7c harness`: the harness writes Kubernetes evidence.
- `Verifying F7c evidence`: a concise pass/fail report from the verifier.

The verifier report is the highest-signal output. For a pass, confirm:

- `Status: passed`
- `Result: real-launch-path`
- `TaskExecutor Pods: 1`
- the TaskExecutor Pod has `phase=Succeeded`
- `TaskExecutor log captures: 1`
- `Cleanup: returncode=0` unless you used `--skip-cleanup`
- `Verification: PASSED`

## Evidence Bundle

Each run writes an evidence directory. The wrapper prints the path at the end.
Useful files include:

- `summary.json`: machine-readable pass/fail summary and key details.
- `events.jsonl`: ordered harness events.
- `harness.log`: short human-readable harness note.
- `objects/namespace.yaml`: rendered namespace manifest.
- `objects/rbac.yaml`: rendered ServiceAccount, Role, and RoleBinding.
- `objects/real-launch.yaml`: rendered SuperLink, SuperExec, and executor
  config manifests.
- `objects/seed-job.yaml`: rendered Control API seed Job.
- `objects/pods.json`: final observed TaskExecutor Pod state.
- `diagnostics/commands.txt`: commands run by the harness and their output.
- `diagnostics/image-preflight.txt`: required images and import command.
- `diagnostics/taskexecutor-logs.txt`: captured TaskExecutor logs.
- `diagnostics/cleanup.txt`: cleanup command and cleanup behavior.

## Test Scope

This slice proves the local real launch path for one ServerApp task.

| Area | Covered? | What this slice verifies |
| --- | --- | --- |
| Local image build | Yes | Builds and verifies the local runtime images. |
| k3d cluster setup | Yes | Creates or reuses the configured k3d cluster. |
| Image import | Yes | Imports SuperLink and SuperExec images into k3d. |
| Namespace/RBAC setup | Yes | Applies namespace, ServiceAccount, Role, and RoleBinding. |
| RBAC auth checks | Yes | Checks expected Pod/Secret permissions and selected negative permissions. |
| SuperLink startup | Yes | Applies SuperLink Pod/Service and waits for Pod readiness. |
| SuperExec startup | Yes | Applies SuperExec with Kubernetes executor config and waits for readiness. |
| FAB/run seeding | Yes | Seed Job builds a probe FAB and calls Control API `StartRun`. |
| TaskExecutor Pod creation | Yes | Observes a TaskExecutor Pod with the expected labels. |
| TaskExecutor terminal phase | Yes | Waits for the Pod to reach `Succeeded` and re-reads Pod state. |
| ServerApp execution marker | Yes | Verifies `F7 probe ServerApp ran` in TaskExecutor logs. |
| Harness cleanup | Yes | Deletes the harness namespace by default. |
| Dirty namespace rerun | Yes | Prunes prior runtime Pods and seed Job before applying new manifests. |

## Explicit Non-Scope

These behaviors are intentionally not proven by this slice:

| Area | Covered? | Clarification |
| --- | --- | --- |
| Capacity waiting | No | No queueing, resource-capacity blocking, or capacity-release proof. |
| Sweeping/reconciler cleanup | No | No sweeper loop or orphan cleanup validation. |
| Executor-driven TaskExecutor Pod deletion | No | Namespace cleanup deletes the Pod; executor-owned deletion is not asserted. |
| Executor-driven Secret deletion | No | Secret RBAC is checked, but per-task Secret cleanup is not asserted. |
| Full AppIo completion semantics | Mostly no | The ServerApp launches and runs, but final AppIo result state is not asserted. |
| ClientApp path | No | This slice exercises only `serverapp`. |
| AppIo TLS | No | The local run uses insecure AppIo. |
| NetworkPolicy/CNI isolation | No | Network isolation is not validated. |
| Production RBAC posture | No | RBAC checks are for this local harness shape only. |
| Concurrency/multiple tasks | No | The harness seeds one run. |
| Failure/retry behavior | No | The real k3d run is a happy-path smoke test. |

## Troubleshooting

If Docker access fails, confirm Docker Desktop or the local Docker daemon is
running and that your user can access the Docker socket.

If `k3d image import` fails, confirm the cluster exists and Docker is reachable.
The wrapper uses `--create-cluster`, so a missing cluster should normally be
created automatically.

If Pods cannot pull local images, rerun without `--skip-build` and confirm the
image import section succeeds.

If a run times out, inspect:

```bash
kubectl --context k3d-flower-f7 get pods -n flower-f7
kubectl --context k3d-flower-f7 logs pod/flower-superlink -n flower-f7
kubectl --context k3d-flower-f7 logs pod/flower-superexec -n flower-f7
```

If you used `--skip-cleanup`, remove the test namespace manually:

```bash
kubectl --context k3d-flower-f7 delete namespace flower-f7 \
  --ignore-not-found=true --wait=true
```

The evidence bundle's `diagnostics/commands.txt` is usually the best first file
to inspect after a failure.
