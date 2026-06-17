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

To run the budget-1/two-task capacity and cleanup proof:

```bash
./framework/dev/k8s/test-real-launch-path.sh --capacity-cleanup-proof
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
| `invocation.json` | Redacted invocation, cwd/repo context, selected profile, and run settings. |
| `events.jsonl` | Ordered harness events. |
| `task-lineage.json` | Seeded run to TaskExecutor Pod and credential Secret mapping. |
| `taskexecutor-pods.json` | Full redacted TaskExecutor Pod object snapshot. |
| `taskexecutor-secrets.redacted.json` | Redacted per-task Secret evidence with key names and byte lengths. |
| `final-state.json` | Pre-cleanup resource counts and object summaries for the run selectors. |
| `proof-checklist.json` | Reviewer-facing map from claims to artifact fields, with out-of-scope claims. |
| `objects/capacity-blocked-pods.json` | Capacity-proof snapshot of the first active TaskExecutor Pod. |
| `objects/secrets-before-cleanup.redacted.json` | Capacity-proof redacted Secret snapshot before executor cleanup. |
| `objects/cleanup-pods.json` | Capacity-proof TaskExecutor Pod state after capacity opens. |
| `objects/secrets-after-cleanup.redacted.json` | Capacity-proof redacted Secret snapshot after executor cleanup. |
| `objects/real-launch.yaml` | Rendered SuperLink, executor config, and SuperExec objects. |
| `objects/seed-job.yaml` | Rendered seed ConfigMap and Job. |
| `objects/pods.json` | Observed TaskExecutor Pod list and phases. |
| `diagnostics/commands.txt` | Planned or executed host commands. |
| `diagnostics/taskexecutor-logs.txt` | Captured TaskExecutor logs. |
| `diagnostics/cleanup.txt` | Cleanup defaults and the namespace delete command. |

## How The Evidence Proves Correctness

Use this section when reviewing an evidence directory without rerunning the
harness. The generated `proof-checklist.json` contains the same claim-to-file
map in machine-readable form.

1. Confirm the run was real and from the expected source tree.

   Open `invocation.json` and check:

   - `mode` is `local-k8s-launch-path`;
   - `dry_run` is `false`;
   - `repo.branch` and `repo.sha` match the checkout under review;
   - `equivalent_argv` shows the harness mode, output directory, namespace,
     images, `--execute`, `--apply-manifests`, and `--import-images`.

2. Confirm SuperExec was actually configured to use the Kubernetes executor.

   Open `objects/real-launch.yaml` and inspect the SuperExec Pod. Its container
   args must include `--executor kubernetes` and `--executor-config
   /etc/flower/executor-config.yaml`. The ConfigMap in the same file should
   contain the executor config used to render TaskExecutor Pods, including the
   namespace, image, resource pool, and harness-run label.

3. Confirm one deterministic ServerApp task was seeded through AppIo.

   Open `objects/seed-job.yaml` and check that the Job runs
   `/opt/flower-local-k8s/seed_run.py` against the SuperLink Control API.
   Then check `summary.json` and `task-lineage.json`: `seed_run_id` and
   `seeded_run_id` should be present and should match.

4. Confirm the Kubernetes executor created the TaskExecutor Pod.

   Open `task-lineage.json`. Each task record should have a `pod_name`,
   `pod_uid`, `task_id`, `launch_attempt`, `resource_pool`, and
   `credential_secret_name`. Then open `taskexecutor-pods.json` and find the
   same Pod. Its labels should include:

   - `app.kubernetes.io/component: taskexecutor`;
   - `flower.ai/harness-run`;
   - `flower.ai/superexec-task-id`;
   - `flower.ai/launch-attempt`;
   - `flower.ai/resource-pool`.

   The Pod spec should show the TaskExecutor command, `--token-file
   /run/flwr/appio/token`, and a Secret volume mounted at `/run/flwr/appio`.

5. Confirm the credential Secret existed without exposing the token.

   Open `taskexecutor-secrets.redacted.json`. The matching Secret should have
   the same task labels as the Pod, a `token` entry in `data_keys`, useful byte
   length evidence in `data_byte_lengths`, and `redacted: true`. The file must
   not contain the token value.

6. Confirm the TaskExecutor actually ran the probe ServerApp.

   Open `diagnostics/taskexecutor-logs.txt` and look for
   `K8s launch probe ServerApp ran`. The verifier requires this marker. Also
   check `taskexecutor-pods.json` or `summary.json` for Pod phase `Succeeded`.

7. Confirm the final state was captured before broad namespace cleanup.

   Open `final-state.json`. It records the Pod, Secret, Job, Service, and
   Namespace observation commands plus resource counts before namespace
   deletion. This proves what remained at the end of the proof stage. It does
   not claim executor-owned completed Pod or Secret cleanup; that is deliberately
   out of scope for this slice.

8. Confirm the verifier accepted the bundle.

   The wrapper runs `framework/dev/k8s/verify_evidence.py` after the harness.
   A passing report should show `Verification: PASSED`, one TaskExecutor Pod,
   one lineage record, one credential Secret record, final state Pod/Secret
   counts, a `Succeeded` phase, and a successful cleanup command when cleanup
   was required.

For `--capacity-cleanup-proof`, additionally confirm:

1. `objects/executor-config.yaml` sets `active-pod-budget: 1`.
2. `summary.json` lists two `seed_run_ids`.
3. `events.jsonl` has a passing `capacity.wait_observed` event.
4. `summary.json` has `cleanup_observed.observed: true` with removed Pod and
   Secret names for the first task.
5. `objects/cleanup-pods.json` and
   `objects/secrets-after-cleanup.redacted.json` show the post-wait selector
   state before broad namespace cleanup.

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
| Capacity wait | Optional | `--capacity-cleanup-proof` seeds two runs with active Pod budget `1` and requires SuperExec wait evidence. |
| Sweeper cleanup | Optional | `--capacity-cleanup-proof` requires the first completed TaskExecutor Pod and Secret to be removed before namespace cleanup. |
| Wrapper cleanup | Yes | Default wrapper behavior deletes the namespace and verifies cleanup evidence. |

## Out Of Scope

| Area | Tested | Notes |
| --- | --- | --- |
| Cardinality proof | No | The capacity proof uses budget `1` and two tasks; budget `2`/three-task cardinality is a later slice. |
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
