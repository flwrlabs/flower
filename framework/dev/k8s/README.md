# Local k8s Launch-Path Harness

This dev-only harness builds local Flower runtime images, configures a k3d
cluster, starts SuperLink and SuperExec, seeds deterministic ServerApp runs
through the Control API, and verifies that the Kubernetes executor creates
TaskExecutor Pods that reach `Succeeded`.

It has three common modes:

- the default one-task launch-path proof; and
- the `--capacity-cleanup-proof` mode, which uses active Pod budget `1`, seeds
  two tasks, observes SuperExec waiting for capacity, and verifies completed
  TaskExecutor Pod/Secret cleanup before broad namespace cleanup; and
- the `--demo` preset, which uses active Pod budget `4`, seeds eight tasks,
  keeps probe ServerApps active for inspection, leaves resources in place, and
  proves the capacity cardinality case.

## Prerequisites

Run commands from the repository root. The wrapper expects these tools on
`PATH`:

- `docker`;
- `k3d`;
- `kubectl`;
- `uv`;
- `python`.

The Docker daemon must already be running. If `--skip-build` is used, the local
runtime images selected by the wrapper must already exist and be importable into
k3d.

## Quick Runs

Run the full local smoke test from the repository root:

```bash
./framework/dev/k8s/test-real-launch-path.sh
```

To reuse previously built images:

```bash
./framework/dev/k8s/test-real-launch-path.sh --skip-build
```

Add `--tls` to any canned run to enable local server-auth TLS for SuperLink,
SuperExec, the seed Job, and TaskExecutor AppIo. The wrapper generates a local
test CA/server certificate under the evidence directory, applies a Kubernetes
Secret, and verifies TLS evidence after the run.

Default launch-path proof with TLS:

```bash
./framework/dev/k8s/test-real-launch-path.sh --tls
```

Skip-build launch-path proof with TLS:

```bash
./framework/dev/k8s/test-real-launch-path.sh --skip-build --tls
```

To run the budget-1/two-task capacity and cleanup proof:

```bash
output_dir=/private/tmp/f7d-v2-capacity-cleanup-proof-$(date +%Y%m%d-%H%M%S)
./framework/dev/k8s/test-real-launch-path.sh \
  --capacity-cleanup-proof \
  --output-dir "${output_dir}"
```

Capacity and cleanup proof with TLS:

```bash
output_dir=/private/tmp/f7d-v2-capacity-cleanup-proof-tls-$(date +%Y%m%d-%H%M%S)
./framework/dev/k8s/test-real-launch-path.sh \
  --capacity-cleanup-proof \
  --tls \
  --output-dir "${output_dir}"
```

To verify the saved capacity evidence manually after the wrapper finishes:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof
```

To run the demo-friendly budget-4/eight-task cardinality proof:

```bash
output_dir=/private/tmp/f7e-demo-cardinality-proof-$(date +%Y%m%d-%H%M%S)
./framework/dev/k8s/test-real-launch-path.sh \
  --demo \
  --output-dir "${output_dir}"
```

Demo-friendly cardinality proof with TLS:

```bash
output_dir=/private/tmp/f7e-demo-cardinality-proof-tls-$(date +%Y%m%d-%H%M%S)
./framework/dev/k8s/test-real-launch-path.sh \
  --demo \
  --tls \
  --output-dir "${output_dir}"
```

The demo preset leaves namespace resources in place for live inspection. Verify
the saved bundle with the explicit demo expectations:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof \
  --expected-active-pod-budget 4 \
  --expected-seed-run-count 8 \
  --no-require-cleanup
```

For the TLS demo preset, add `--require-tls`:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof \
  --expected-active-pod-budget 4 \
  --expected-seed-run-count 8 \
  --no-require-cleanup \
  --require-tls
```

`/private/tmp` is only an example local scratch location. For handoff or review,
choose a durable writable directory, or archive the completed evidence directory
after saving the verifier report.

The wrapper prints verifier output to stdout. To make the verifier report part
of an evidence bundle for review, rerun the verifier and save the output:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof \
  > "${output_dir}/diagnostics/verifier-output.txt"
```

The wrapper deletes the test namespace by default. To inspect resources after a
run:

```bash
output_dir=/private/tmp/f7d-v2-capacity-cleanup-proof-live-$(date +%Y%m%d-%H%M%S)
./framework/dev/k8s/test-real-launch-path.sh \
  --capacity-cleanup-proof \
  --skip-cleanup \
  --tls \
  --output-dir "${output_dir}"
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof \
  --no-require-cleanup \
  --require-tls
```

## Release Testing Toolkit

Use `harnessctl.sh` when you want to run the local harness one step at a time
for release testing, demos, or manual failure scenarios. It reuses the same
default k3d cluster, namespace, images, manifest renderers, probe app, and
Kubernetes executor config as the all-in-one wrapper.

Build or reuse local images first:

```bash
./framework/dev/k8s/build-local-runtime-images.sh
```

This builds the local SuperLink and SuperExec images used by the harness.
Set `IMPORT_IMAGES=false` only when the selected cluster can already pull the
selected images.

Start SuperLink:

```bash
./framework/dev/k8s/harnessctl.sh start-superlink
```

This creates the default k3d cluster if needed, imports the SuperLink image,
applies the namespace, Service, and Pod, then waits for the Pod to be ready.

Start SuperExec with the Kubernetes executor:

```bash
./framework/dev/k8s/harnessctl.sh start-superexec --active-pod-budget 2
```

This applies the SuperExec ServiceAccount/RBAC, executor ConfigMap, and Pod,
then waits for SuperExec readiness. The active Pod budget is optional; omit it
to use the executor defaults.

Enable SuperLink and AppIo TLS:

```bash
./framework/dev/k8s/harnessctl.sh init-tls
./framework/dev/k8s/harnessctl.sh start-superlink --tls
./framework/dev/k8s/harnessctl.sh start-superexec --tls --active-pod-budget 2
./framework/dev/k8s/harnessctl.sh seed --tls --count 3 --hold-seconds 45
```

This generates a local test CA/server certificate, stores it in a Kubernetes
Secret, starts SuperLink without `--insecure`, and configures SuperExec,
TaskExecutors, and the seed Job to trust the same CA. This is server-auth TLS
for the local Fleet/Control and AppIo paths, not mTLS.

Seed held probe ServerApp tasks:

```bash
./framework/dev/k8s/harnessctl.sh seed --count 3 --hold-seconds 45
```

This creates three deterministic ServerApp runs through the local Control API.
Each probe TaskExecutor stays active for roughly 45 seconds so Pod scheduling,
capacity waiting, and cleanup can be observed.

Watch all Pods:

```bash
./framework/dev/k8s/harnessctl.sh watch-pods
```

This opens a live Pod view for the harness namespace. On systems without
`watch`, the wrapper falls back to a one-second shell loop.

Watch only TaskExecutors:

```bash
./framework/dev/k8s/harnessctl.sh watch-taskexecutors
```

This shows only TaskExecutor Pods for the current harness run, including
resource-pool, task-id, and launch-attempt labels.

Inspect SuperExec logs:

```bash
./framework/dev/k8s/harnessctl.sh logs-superexec --tail=200
```

This prints the current SuperExec Pod logs. Add `-f` to follow.

Inspect TaskExecutor logs:

```bash
./framework/dev/k8s/harnessctl.sh logs-taskexecutors --tail=200 --prefix
```

This prints logs for TaskExecutor Pods selected by the harness labels. Add `-f`
to follow active Pods.

macOS watch and tmux setup:

```bash
brew install watch tmux
```

This installs the `watch` command used by the live Pod views and `tmux` for
keeping watch and log panes open during a release-test run.

Two-pane tmux demo layout:

```bash
tmux new-session -d -s flower-k8s-demo -n release './framework/dev/k8s/harnessctl.sh watch-taskexecutors'
tmux split-window -h -t flower-k8s-demo:release './framework/dev/k8s/harnessctl.sh logs-superexec -f --tail=200'
tmux attach -t flower-k8s-demo
```

This opens TaskExecutor Pod status on the left and SuperExec logs on the right.
Detach with `Ctrl-b d`; reattach with `tmux attach -t flower-k8s-demo`.

Four-pane tmux demo layout:

```bash
tmux new-session -d -s flower-k8s-demo4 -n release './framework/dev/k8s/harnessctl.sh watch-pods'
tmux split-window -h -t flower-k8s-demo4:release './framework/dev/k8s/harnessctl.sh watch-taskexecutors'
tmux split-window -v -t flower-k8s-demo4:release.0 './framework/dev/k8s/harnessctl.sh logs-superexec -f --tail=200'
tmux split-window -v -t flower-k8s-demo4:release.1 './framework/dev/k8s/harnessctl.sh logs-taskexecutors -f --tail=200 --prefix'
tmux select-layout -t flower-k8s-demo4:release tiled
tmux attach -t flower-k8s-demo4
```

This opens all Pod status, TaskExecutor-only status, SuperExec logs, and
TaskExecutor logs in one terminal window.

Stop currently observed TaskExecutors:

```bash
./framework/dev/k8s/harnessctl.sh stop-taskexecutors --count 1
```

This deletes one selected TaskExecutor Pod for the active harness run. The
wrapper chooses Pods in creation-time order.

Kill SuperExec:

```bash
./framework/dev/k8s/harnessctl.sh kill-superexec
```

This deletes the SuperExec Kubernetes Pod. It does not send a signal inside the
container.

Clean up the local namespace:

```bash
./framework/dev/k8s/harnessctl.sh cleanup
```

This deletes the harness namespace and waits for Kubernetes cleanup to finish.

Minimal happy path:

```bash
./framework/dev/k8s/build-local-runtime-images.sh
./framework/dev/k8s/harnessctl.sh start-superlink
./framework/dev/k8s/harnessctl.sh start-superexec --active-pod-budget 2
./framework/dev/k8s/harnessctl.sh seed --count 3 --hold-seconds 45
./framework/dev/k8s/harnessctl.sh watch-taskexecutors
./framework/dev/k8s/harnessctl.sh logs-superexec --tail=200
./framework/dev/k8s/harnessctl.sh logs-taskexecutors --tail=200 --prefix
./framework/dev/k8s/harnessctl.sh stop-taskexecutors --count 1
./framework/dev/k8s/harnessctl.sh kill-superexec
./framework/dev/k8s/harnessctl.sh cleanup
```

Crash-task scenario:

```bash
./framework/dev/k8s/harnessctl.sh start-superlink
./framework/dev/k8s/harnessctl.sh start-superexec
./framework/dev/k8s/harnessctl.sh seed --count 2 --crash
./framework/dev/k8s/harnessctl.sh watch-taskexecutors
./framework/dev/k8s/harnessctl.sh logs-taskexecutors --tail=200 --prefix
./framework/dev/k8s/harnessctl.sh cleanup
```

The crash scenario fails inside the probe ServerApp running in TaskExecutor
Pods. It is not a SuperExec crash test.

Environment overrides:

```bash
CLUSTER_NAME=flower-release-k8s \
NAMESPACE=flower-release-k8s \
IMAGE_TAG=dev \
./framework/dev/k8s/harnessctl.sh start-superlink

CLUSTER_NAME=flower-release-k8s \
NAMESPACE=flower-release-k8s \
IMAGE_TAG=dev \
./framework/dev/k8s/harnessctl.sh start-superexec --active-pod-budget 4
```

The most useful overrides are `CLUSTER_NAME`, `KUBECTL_CONTEXT`, `NAMESPACE`,
`IMAGE_TAG`, `SUPERLINK_IMAGE`, `SUPEREXEC_IMAGE`, `TASKEXECUTOR_IMAGE`,
`ACTIVE_POD_BUDGET`, `CREATE_CLUSTER`, `IMPORT_IMAGES`, and `OUTPUT_DIR`.

`kill-superexec-after-claim-before-launch` is not implemented in this slice.
That scenario needs deterministic SuperExec fault injection after `ClaimTask`
and before TaskExecutor launch; an external timing-based kill would be too
racy for this wrapper.

## Defaults

| Setting | Default |
| --- | --- |
| Cluster | `flower-local-k8s` |
| Namespace | `flower-local-k8s` |
| Seed Job | `flower-local-k8s-seed-run` |
| Executor ConfigMap | `flower-local-k8s-executor-config` |
| Default result | `local-k8s-launch-path` |
| Capacity-proof result | `local-k8s-capacity-cleanup-proof` |
| Default seeded runs | `1` |
| Capacity-proof seeded runs | `2` |
| Capacity-proof active Pod budget | `1` |
| Capacity-proof probe hold | `5.0` seconds |
| Demo seeded runs | `8` |
| Demo active Pod budget | `4` |
| Demo probe hold | `45` seconds |
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
| `harness.log` | Short harness result log. |
| `sanitized-config.yaml` | Sanitized copy of the selected harness profile. |
| `objects/capacity-blocked-pods.json` | Capacity-proof snapshot of the first active TaskExecutor Pod. |
| `objects/executor-config.yaml` | Rendered Kubernetes executor config, including capacity settings. |
| `objects/executor-config.json` | JSON form of the rendered Kubernetes executor config. |
| `objects/secrets-before-cleanup.redacted.json` | Capacity-proof redacted Secret snapshot before executor cleanup. |
| `objects/cleanup-pods.json` | Capacity-proof TaskExecutor Pod state after capacity opens. |
| `objects/secrets-after-cleanup.redacted.json` | Capacity-proof redacted Secret snapshot after executor cleanup. |
| `objects/real-launch.yaml` | Rendered SuperLink, executor config, and SuperExec objects. |
| `objects/seed-job.yaml` | Rendered seed ConfigMap and Job. |
| `objects/pods.json` | Observed TaskExecutor Pod list and phases. |
| `diagnostics/commands.txt` | Planned or executed host commands. |
| `diagnostics/failures.txt` | Failure messages when the harness records failures. |
| `diagnostics/image-preflight.json` | Docker image inspection and k3d import plan/results. |
| `diagnostics/image-preflight.txt` | Docker image inspection and k3d import command output. |
| `diagnostics/cleanup.json` | Namespace cleanup command plan/results. |
| `diagnostics/superexec-logs.txt` | Captured SuperExec logs used for claim and capacity-wait evidence. |
| `diagnostics/taskexecutor-logs.txt` | Captured TaskExecutor logs. |
| `diagnostics/cleanup.txt` | Cleanup defaults and the namespace delete command. |
| `diagnostics/verifier-output.txt` | Optional saved verifier report when rerun manually with shell redirection. |

## How The Evidence Proves Correctness

Use this section when reviewing an evidence directory without rerunning the
harness. The generated `proof-checklist.json` contains the same claim-to-file
map in machine-readable form.

1. Confirm the run was real and from the expected source tree.

   Open `invocation.json` and check:

   - `mode` is `local-k8s-launch-path` for the default proof or
     `local-k8s-capacity-cleanup-proof` for the capacity cleanup proof;
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

3. Confirm deterministic ServerApp tasks were seeded through AppIo.

   Open `objects/seed-job.yaml` and check that the Job runs
   `/opt/flower-local-k8s/seed_run.py` against the SuperLink Control API.
   Then check `summary.json` and `task-lineage.json`.

   For the default proof, `seed_run_id` and `seeded_run_id` should be present
   and should match. For the capacity cleanup proof, `summary.json` should list
   the expected `seed_run_ids`, `task-lineage.json` should list the same
   `seeded_run_ids`, and `seeded_task_count` should match the expected run
   count. The `--demo` preset expects three seeded runs.

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
   not claim completed Pod or Secret cleanup for the default one-task proof;
   that cleanup assertion belongs to the capacity cleanup proof.

8. Confirm the verifier accepted the bundle.

   The wrapper runs `framework/dev/k8s/verify_evidence.py` after the harness.
   A passing report should show `Verification: PASSED`, one TaskExecutor Pod,
   one lineage record, one credential Secret record, final state Pod/Secret
   counts, a `Succeeded` phase, and a successful cleanup command when cleanup
   was required.

For `--capacity-cleanup-proof`, additionally confirm:

1. `objects/executor-config.yaml` sets the selected `active-pod-budget`.
2. `summary.json` lists the expected `seed_run_ids`, and
   `task-lineage.json` records at least that many observed TaskExecutor task
   records.
3. `events.jsonl` has a passing `capacity.wait_observed` event. Also open
   `diagnostics/superexec-logs.txt`; it should include the specific SuperExec
   wait marker
   `waiting for kubernetes taskexecutor capacity`; `active pods` and `budget`
   are useful context, but they are not sufficient evidence on their own.
4. `summary.json` has `cleanup_observed.observed: true`, removed Pod and Secret
   names for the completed task, and at least one remaining/new TaskExecutor Pod
   after cleanup. Removed and remaining Pod names should be disjoint.
5. `objects/cleanup-pods.json` and
   `objects/secrets-after-cleanup.redacted.json` show the post-wait selector
   state before broad namespace cleanup.
6. The capacity verifier report should identify the result as
   `local-k8s-capacity-cleanup-proof`, show `Task lineage records: 2`, show
   `Capacity wait observed: True`, include non-empty removed Pod/Secret lines,
   and end with `Verification: PASSED`.

   In the budget-1/two-task mode, `TaskExecutor Pods: 1` in the verifier report
   is expected after cleanup: it is the remaining/new TaskExecutor Pod. The
   full task-count evidence comes from `Task lineage records` and
   `task-lineage.json`.

For `--demo`, additionally confirm:

1. `objects/executor-config.yaml` sets `active-pod-budget: 4`.
2. `summary.json` has `expected_seed_run_count: 8`,
   `active_pod_budget: 4`, and `cardinality.observed: true`.
3. `summary.json` lists four `cardinality.first_active_pods`, proving the
   budget was full before additional launches.
4. `summary.json` lists four `cardinality.launched_after_capacity_opened`
   entries, proving waiting TaskExecutors launched after capacity opened.
5. `diagnostics/superexec-logs.txt` includes the capacity wait marker with
   `4 active Pods` and `budget 4`.
6. `proof-checklist.json` does not list capacity cardinality as out of scope.

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
| Cardinality proof | Optional | `--demo` seeds eight runs with active Pod budget `4` and requires four active Pods before waiting work launches after slots open. |
| AppIo TLS | Optional | `--tls` configures local server-auth TLS and `verify_evidence.py --require-tls` checks the saved TLS evidence. |
| Wrapper cleanup | Yes | Default wrapper behavior deletes the namespace and verifies cleanup evidence. |

## Out Of Scope

| Area | Tested | Notes |
| --- | --- | --- |
| AppIo result completion semantics | No | This slice observes launch and Pod success, not full result semantics. |
| ClientApp execution | No | The probe includes a minimal ClientApp file only because the FAB schema expects it. |
| CNI/NetworkPolicy, production RBAC | No | This is local/dev-only and does not validate production deployment policy. |
| Concurrency, retry, failure behavior | No | The default proof starts one deterministic run; the capacity proof starts two deterministic runs only to exercise budget waiting and cleanup. |

## Useful Commands

Inspect resources after `--skip-cleanup`:

```bash
kubectl --context k3d-flower-local-k8s get pods -n flower-local-k8s
kubectl --context k3d-flower-local-k8s get jobs,secrets -n flower-local-k8s
kubectl --context k3d-flower-local-k8s logs pod/flower-superlink -n flower-local-k8s
kubectl --context k3d-flower-local-k8s logs pod/flower-superexec -n flower-local-k8s
```

Live demo watch commands:

```bash
watch -n 1 'kubectl get pods -n flower-local-k8s -o wide --sort-by=.metadata.creationTimestamp'
```

```bash
watch -n 1 'kubectl get pods -n flower-local-k8s -l app.kubernetes.io/component=taskexecutor -L flower.ai/resource-pool,flower.ai/superexec-task-id,flower.ai/launch-attempt --sort-by=.metadata.creationTimestamp'
```

```bash
kubectl logs -n flower-local-k8s -f pod/flower-superexec --tail=200
```

On macOS without `watch`, use a loop:

```bash
while true; do clear; date; kubectl get pods -n flower-local-k8s -o wide --sort-by=.metadata.creationTimestamp; sleep 1; done
```

Verify an existing default launch-path bundle:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}"
```

Verify an existing capacity cleanup bundle:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof
```

Verify a bundle from a run that used `--skip-cleanup`:

```bash
python framework/dev/k8s/verify_evidence.py "${output_dir}" \
  --expected-result local-k8s-capacity-cleanup-proof \
  --no-require-cleanup
```

Remove the namespace manually:

```bash
kubectl --context k3d-flower-local-k8s delete namespace flower-local-k8s \
  --ignore-not-found=true --wait=true
```

If Docker was restarted and an existing local k3d cluster appears stale, recreate
only the local harness cluster and rerun:

```bash
k3d cluster delete flower-local-k8s
./framework/dev/k8s/test-real-launch-path.sh \
  --capacity-cleanup-proof \
  --output-dir "${output_dir}"
```
