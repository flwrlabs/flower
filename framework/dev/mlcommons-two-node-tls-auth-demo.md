# MLCommons two-SuperNode TLS/auth capability demo

This local-only demo starts a real Guardian HTTP process, one SuperLink, and two
distinct SuperNodes. Control, Fleet, and both Runtime API connections use a generated
private CA; the deployment never passes `--insecure`. Each SuperNode uses a generated
P-384 OpenSSH private key, and the corresponding public key is registered through the
real TLS Control API before the node starts.

From `framework/`, run:

```shell
uv run --no-sync python -m dev.run_mlcommons_two_node_tls_auth_demo
```

The default `--scenario allow` preserves the successful two-partition run. Three
bounded fail-closed demonstrations are also available:

```shell
uv run --no-sync python -m dev.run_mlcommons_two_node_tls_auth_demo --scenario guardian-deny
uv run --no-sync python -m dev.run_mlcommons_two_node_tls_auth_demo --scenario binding-mismatch
uv run --no-sync python -m dev.run_mlcommons_two_node_tls_auth_demo --scenario missing-capability
```

`guardian-deny` routes `deny:<correct fed/FAB binding>`. `binding-mismatch` routes
an allowed package bound to a deterministic, deliberately different FAB hash.
`missing-capability` keeps capability enforcement enabled with one unrelated
participant entry, so both authenticated SuperNodes receive an empty package and do
not call Guardian. Expected denials are successful demo outcomes: both nodes must
return a capability-verification error, with zero FAB requests and zero ClientApp
task starts.

The harness creates a deterministic dependency-free FAB, derives each participant ID
from the canonical P-384 public key representation, and writes the scenario-specific
capabilities file. It invokes the real CLI with `flwr run --capabilities-file`, then
requires either replies from partitions 0 and 1 or two expected fail-closed replies.
All ports are kernel-selected loopback ports, all waits and foreground commands have
deadlines, and teardown stops SuperNodes before the Guardian and SuperLink and verifies
every owned process group is gone.

The final JSON is safe to retain: it reports the selected scenario, evidence-derived
story outcome, participant IDs, hashes, node IDs, TLS/auth booleans, observed rejection
count and reasons, Guardian/fed-FAB-check/FAB/task-start counts, partition results, and
cleanup outcomes, but no private keys or capability contents. To inspect the generated
configuration, certificates, keys, database, app, and individual process logs, add
`--keep-artifacts`; the output identifies the retained artifact directory. Without that
option, the directory is deleted after the evidence is printed.

Trust-flow events use the stable `[CAPABILITY]` marker. Human-readable hash fields show
only their first 12 hexadecimal characters; canonical values in state, protocol
messages, and final JSON evidence remain unchanged. `logs/orchestration.log` records
registration mappings, scenario, capability-file hash prefix, final partitions or
rejections, and cleanup. The stable `[STORY]` layer is emitted only after readiness or
decision evidence exists: it presents topology/TLS/authentication live, then replays a
concise causal summary from the component-owned process events, followed by one
evidence-derived outcome and one confirmed-cleanup line. The process logs retain
StartRun, per-node routing, Guardian decisions, and the SuperNode fail-closed lifecycle.
On a cache hit, verification still blocks task creation while FAB retrieval remains
cached; on a cache miss, verification blocks both task creation and FAB retrieval.
