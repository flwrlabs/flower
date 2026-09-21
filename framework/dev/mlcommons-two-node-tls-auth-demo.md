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

The harness creates a deterministic dependency-free FAB, derives each participant ID
from the canonical P-384 public key representation, and writes a matching two-entry
capabilities file. It invokes the real CLI with `flwr run --capabilities-file`, then
requires replies from partitions 0 and 1. All ports are kernel-selected loopback ports,
all waits and foreground commands have deadlines, and teardown stops SuperNodes before
the Guardian and SuperLink and verifies every owned process group is gone.

The final JSON is safe to retain: it reports participant IDs, hashes, node IDs, TLS/auth
booleans, partition results, and cleanup outcomes, but no private keys or capability
contents. To inspect the generated configuration, certificates, keys, database, app,
and individual process logs, add `--keep-artifacts`; the output identifies the retained
artifact directory. Without that option, the directory is deleted after the evidence is
printed.

Trust-flow events use the stable `[CAPABILITY]` marker. Human-readable hash fields show
only their first 12 hexadecimal characters; canonical values in state, protocol
messages, and final JSON evidence remain unchanged. `logs/orchestration.log` records
registration mappings, the capability-file hash prefix, final partitions, and cleanup,
while the four process logs record StartRun, per-node routing, Guardian verification,
and the SuperNode fail-closed lifecycle.
