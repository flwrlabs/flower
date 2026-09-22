# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the real-process two-SuperNode MLCommons capability demo.

Run from ``framework/`` with:

``uv run --no-sync python -m dev.run_mlcommons_two_node_tls_auth_demo``

The demo generates a private CA, TLS leaf certificates, two P-384 SuperNode
authentication identities, a deterministic dependency-free FAB, and a matching
two-entry capability file. It then starts a local Guardian, one SuperLink, and two
SuperNodes as real child processes. The Control, Fleet, and Runtime connections all
use TLS; the two public keys are registered through the real TLS Control API before
the authenticated SuperNodes connect. Finally, the real ``flwr run
--capabilities-file`` command executes one query on each partition.

All state is written below a temporary artifact directory. Pass ``--keep-artifacts``
to retain its certificates, keys, database, app, and logs for inspection. The JSON
evidence printed on success contains hashes and process outcomes, but never private
keys or capability contents.
"""

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from flwr.cli.build import build_fab_from_disk
from flwr.common.capability import (
    CAPABILITY_FILE_VERSION,
    CAPABILITY_LOG_PREFIX,
    capability_binding,
    participant_id_from_public_key,
    safe_digest_prefix,
)
from flwr.supercore.constant import NOOP_FEDERATION_ID
from flwr.supercore.primitives.asymmetric import public_key_to_bytes

_HOST = "127.0.0.1"
_READY_TIMEOUT = 30.0
_RUN_TIMEOUT = 90.0
_STOP_TIMEOUT = 8.0
_RESULT_PREFIX = "MLCOMMONS_TWO_NODE_RESULT="
_FAILURE_PREFIX = "MLCOMMONS_TWO_NODE_FAILURE="
_LOG_TAIL_LIMIT = 2000
_SCENARIOS = ("allow", "guardian-deny", "binding-mismatch", "missing-capability")

_PYPROJECT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mlcommons-two-node-demo"
version = "0.1.0"
description = "Deterministic two-node capability demo"
license = {file = "LICENSE"}
dependencies = ["flwr>=1.36.0"]

[tool.hatch.build.targets.wheel]
packages = ["two_node_demo"]

[tool.flwr.app]
publisher = "mlcommons"
fab-format-version = 1
flwr-version-target = "1.36.0"

[tool.flwr.app.components]
serverapp = "two_node_demo.server_app:app"
clientapp = "two_node_demo.client_app:app"
"""

_SERVER_APP_TEMPLATE = f"""\
import json
import time

from flwr.app import Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp

app = ServerApp()
SCENARIO = "__SCENARIO__"


@app.main()
def main(grid: Grid, context) -> None:
    deadline = time.monotonic() + 30.0
    node_ids = []
    while len(node_ids) < 2 and time.monotonic() < deadline:
        node_ids = sorted(grid.get_node_ids())
        if len(node_ids) < 2:
            time.sleep(0.2)
    if len(node_ids) != 2:
        raise RuntimeError(f"expected two SuperNodes, found {{len(node_ids)}}")
    messages = [
        Message(
            content=RecordDict(),
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id="mlcommons-two-node",
        )
        for node_id in node_ids
    ]
    replies = grid.send_and_receive(messages, timeout=30.0)
    partitions = sorted(
        int(reply.content["result"]["partition-id"])
        for reply in replies
        if not reply.has_error()
    )
    rejection_reasons = sorted(
        reply.error.reason or "unspecified"
        for reply in replies
        if reply.has_error()
    )
    result = {{
        "node_count": len(node_ids),
        "partitions": partitions,
        "node_rejection_count": len(rejection_reasons),
        "node_rejection_reasons": rejection_reasons,
    }}
    if SCENARIO == "allow" and partitions != [0, 1]:
        raise RuntimeError(
            f"expected replies from partitions 0 and 1, got {{partitions}}"
        )
    if SCENARIO != "allow" and (
        partitions or len(rejection_reasons) != 2
    ):
        raise RuntimeError(
            f"expected two fail-closed replies for {{SCENARIO}}, got {{result}}"
        )
    print(
        "{_RESULT_PREFIX}"
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
"""

_CLIENT_APP = """\
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.query()
def query(message: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    content = RecordDict({"result": MetricRecord({"partition-id": partition_id})})
    return Message(content, reply_to=message)
"""


@dataclass
class _OwnedProcess:
    """One child process and its private process group."""

    name: str
    process: subprocess.Popen[str]
    log_path: Path
    log_file: Any


class _ProcessManager:
    """Start and stop only processes owned by this demo."""

    def __init__(self, logs_dir: Path, env: dict[str, str]) -> None:
        self.logs_dir = logs_dir
        self.env = env
        self.processes: dict[str, _OwnedProcess] = {}

    def start(self, name: str, command: list[str]) -> _OwnedProcess:
        """Start a child in its own process group and redirect its output."""
        log_path = self.logs_dir / f"{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            env=self.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        owned = _OwnedProcess(name, process, log_path, log_file)
        self.processes[name] = owned
        return owned

    def stop(self, name: str) -> bool:
        """Terminate a child process group and return whether all of it is dead."""
        owned = self.processes.get(name)
        if owned is None:
            return True
        process = owned.process
        process_group_id = process.pid
        if not _process_group_is_dead(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + _STOP_TIMEOUT
            while time.monotonic() < deadline and not _process_group_is_dead(
                process_group_id
            ):
                time.sleep(0.05)
            if not _process_group_is_dead(process_group_id):
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        owned.log_file.close()
        deadline = time.monotonic() + _STOP_TIMEOUT
        while time.monotonic() < deadline and not _process_group_is_dead(
            process_group_id
        ):
            time.sleep(0.05)
        return process.poll() is not None and _process_group_is_dead(process_group_id)

    def cleanup(self) -> dict[str, bool]:
        """Stop SuperNodes first, then Guardian and SuperLink."""
        outcomes: dict[str, bool] = {}
        for name in ("supernode-1", "supernode-0", "guardian", "superlink"):
            outcomes[name] = self.stop(name)
        return outcomes

    def log_tails(self, redactions: list[str]) -> dict[str, str]:
        """Return concise process-log tails with capability values redacted."""
        tails = {}
        for name, owned in self.processes.items():
            if not owned.log_file.closed:
                owned.log_file.flush()
            tail = owned.log_path.read_text(encoding="utf-8", errors="replace")[
                -_LOG_TAIL_LIMIT:
            ]
            tails[name] = _sanitize_log_text(tail, redactions)
        return tails

    def sanitize_logs(self, redactions: list[str]) -> None:
        """Remove capability values and full hexadecimal hashes from saved logs."""
        for owned in self.processes.values():
            if owned.log_path.exists():
                text = owned.log_path.read_text(encoding="utf-8", errors="replace")
                owned.log_path.write_text(
                    _sanitize_log_text(text, redactions), encoding="utf-8"
                )


def _sanitize_log_text(text: str, redactions: list[str]) -> str:
    """Redact packages and full digest-shaped values from human-readable logs."""
    for value in redactions:
        text = text.replace(value, "<redacted-capability>")
    return re.sub(r"\b[0-9a-fA-F]{64}\b", "<redacted-hash>", text)


def _allocate_ports(count: int) -> list[int]:
    """Ask the kernel for distinct loopback ports with a short reservation."""
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((_HOST, 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _process_group_is_dead(process_group_id: int) -> bool:
    """Return whether an owned process group has no surviving processes."""
    try:
        os.killpg(process_group_id, 0)
    except (PermissionError, ProcessLookupError):
        # macOS sandboxed processes report EPERM after an owned group disappears.
        return True
    return False


def _write_tls_material(root: Path) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    """Generate a private CA and one loopback server certificate per service."""
    certs_dir = root / "certs"
    certs_dir.mkdir()
    now = datetime.now(UTC)
    ca_key = ec.generate_private_key(ec.SECP384R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Flower demo CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = certs_dir / "ca.pem"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    leaves: dict[str, tuple[Path, Path]] = {}
    for name in ("superlink", "supernode-0", "supernode-1"):
        key = ec.generate_private_key(ec.SECP384R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address(_HOST)),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        cert_path = certs_dir / f"{name}.pem"
        key_path = certs_dir / f"{name}.key"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        leaves[name] = (cert_path, key_path)
    return ca_path, leaves


def _write_supernode_identities(root: Path) -> list[dict[str, str]]:
    """Generate two P-384 OpenSSH identities and their canonical fingerprints."""
    keys_dir = root / "keys"
    keys_dir.mkdir()
    identities = []
    for partition_id in range(2):
        private_key = ec.generate_private_key(ec.SECP384R1())
        public_key = private_key.public_key()
        private_path = keys_dir / f"supernode-{partition_id}"
        public_path = keys_dir / f"supernode-{partition_id}.pub"
        private_path.write_bytes(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.OpenSSH,
                serialization.NoEncryption(),
            )
        )
        private_path.chmod(0o600)
        public_path.write_bytes(
            public_key.public_bytes(
                serialization.Encoding.OpenSSH,
                serialization.PublicFormat.OpenSSH,
            )
            + b"\n"
        )
        participant_id = participant_id_from_public_key(public_key_to_bytes(public_key))
        identities.append(
            {
                "partition_id": str(partition_id),
                "private_path": str(private_path),
                "public_path": str(public_path),
                "participant_id": participant_id,
            }
        )
    return identities


def _write_app(root: Path, scenario: str) -> Path:
    """Write the deterministic dependency-free two-partition Flower App."""
    app_dir = root / "app"
    package_dir = app_dir / "two_node_demo"
    package_dir.mkdir(parents=True)
    (app_dir / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (app_dir / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    server_app = _SERVER_APP_TEMPLATE.replace("__SCENARIO__", scenario)
    (package_dir / "server_app.py").write_text(server_app, encoding="utf-8")
    (package_dir / "client_app.py").write_text(_CLIENT_APP, encoding="utf-8")
    return app_dir


def _capabilities_for_scenario(
    scenario: str,
    identities: list[dict[str, str]],
    binding: str,
    fab_hash: str,
) -> tuple[dict[str, str], list[str]]:
    """Return scenario packages and the raw values that logs must redact."""
    if scenario == "allow":
        package = binding
        capabilities = {identity["participant_id"]: package for identity in identities}
    elif scenario == "guardian-deny":
        package = f"deny:{binding}"
        capabilities = {identity["participant_id"]: package for identity in identities}
    elif scenario == "binding-mismatch":
        different_hash = "0" * 64 if fab_hash != "0" * 64 else "1" * 64
        package = capability_binding(NOOP_FEDERATION_ID, different_hash)
        capabilities = {identity["participant_id"]: package for identity in identities}
    elif scenario == "missing-capability":
        unrelated_participant = "flwr-p384-spki-pem-sha256:" + "0" * 64
        if unrelated_participant in {
            identity["participant_id"] for identity in identities
        }:
            unrelated_participant = "flwr-p384-spki-pem-sha256:" + "1" * 64
        package = binding
        capabilities = {unrelated_participant: package}
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    return capabilities, [binding, package]


def _read_process_log(manager: _ProcessManager, name: str) -> str:
    """Read one complete child-process log after cleanup."""
    owned = manager.processes.get(name)
    if owned is None or not owned.log_path.exists():
        return ""
    return owned.log_path.read_text(encoding="utf-8", errors="replace")


def _collect_observations(manager: _ProcessManager) -> dict[str, object]:
    """Derive machine-readable lifecycle evidence from real process logs."""
    node_logs = "\n".join(
        _read_process_log(manager, f"supernode-{partition_id}")
        for partition_id in range(2)
    )
    guardian_log = _read_process_log(manager, "guardian")
    reasons = sorted(
        match.strip()
        for match in re.findall(r"verification=denied[^\n]*?reason=([^\n]+)", node_logs)
    )
    return {
        "observed_node_rejection_count": node_logs.count("verification=denied"),
        "observed_node_rejection_reasons": reasons,
        "fab_request_count": node_logs.count("fab_retrieval=requested"),
        "clientapp_task_start_count": node_logs.count("task_creation=started"),
        "guardian_request_count": guardian_log.count("GuardianMock verify"),
        "binding_mismatch_count": node_logs.count("match=false"),
    }


def _write_control_config(flwr_home: Path, control_port: int, ca_path: Path) -> None:
    """Configure the real CLI to use the TLS Control API."""
    flwr_home.mkdir()
    (flwr_home / "config.toml").write_text(
        "[superlink]\n"
        'default = "demo"\n\n'
        "[superlink.demo]\n"
        f'address = "{_HOST}:{control_port}"\n'
        f'root-certificates = "{ca_path}"\n'
        "insecure = false\n",
        encoding="utf-8",
    )


def _demo_env(
    base_env: Mapping[str, str], flwr_home: Path, guardian_port: int
) -> dict[str, str]:
    """Return the isolated subprocess environment for this deployment."""
    env = dict(base_env)
    env.update(
        {
            "FLWR_HOME": str(flwr_home),
            "FLWR_GUARDIAN_URL": f"http://{_HOST}:{guardian_port}",
            "FLWR_TELEMETRY_ENABLED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _superlink_command(
    executable: str,
    *,
    control_port: int,
    fleet_port: int,
    database_path: Path,
    ca_path: Path,
    cert_path: Path,
    key_path: Path,
) -> list[str]:
    """Build the TLS/auth SuperLink command."""
    return [
        executable,
        "--host",
        _HOST,
        "--port",
        str(control_port),
        "--fleet-api-address",
        f"{_HOST}:{fleet_port}",
        "--database",
        str(database_path),
        "--ssl-ca-certfile",
        str(ca_path),
        "--ssl-certfile",
        str(cert_path),
        "--ssl-keyfile",
        str(key_path),
        "--enable-supernode-auth",
        "--disable-runtime-dependency-installation",
    ]


def _supernode_command(
    executable: str,
    *,
    fleet_port: int,
    runtime_port: int,
    partition_id: str,
    private_key_path: str,
    ca_path: Path,
    cert_path: Path,
    key_path: Path,
) -> list[str]:
    """Build one authenticated TLS SuperNode command."""
    return [
        executable,
        "--superlink",
        f"{_HOST}:{fleet_port}",
        "--root-certificates",
        str(ca_path),
        "--auth-supernode-private-key",
        private_key_path,
        "--node-config",
        f"partition-id={partition_id} num-partitions=2",
        "--host",
        _HOST,
        "--port",
        str(runtime_port),
        "--ssl-ca-certfile",
        str(ca_path),
        "--ssl-certfile",
        str(cert_path),
        "--ssl-keyfile",
        str(key_path),
        "--max-wait-time",
        str(_RUN_TIMEOUT),
    ]


def _run_command(executable: str, app_dir: Path, capabilities_path: Path) -> list[str]:
    """Build the real capability-bearing CLI run command."""
    return [
        executable,
        "run",
        str(app_dir),
        "demo",
        "--capabilities-file",
        str(capabilities_path),
        "--stream",
    ]


def _orchestration_event(log_path: Path, event: str) -> None:
    """Persist and print one sanitized orchestration event."""
    line = f"{CAPABILITY_LOG_PREFIX} Demo {event}"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")
    print(line, flush=True)


def _wait_for_port(
    port: int,
    *,
    timeout: float,
    process: _OwnedProcess,
    ca_path: Path | None = None,
) -> None:
    """Wait for a TCP or TLS listener while detecting early child exit."""
    deadline = time.monotonic() + timeout
    context = ssl.create_default_context(cafile=str(ca_path)) if ca_path else None
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            tail = process.log_path.read_text(encoding="utf-8")[-4000:]
            raise RuntimeError(f"{process.name} exited during readiness:\n{tail}")
        try:
            with socket.create_connection((_HOST, port), timeout=0.3) as connection:
                if context:
                    with context.wrap_socket(connection, server_hostname=_HOST):
                        return
                return
        except (OSError, ssl.SSLError):
            time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {process.name} on {_HOST}:{port}")


def _run_checked(
    command: list[str], env: dict[str, str], *, timeout: float = _READY_TIMEOUT
) -> str:
    """Run one bounded foreground command and return combined output."""
    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        rendered_command = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered_command}\n{result.stdout}"
        )
    return result.stdout


def _json_object_from_output(output: str) -> dict[str, Any]:
    """Extract the outer JSON object emitted by a Flower CLI command."""
    decoder = json.JSONDecoder()
    candidates = [index for index, char in enumerate(output) if char == "{"]
    for index in candidates:
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"CLI output did not contain a JSON object: {output}")


def _wait_for_nodes(flwr: str, env: dict[str, str], timeout: float) -> list[str]:
    """Wait until the TLS Control API reports exactly two online SuperNodes."""
    deadline = time.monotonic() + timeout
    last_output = ""
    while time.monotonic() < deadline:
        try:
            last_output = _run_checked(
                [flwr, "supernode", "list", "demo", "--format", "json"],
                env,
                timeout=5.0,
            )
            document = _json_object_from_output(last_output)
            online = [
                str(node["node-id"])
                for node in document.get("nodes", [])
                if node.get("status") == "online"
            ]
            if len(online) == 2:
                return sorted(online)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(0.2)
    raise TimeoutError(
        f"timed out waiting for two online nodes; last output: {last_output}"
    )


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"required executable '{name}' was not found; run through the framework "
            "environment with `uv run --no-sync`"
        )
    return executable


def run_demo(
    root: Path, scenario: str = "allow"
) -> dict[str, object]:  # pylint: disable=too-many-locals
    """Run the deployment and return secret-free machine-readable evidence."""
    if scenario not in _SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir()
    orchestration_log = logs_dir / "orchestration.log"
    flwr_home = root / "flwr-home"
    database_path = root / "state.db"
    guardian_port, control_port, fleet_port, node0_port, node1_port = _allocate_ports(5)
    ca_path, leaves = _write_tls_material(root)
    identities = _write_supernode_identities(root)
    app_dir = _write_app(root, scenario)
    fab_bytes = build_fab_from_disk(app_dir)
    if fab_bytes != build_fab_from_disk(app_dir):
        raise RuntimeError("tiny FAB build is not deterministic")
    fab_hash = hashlib.sha256(fab_bytes).hexdigest()
    binding = capability_binding(NOOP_FEDERATION_ID, fab_hash)
    capabilities, redactions = _capabilities_for_scenario(
        scenario, identities, binding, fab_hash
    )
    capabilities_path = root / "capabilities.json"
    capabilities_path.write_text(
        json.dumps(
            {
                "version": CAPABILITY_FILE_VERSION,
                "capabilities": capabilities,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    capabilities_hash = hashlib.sha256(capabilities_path.read_bytes()).hexdigest()
    _orchestration_event(
        orchestration_log,
        f"scenario={scenario} capabilities_file="
        f"{safe_digest_prefix(capabilities_hash)} entries={len(capabilities)}",
    )
    _write_control_config(flwr_home, control_port, ca_path)

    env = _demo_env(os.environ, flwr_home, guardian_port)
    python = _required_executable("python")
    flwr = _required_executable("flwr")
    superlink = _required_executable("flower-superlink")
    supernode = _required_executable("flower-supernode")
    manager = _ProcessManager(logs_dir, env)
    cleanup: dict[str, bool] = {}
    evidence: dict[str, object] = {}
    failure: Exception | None = None
    try:
        guardian = manager.start(
            "guardian",
            [
                python,
                "-m",
                "flwr.supernode.guardian_mock",
                "--host",
                _HOST,
                "--port",
                str(guardian_port),
            ],
        )
        _wait_for_port(guardian_port, timeout=_READY_TIMEOUT, process=guardian)

        link_cert, link_key = leaves["superlink"]
        link = manager.start(
            "superlink",
            _superlink_command(
                superlink,
                control_port=control_port,
                fleet_port=fleet_port,
                database_path=database_path,
                ca_path=ca_path,
                cert_path=link_cert,
                key_path=link_key,
            ),
        )
        _wait_for_port(
            control_port,
            timeout=_READY_TIMEOUT,
            process=link,
            ca_path=ca_path,
        )
        _wait_for_port(
            fleet_port,
            timeout=_READY_TIMEOUT,
            process=link,
            ca_path=ca_path,
        )

        registered_node_ids = []
        for identity in identities:
            output = _run_checked(
                [
                    flwr,
                    "supernode",
                    "register",
                    identity["public_path"],
                    "demo",
                    "--format",
                    "json",
                ],
                env,
            )
            registered_node_ids.append(str(_json_object_from_output(output)["node-id"]))
            participant_prefix = safe_digest_prefix(identity["participant_id"])
            _orchestration_event(
                orchestration_log,
                f"registered participant={participant_prefix} "
                f"node_id={registered_node_ids[-1]} "
                f"partition={identity['partition_id']}",
            )

        node_ports = [node0_port, node1_port]
        for identity, port in zip(identities, node_ports, strict=True):
            partition_id = identity["partition_id"]
            cert_path, key_path = leaves[f"supernode-{partition_id}"]
            node = manager.start(
                f"supernode-{partition_id}",
                _supernode_command(
                    supernode,
                    fleet_port=fleet_port,
                    runtime_port=port,
                    partition_id=partition_id,
                    private_key_path=identity["private_path"],
                    ca_path=ca_path,
                    cert_path=cert_path,
                    key_path=key_path,
                ),
            )
            _wait_for_port(port, timeout=_READY_TIMEOUT, process=node, ca_path=ca_path)

        online_node_ids = _wait_for_nodes(flwr, env, _READY_TIMEOUT)
        run_output = _run_checked(
            _run_command(flwr, app_dir, capabilities_path),
            env,
            timeout=_RUN_TIMEOUT,
        )
        result_line = next(
            (line for line in run_output.splitlines() if _RESULT_PREFIX in line), None
        )
        if result_line is None:
            raise RuntimeError(f"run output lacked partition evidence:\n{run_output}")
        result = json.loads(result_line.split(_RESULT_PREFIX, maxsplit=1)[1])
        expected_result = {
            "node_count": 2,
            "partitions": [0, 1] if scenario == "allow" else [],
            "node_rejection_count": 0 if scenario == "allow" else 2,
            "node_rejection_reasons": (
                []
                if scenario == "allow"
                else ["The run capability could not be verified."] * 2
            ),
        }
        if result != expected_result:
            raise RuntimeError(
                f"unexpected app result for scenario {scenario}: {result}"
            )
        _orchestration_event(
            orchestration_log,
            f"scenario={scenario} completed partitions={result['partitions']} "
            f"rejections={result['node_rejection_count']}",
        )
        evidence = {
            "artifact_root": str(root),
            "scenario": scenario,
            "expected_outcome": "allow" if scenario == "allow" else "fail-closed",
            "fab_hash": fab_hash,
            "capabilities_file_sha256": capabilities_hash,
            "participant_ids": sorted(
                identity["participant_id"] for identity in identities
            ),
            "registered_node_ids": sorted(registered_node_ids),
            "online_node_ids": online_node_ids,
            "tls": {"control": True, "fleet": True, "runtime": True},
            "supernode_auth": True,
            "cli_option": "--capabilities-file",
            "result": result,
        }
    except Exception as err:  # pylint: disable=broad-exception-caught
        failure = err
    finally:
        cleanup = manager.cleanup()
        _orchestration_event(
            orchestration_log,
            "cleanup "
            + " ".join(f"{name}={outcome}" for name, outcome in cleanup.items()),
        )
    if failure is not None:
        diagnostics = {
            "cleanup": cleanup,
            "log_tails": manager.log_tails(redactions),
        }
        manager.sanitize_logs(redactions)
        diagnostic_text = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(
            f"deployment failed: {failure}\n{_FAILURE_PREFIX}{diagnostic_text}"
        ) from failure
    if not all(cleanup.values()):
        raise RuntimeError(f"owned process cleanup failed: {cleanup}")
    observations = _collect_observations(manager)
    manager.sanitize_logs(redactions)
    expected_count = 0 if scenario == "allow" else 2
    expected_guardian_requests = 0 if scenario == "missing-capability" else 2
    expected_fab_requests = 2 if scenario == "allow" else 0
    expected_task_starts = 2 if scenario == "allow" else 0
    if observations["observed_node_rejection_count"] != expected_count:
        raise RuntimeError(f"unexpected rejection evidence: {observations}")
    if observations["guardian_request_count"] != expected_guardian_requests:
        raise RuntimeError(f"unexpected Guardian request evidence: {observations}")
    if observations["fab_request_count"] != expected_fab_requests:
        raise RuntimeError(f"unexpected FAB request evidence: {observations}")
    if observations["clientapp_task_start_count"] != expected_task_starts:
        raise RuntimeError(f"unexpected task-start evidence: {observations}")
    if scenario == "binding-mismatch" and observations["binding_mismatch_count"] != 2:
        raise RuntimeError(f"missing binding mismatch evidence: {observations}")
    evidence.update(observations)
    evidence["cleanup"] = cleanup
    return evidence


def main() -> None:
    """Parse arguments, run the demo, and print its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Retain generated state and logs instead of deleting them.",
    )
    parser.add_argument(
        "--scenario",
        choices=_SCENARIOS,
        default="allow",
        help="Run the successful path or one expected fail-closed scenario.",
    )
    args = parser.parse_args()
    if args.keep_artifacts:
        root = Path(tempfile.mkdtemp(prefix="flwr-mlcommons-two-node-"))
        print(json.dumps(run_demo(root, args.scenario), indent=2, sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix="flwr-mlcommons-two-node-") as tmp:
        print(json.dumps(run_demo(Path(tmp), args.scenario), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
