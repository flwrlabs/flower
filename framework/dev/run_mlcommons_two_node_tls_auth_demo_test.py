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
"""Tests for the real-process two-SuperNode MLCommons demo."""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import dev.run_mlcommons_two_node_tls_auth_demo as demo_module
from dev.run_mlcommons_two_node_tls_auth_demo import (
    _FAILURE_PREFIX,
    _allocate_ports,
    _capabilities_for_scenario,
    _classify_outcome,
    _demo_env,
    _json_object_from_output,
    _orchestration_event,
    _ordered_process_story,
    _ProcessManager,
    _run_command,
    _sanitize_log_text,
    _story_event,
    _superlink_command,
    _supernode_command,
    _wait_for_port,
    _write_control_config,
    _write_supernode_identities,
    _write_tls_material,
    run_demo,
)


def _evidence(**overrides: object) -> dict[str, object]:
    """Build minimal collected evidence for outcome classification tests."""
    evidence: dict[str, object] = {
        "scenario": "guardian-deny",
        "result": {"partitions": []},
        "observed_node_rejection_count": 2,
        "observed_node_rejection_reasons": [],
        "guardian_request_count": 0,
        "fab_request_count": 0,
        "clientapp_task_start_count": 0,
        "binding_mismatch_count": 0,
        "binding_match_count": 0,
        "capability_missing_count": 0,
    }
    evidence.update(overrides)
    return evidence


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            _evidence(
                result={"partitions": [0, 1]},
                observed_node_rejection_count=0,
                guardian_request_count=2,
                binding_match_count=2,
                fab_request_count=2,
                clientapp_task_start_count=2,
            ),
            "OUTCOME: AUTHORIZED — 2/2 SuperNodes executed; partitions=[0,1]",
        ),
        (
            _evidence(
                observed_node_rejection_reasons=["Guardian denied the capability"] * 2,
                guardian_request_count=2,
            ),
            "OUTCOME: BLOCKED BY GUARDIAN",
        ),
        (
            _evidence(binding_mismatch_count=2, guardian_request_count=2),
            "OUTCOME: BLOCKED BY SUPERNODE FED/FAB BINDING CHECK",
        ),
        (
            _evidence(capability_missing_count=2),
            "OUTCOME: BLOCKED BEFORE GUARDIAN",
        ),
    ],
)
def test_outcome_classification_uses_evidence_not_scenario(
    evidence: dict[str, object], expected: str
) -> None:
    """Classify from observations even when the scenario label is misleading."""
    outcome, valid = _classify_outcome(evidence)

    assert valid
    assert outcome.startswith(expected)


def test_unexpected_evidence_is_reported_as_failure() -> None:
    """Reject evidence that does not describe one internally consistent outcome."""
    outcome, valid = _classify_outcome(
        _evidence(observed_node_rejection_count=1, capability_missing_count=1)
    )

    assert not valid
    assert outcome == (
        "OUTCOME: UNEXPECTED — evidence did not describe a valid demo outcome"
    )


@pytest.mark.parametrize(
    "missing_evidence",
    ["guardian_request_count", "binding_match_count"],
)
def test_authorized_requires_guardian_and_binding_match_evidence(
    missing_evidence: str,
) -> None:
    """Do not authorize from partitions/FAB/tasks without trust evidence."""
    evidence = _evidence(
        result={"partitions": [0, 1]},
        observed_node_rejection_count=0,
        guardian_request_count=2,
        binding_match_count=2,
        fab_request_count=2,
        clientapp_task_start_count=2,
    )
    evidence[missing_evidence] = 0

    outcome, valid = _classify_outcome(evidence)

    assert not valid
    assert outcome.startswith("OUTCOME: UNEXPECTED")


def test_component_story_is_ordered_by_decision_causality(tmp_path: Path) -> None:
    """Present component-owned events in stable causal order."""
    logs = tmp_path / "logs"
    logs.mkdir()
    contents = {
        "superlink": "\n".join(
            [
                "[STORY] Capability selected: node_id=7 participant=abc",
                (
                    "[STORY] Run accepted: run_id=1 fab_hash=aaa "
                    "fed_fab_binding=bbb participants=1"
                ),
            ]
        ),
        "supernode-0": "\n".join(
            [
                "[STORY] Execution authorized: FAB requested; ClientApp task started",
                (
                    "[STORY] Execution gated: node_id=7 "
                    "task_creation=blocked fab_retrieval=blocked"
                ),
                (
                    "[STORY] SuperNode fed/FAB binding check: "
                    "BINDING MATCHES expected=bbb returned=bbb"
                ),
            ]
        ),
        "guardian": "[STORY] Guardian decision: ALLOW returned_fed_fab_binding=bbb",
    }
    manager = MagicMock()
    manager.processes = {}
    for name, content in contents.items():
        path = logs / f"{name}.log"
        path.write_text(content, encoding="utf-8")
        manager.processes[name] = MagicMock(log_path=path)

    events = _ordered_process_story(manager)

    assert [event.split(":", maxsplit=1)[0] for event in events] == [
        "Run accepted",
        "Capability selected",
        "Execution gated",
        "Guardian decision",
        "SuperNode fed/FAB binding check",
        "Execution authorized",
    ]


def test_capability_scenarios_route_expected_packages() -> None:
    """Build allow, denial, mismatch, and missing-participant fixtures."""
    identities = [
        {"participant_id": "flwr-p384-spki-pem-sha256:" + digit * 64}
        for digit in ("a", "b")
    ]
    binding = "flwr-capability-binding-v1-sha256:" + "c" * 64
    fab_hash = "d" * 64

    allow, _ = _capabilities_for_scenario("allow", identities, binding, fab_hash)
    denial, _ = _capabilities_for_scenario(
        "guardian-deny", identities, binding, fab_hash
    )
    mismatch, _ = _capabilities_for_scenario(
        "binding-mismatch", identities, binding, fab_hash
    )
    missing, _ = _capabilities_for_scenario(
        "missing-capability", identities, binding, fab_hash
    )

    participant_ids = {identity["participant_id"] for identity in identities}
    assert allow == dict.fromkeys(participant_ids, binding)
    assert denial == dict.fromkeys(participant_ids, f"deny:{binding}")
    assert set(mismatch) == participant_ids
    assert len(set(mismatch.values())) == 1
    assert next(iter(mismatch.values())) != binding
    assert set(missing).isdisjoint(participant_ids)
    assert list(missing.values()) == [binding]


def test_human_log_sanitization_removes_packages_and_full_hashes() -> None:
    """Retain causal log text without capability material or full digests."""
    package = "deny:flwr-capability-binding-v1-sha256:" + "a" * 64
    text = f"package={package} message_id={'b' * 64} match=false"

    sanitized = _sanitize_log_text(text, [package])

    assert package not in sanitized
    assert "a" * 64 not in sanitized
    assert "b" * 64 not in sanitized
    assert "match=false" in sanitized


def test_generated_identities_are_distinct_and_private(tmp_path: Path) -> None:
    """Generate distinct P-384 identities without exposing private material."""
    identities = _write_supernode_identities(tmp_path)

    assert len({identity["participant_id"] for identity in identities}) == 2
    assert {identity["partition_id"] for identity in identities} == {"0", "1"}
    for identity in identities:
        private_path = Path(identity["private_path"])
        assert private_path.stat().st_mode & 0o777 == 0o600
        assert "PRIVATE KEY" in private_path.read_text(encoding="utf-8")


def test_generated_tls_material_covers_each_service(tmp_path: Path) -> None:
    """Generate an isolated CA and leaf certificate for every TLS server."""
    ca_path, leaves = _write_tls_material(tmp_path)

    assert ca_path.is_file()
    assert set(leaves) == {"superlink", "supernode-0", "supernode-1"}
    assert all(path.is_file() for pair in leaves.values() for path in pair)


def test_allocated_ports_are_distinct() -> None:
    """Reserve a collision-resistant set of distinct loopback ports."""
    ports = _allocate_ports(5)

    assert len(ports) == len(set(ports)) == 5


def test_json_output_parser_keeps_outer_document() -> None:
    """Parse the CLI envelope instead of a nested node object."""
    document = _json_object_from_output(
        'status\n{"success":true,"nodes":[{"node-id":"7"}]}\n'
    )

    assert document == {"success": True, "nodes": [{"node-id": "7"}]}


def test_orchestration_event_is_prefixed_persisted_and_sanitized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep the demo trail useful without recording canonical secret values."""
    log_path = tmp_path / "orchestration.log"

    _orchestration_event(log_path, "completed partitions=[0, 1]")

    expected = "[CAPABILITY] Demo completed partitions=[0, 1]"
    assert expected in capsys.readouterr().out
    assert log_path.read_text(encoding="utf-8").strip() == expected


def test_story_event_is_emitted_live_and_persisted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Use the stable story prefix for terminal and orchestration output."""
    log_path = tmp_path / "orchestration.log"

    _story_event(log_path, "Guardian started: endpoint=http://127.0.0.1:1234")

    expected = "[STORY] Guardian started: endpoint=http://127.0.0.1:1234"
    assert capsys.readouterr().out.strip() == expected
    assert log_path.read_text(encoding="utf-8").strip() == expected


def test_readiness_times_out_for_missing_listener(tmp_path: Path) -> None:
    """Bound readiness when a live process never opens its expected port."""
    logs = tmp_path / "logs"
    logs.mkdir()
    manager = _ProcessManager(logs, os.environ.copy())
    process = manager.start(
        "guardian", [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    port = _allocate_ports(1)[0]
    try:
        with pytest.raises(TimeoutError):
            _wait_for_port(port, timeout=0.2, process=process)
    finally:
        assert manager.cleanup()["guardian"]
    assert process.process.poll() is not None


def test_cleanup_stops_every_owned_process_after_failure(tmp_path: Path) -> None:
    """Leave no child alive after a partial-start failure."""
    logs = tmp_path / "logs"
    logs.mkdir()
    manager = _ProcessManager(logs, os.environ.copy())
    guardian = manager.start(
        "guardian", [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    superlink = manager.start(
        "superlink", [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    time.sleep(0.05)

    outcomes = manager.cleanup()

    assert all(outcomes.values())
    assert guardian.process.poll() is not None
    assert superlink.process.poll() is not None


def test_cleanup_kills_descendants_in_owned_process_group(tmp_path: Path) -> None:
    """Stop an auto-launched descendant along with its owning process."""
    logs = tmp_path / "logs"
    logs.mkdir()
    manager = _ProcessManager(logs, os.environ.copy())
    parent = manager.start(
        "supernode-0",
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)']); "
                "time.sleep(30)"
            ),
        ],
    )
    time.sleep(0.1)

    assert manager.stop("supernode-0")
    assert parent.process.poll() is not None


def test_commands_configure_tls_auth_and_distinct_nodes(tmp_path: Path) -> None:
    """Build two distinct authenticated nodes and an explicit capability run."""
    ca_path = tmp_path / "ca.pem"
    link_command = _superlink_command(
        "flower-superlink",
        control_port=10001,
        fleet_port=10002,
        database_path=tmp_path / "state.db",
        ca_path=ca_path,
        cert_path=tmp_path / "link.pem",
        key_path=tmp_path / "link.key",
    )
    node_commands = [
        _supernode_command(
            "flower-supernode",
            fleet_port=10002,
            runtime_port=10003 + partition_id,
            partition_id=str(partition_id),
            private_key_path=str(tmp_path / f"node-{partition_id}"),
            ca_path=ca_path,
            cert_path=tmp_path / f"node-{partition_id}.pem",
            key_path=tmp_path / f"node-{partition_id}.key",
        )
        for partition_id in range(2)
    ]
    run_command = _run_command("flwr", tmp_path / "app", tmp_path / "capabilities.json")

    assert all(
        "--insecure" not in command for command in [link_command, *node_commands]
    )
    assert "--ssl-ca-certfile" in link_command
    assert "--root-certificates" in node_commands[0]
    assert "--ssl-ca-certfile" in node_commands[0]
    assert node_commands[0] != node_commands[1]
    assert node_commands[0][node_commands[0].index("--port") + 1] == "10003"
    assert node_commands[1][node_commands[1].index("--port") + 1] == "10004"
    assert "partition-id=0 num-partitions=2" in node_commands[0]
    assert "partition-id=1 num-partitions=2" in node_commands[1]
    assert str(tmp_path / "node-0") in node_commands[0]
    assert str(tmp_path / "node-1") in node_commands[1]
    assert run_command[-3:] == [
        "--capabilities-file",
        str(tmp_path / "capabilities.json"),
        "--stream",
    ]


def test_subprocess_environment_uses_isolated_flwr_home(tmp_path: Path) -> None:
    """Keep CLI configuration and runtime state under the artifact root."""
    base_env = {"PATH": "/test/bin", "FLWR_HOME": "/outside"}

    env = _demo_env(base_env, tmp_path / "flwr-home", 12345)

    assert env["FLWR_HOME"] == str(tmp_path / "flwr-home")
    assert env["FLWR_GUARDIAN_URL"] == "http://127.0.0.1:12345"
    assert env["FLWR_TELEMETRY_ENABLED"] == "0"
    assert env["PATH"] == "/test/bin"
    assert base_env["FLWR_HOME"] == "/outside"
    _write_control_config(tmp_path / "flwr-home", 12346, tmp_path / "ca.pem")
    config = (tmp_path / "flwr-home" / "config.toml").read_text(encoding="utf-8")
    assert 'address = "127.0.0.1:12346"' in config
    assert f'root-certificates = "{tmp_path / "ca.pem"}"' in config
    assert "insecure = false" in config


def test_partial_start_reports_logs_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve failure context, log diagnostics, and cleanup evidence."""
    monkeypatch.setattr(demo_module, "_READY_TIMEOUT", 1.0)
    monkeypatch.setattr(
        demo_module, "_required_executable", lambda _name: sys.executable
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_demo(tmp_path / "artifacts")

    assert exc_info.value.__cause__ is not None
    message = str(exc_info.value)
    assert "deployment failed:" in message
    assert _FAILURE_PREFIX in message
    diagnostics = _json_object_from_output(message.split(_FAILURE_PREFIX, 1)[1])
    assert all(diagnostics["cleanup"].values())
    assert "superlink" in diagnostics["log_tails"]
    assert "unknown option" in diagnostics["log_tails"]["superlink"].lower()
    assert "PRIVATE KEY" not in message
    assert "flwr-capability-binding-v1" not in message
    orchestration = (tmp_path / "artifacts/logs/orchestration.log").read_text(
        encoding="utf-8"
    )
    assert "[STORY] Guardian started:" in orchestration
    assert "[STORY] SuperLink started:" not in orchestration
    assert "[STORY] TLS verified: SuperLink" not in orchestration
    assert "[STORY] Topology ready:" not in orchestration
