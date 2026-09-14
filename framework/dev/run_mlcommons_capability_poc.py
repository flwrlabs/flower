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
"""Run the deterministic MLCommons capability POC smoke harness.

Run from ``framework/`` with:

``uv run --no-sync python -m dev.run_mlcommons_capability_poc``

The harness creates a dependency-free tiny FAB, generates an authenticated P-384
SuperNode key and its canonical participant fingerprint, writes matching v1
capabilities files, and exercises two real production-code paths in process:

``flwr run`` CLI -> StartRunRequest -> Control handler/LinkState -> Fleet per-node
routing -> SuperNode lifecycle -> local Guardian HTTP mock.

The allowed scenario reaches FAB retrieval and ClientApp task creation. The denied
scenario is blocked before both. The only in-process adapter replaces the network
hop between the CLI stub and Control handler; request protobufs and all named Flower
layers are otherwise the production implementations.
"""

import hashlib
import importlib
import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from flwr.app import Metadata, RecordDict
from flwr.app.constants import DEFAULT_TTL
from flwr.app.message import make_message, remove_content_from_message
from flwr.cli.app import app as cli_app
from flwr.cli.build import build_fab_from_disk
from flwr.common.capability import (
    CAPABILITY_FILE_VERSION,
    capability_binding,
    participant_id_from_public_key,
)
from flwr.common.constant import (
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    SUPERLINK_NODE_ID,
    TRANSPORT_TYPE_GRPC_RERE,
)
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.server.superlink.fleet.message_handler.message_handler import get_run
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import NOOP_FEDERATION_ID
from flwr.supercore.date import now
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
)
from flwr.supercore.object_store.in_memory_object_store import InMemoryObjectStore
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.supercore.run import Run
from flwr.superlink.federation.noop_federation_manager import NoOpFederationManager
from flwr.superlink.servicer.control.control_handlers import start_run
from flwr.supernode.guardian_mock import GuardianMockHandler
from flwr.supernode.nodestate.in_memory_nodestate import InMemoryNodeState
from flwr.supernode.start_client_internal import _pull_and_store_message

run_module = importlib.import_module("flwr.cli.run.run")

_PYPROJECT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mlcommons-capability-poc"
version = "0.1.0"
description = "Deterministic capability POC fixture"
license = {file = "LICENSE"}
dependencies = ["flwr>=1.36.0"]

[tool.hatch.build.targets.wheel]
packages = ["capability_poc"]

[tool.flwr.app]
publisher = "mlcommons"
fab-format-version = 1
flwr-version-target = "1.36.0"

[tool.flwr.app.components]
serverapp = "capability_poc.server_app:app"
clientapp = "capability_poc.client_app:app"

[tool.flwr.app.config]
fixture-value = 7
"""


class _ControlClient:
    """Bridge the CLI's real StartRunRequest to the real Control handler."""

    def __init__(self, state: InMemoryLinkState) -> None:
        self.state = state
        self.last_run_id = 0

    def StartRun(  # pylint: disable=invalid-name
        self, request: StartRunRequest
    ) -> StartRunResponse:
        """Submit one run through the production Control handler."""
        response = start_run(
            request,
            AccountInfo(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME),
            self.state,
            fleet_api_type=TRANSPORT_TYPE_GRPC_RERE,
        )
        self.last_run_id = response.run_id
        return response

    def close(self) -> None:
        """Close the in-process client."""


def _write_tiny_app(root: Path) -> Path:
    """Write a deterministic dependency-free FAB fixture."""
    app_dir = root / "tiny-app"
    package_dir = app_dir / "capability_poc"
    package_dir.mkdir(parents=True)
    (app_dir / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (app_dir / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    app_source = "from flwr.app import ClientApp\n\napp = ClientApp()\n"
    (package_dir / "client_app.py").write_text(app_source, encoding="utf-8")
    (package_dir / "server_app.py").write_text(
        "from flwr.app import ServerApp\n\napp = ServerApp()\n", encoding="utf-8"
    )
    return app_dir


def _submit_run(
    app_dir: Path,
    capabilities_file: Path,
    state: InMemoryLinkState,
) -> int:
    """Invoke the real CLI and return the created run ID."""
    control_client = _ControlClient(state)
    connection = SimpleNamespace(
        federation=NOOP_FEDERATION_ID,
        name="in-process",
        address="unused",
        options=None,
    )
    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch.object(run_module, "read_superlink_connection", return_value=connection),
        patch.object(
            run_module,
            "init_http_client_from_connection",
            return_value=control_client,
        ),
    ):
        result = CliRunner().invoke(
            cli_app,
            [
                "run",
                str(app_dir),
                "--capabilities-file",
                str(capabilities_file),
            ],
        )
    if result.exit_code != 0:
        raise RuntimeError(f"flwr run failed: {result.output}") from result.exception
    if control_client.last_run_id == 0:
        raise RuntimeError("flwr run did not create a run")
    return control_client.last_run_id


def _transition_to_running(state: InMemoryLinkState, run_id: int) -> None:
    """Advance the primary task so Fleet permits GetRun."""
    run = state.get_run_info(run_ids=[run_id])[0]
    if run.primary_task_id is None:
        raise RuntimeError("run has no primary task")
    if state.claim_task(run.primary_task_id) is None:
        raise RuntimeError("could not claim primary task")
    if not state.activate_task(run.primary_task_id):
        raise RuntimeError("could not activate primary task")


def _exercise_supernode(
    state: InMemoryLinkState,
    node_id: int,
    run_id: int,
) -> dict[str, object]:
    """Route the capability and execute the pre-ClientApp SuperNode lifecycle."""
    routed = get_run(
        GetRunRequest(node=Node(node_id=node_id), run_id=run_id), state
    ).run
    run = run_from_proto(routed)
    node_store = InMemoryObjectStore()
    node_state = InMemoryNodeState(node_store)
    node_state.set_node_id(node_id)
    message = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=run_id,
            message_id="",
            src_node_id=SUPERLINK_NODE_ID,
            dst_node_id=node_id,
            reply_to_message_id="",
            group_id="capability-poc",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type="query",
        ),
    )
    message.metadata.__dict__["_message_id"] = message.object_id
    object_contents = {
        object_id: obj.deflate()
        for object_id, obj in get_all_nested_objects(message).items()
    }
    fab_retrievals = 0
    confirmed = False

    def pull_fab(fab_hash: str, requested_run_id: int) -> Fab:
        nonlocal fab_retrievals
        if requested_run_id != run_id:
            raise RuntimeError("unexpected run ID during FAB retrieval")
        fab_retrievals += 1
        fab = state.get_fab(fab_hash)
        if fab is None:
            raise RuntimeError("FAB was not stored by StartRun")
        return fab

    def confirm(requested_run_id: int, message_id: str) -> None:
        nonlocal confirmed
        if requested_run_id != run_id or not message_id:
            raise RuntimeError("invalid message confirmation")
        confirmed = True

    def pull_run(requested_run_id: int) -> Run:
        if requested_run_id != run_id:
            raise RuntimeError("unexpected run ID")
        return run

    def pull_object(requested_run_id: int, object_id: str) -> bytes:
        if requested_run_id != run_id:
            raise RuntimeError("unexpected run ID")
        return object_contents[object_id]

    result_run_id = _pull_and_store_message(
        state=node_state,
        object_store=node_store,
        node_config={},
        receive=lambda: (
            remove_content_from_message(message),
            get_object_tree(message),
        ),
        get_run=pull_run,
        get_fab=pull_fab,
        pull_object=pull_object,
        confirm_message_received=confirm,
        trusted_entities={},
    )
    tasks = node_state.get_tasks(run_ids=[run_id])
    return {
        "run_id": result_run_id,
        "package_routed": bool(run.capability_package),
        "fab_retrieved": fab_retrievals == 1,
        "task_created": bool(tasks),
        "message_confirmed": confirmed,
    }


def run_smoke() -> dict[str, object]:
    """Run allowed and denied scenarios and return their evidence."""
    with tempfile.TemporaryDirectory(prefix="flwr-capability-poc-") as tmp:
        root = Path(tmp)
        app_dir = _write_tiny_app(root)
        fab_bytes = build_fab_from_disk(app_dir)
        if fab_bytes != build_fab_from_disk(app_dir):
            raise RuntimeError("tiny FAB build is not deterministic")
        fab_hash = hashlib.sha256(fab_bytes).hexdigest()

        _, public_key = generate_key_pairs()
        public_key_bytes = public_key_to_bytes(public_key)
        participant_id = participant_id_from_public_key(public_key_bytes)
        state = InMemoryLinkState(NoOpFederationManager(), InMemoryObjectStore())
        node_id = state.create_node(
            NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, public_key_bytes, heartbeat_interval=30
        )
        binding = capability_binding(NOOP_FEDERATION_ID, fab_hash)

        server = ThreadingHTTPServer(("127.0.0.1", 0), GuardianMockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        scenarios: dict[str, dict[str, object]] = {}
        try:
            with patch.dict(
                "os.environ",
                {"FLWR_GUARDIAN_URL": f"http://127.0.0.1:{server.server_port}"},
                clear=True,
            ):
                for scenario, package in (
                    ("allowed", binding),
                    ("denied", f"deny:{binding}"),
                ):
                    capabilities_file = root / f"{scenario}-capabilities.json"
                    capabilities_file.write_text(
                        json.dumps(
                            {
                                "version": CAPABILITY_FILE_VERSION,
                                "capabilities": {participant_id: package},
                            },
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    run_id = _submit_run(app_dir, capabilities_file, state)
                    _transition_to_running(state, run_id)
                    scenarios[scenario] = _exercise_supernode(state, node_id, run_id)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        allowed = scenarios["allowed"]
        denied = scenarios["denied"]
        if not all(
            allowed[key]
            for key in (
                "package_routed",
                "fab_retrieved",
                "task_created",
                "message_confirmed",
            )
        ):
            raise RuntimeError(f"allowed scenario did not complete: {allowed}")
        if denied["fab_retrieved"] or denied["task_created"]:
            raise RuntimeError(f"denied scenario did not fail closed: {denied}")
        return {
            "participant_id": participant_id,
            "fab_hash": fab_hash,
            "binding": binding,
            "scenarios": scenarios,
        }


def main() -> None:
    """Run the smoke harness and print machine-readable evidence."""
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
