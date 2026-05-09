# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower Simulation."""


import importlib
import json
import threading
import traceback
from logging import DEBUG, ERROR, INFO, WARNING
from queue import Empty, Queue
from typing import Any

from flwr.app.user_config import UserConfig
from flwr.cli.utils import get_sha256_hash
from flwr.common import Context, EventType, RecordDict, event, log, now
from flwr.common.constant import Status
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.typing import Run, RunStatus
from flwr.server.grid import Grid, InMemoryGrid
from flwr.server.run_serverapp import run as _run
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.linkstate.in_memory_linkstate import RunRecord
from flwr.simulation.ray_transport.utils import (
    enable_tf_gpu_growth as enable_gpu_growth,
)
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


def _replace_keys(d: Any, match: str, target: str) -> Any:
    if isinstance(d, dict):
        return {
            k.replace(match, target): _replace_keys(v, match, target)
            for k, v in d.items()
        }
    if isinstance(d, list):
        return [_replace_keys(i, match, target) for i in d]
    return d


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_serverapp_th(
    server_app_attr: str,
    server_app_context: Context,
    grid: Grid,
    app_dir: str,
    f_stop: threading.Event,
    has_exception: threading.Event,
    enable_tf_gpu_growth: bool,
    ctx_queue: "Queue[Context]",
) -> threading.Thread:
    """Run SeverApp in a thread."""

    def server_th_with_start_checks(
        tf_gpu_growth: bool,
        stop_event: threading.Event,
        exception_event: threading.Event,
        _grid: Grid,
        _server_app_dir: str,
        _server_app_attr: str,
        _ctx_queue: "Queue[Context]",
    ) -> None:
        """Run SeverApp, after check if GPU memory growth has to be set.

        Upon exception, trigger stop event for Simulation Engine.
        """
        try:
            if tf_gpu_growth:
                log(INFO, "Enabling GPU growth for Tensorflow on the server thread.")
                enable_gpu_growth()

            # Run ServerApp
            updated_context = _run(
                grid=_grid,
                context=server_app_context,
                server_app_dir=_server_app_dir,
                server_app_attr=_server_app_attr,
            )
            _ctx_queue.put(updated_context)
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "ServerApp thread raised an exception: %s", ex)
            log(ERROR, traceback.format_exc())
            exception_event.set()
            raise
        finally:
            log(DEBUG, "ServerApp finished running.")
            # Upon completion, trigger stop event if one was passed
            if stop_event is not None:
                stop_event.set()
                log(DEBUG, "Triggered stop event for Simulation Engine.")

    serverapp_th = threading.Thread(
        target=server_th_with_start_checks,
        args=(
            enable_tf_gpu_growth,
            f_stop,
            has_exception,
            grid,
            app_dir,
            server_app_attr,
            ctx_queue,
        ),
    )
    serverapp_th.start()
    return serverapp_th


# pylint: disable=too-many-locals,too-many-positional-arguments
def _main_loop(
    num_supernodes: int,
    backend_name: str,
    backend_config_stream: str,
    app_dir: str,
    enable_tf_gpu_growth: bool,
    run: Run,
    exit_event: EventType,
    client_app_attr: str,
    server_app_attr: str,
    server_app_context: Context | None = None,
) -> Context:
    """Start ServerApp on a separate thread, then launch Simulation Engine."""
    # Initialize StateFactory
    state_factory = LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
    )

    f_stop = threading.Event()
    # A Threading event to indicate if an exception was raised in the ServerApp thread
    server_app_thread_has_exception = threading.Event()
    serverapp_th = None
    success = True
    if server_app_context is None:
        server_app_context = Context(
            run_id=run.run_id,
            node_id=0,
            node_config=UserConfig(),
            state=RecordDict(),
            run_config=UserConfig(),
        )
    updated_context = server_app_context
    try:
        # Register run
        log(DEBUG, "Pre-registering run with id %s", run.run_id)
        run.status = RunStatus(Status.RUNNING, "", "")
        run.starting_at = now().isoformat()
        run.running_at = run.starting_at
        state_factory.state().run_ids[run.run_id] = RunRecord(run=run)  # type: ignore

        # Initialize Grid
        grid = InMemoryGrid(state_factory=state_factory)
        grid.set_run(run)
        output_context_queue: Queue[Context] = Queue()

        # Get and run ServerApp thread
        serverapp_th = run_serverapp_th(
            server_app_attr=server_app_attr,
            server_app_context=server_app_context,
            grid=grid,
            app_dir=app_dir,
            f_stop=f_stop,
            has_exception=server_app_thread_has_exception,
            enable_tf_gpu_growth=enable_tf_gpu_growth,
            ctx_queue=output_context_queue,
        )

        # Start Simulation Engine
        vce.start_vce(
            num_supernodes=num_supernodes,
            client_app_attr=client_app_attr,
            backend_name=backend_name,
            backend_config_json_stream=backend_config_stream,
            app_dir=app_dir,
            state_factory=state_factory,
            f_stop=f_stop,
            run=run,
        )

        updated_context = output_context_queue.get(timeout=3)

    except Empty:
        log(DEBUG, "Queue timeout. No context received.")

    except Exception as ex:
        log(ERROR, "An exception occurred !! %s", ex)
        log(ERROR, traceback.format_exc())
        success = False
        raise RuntimeError("An error was encountered. Ending simulation.") from ex

    finally:
        # Trigger stop event
        f_stop.set()
        event(
            exit_event,
            event_details={
                "run-id-hash": get_sha256_hash(run.run_id),
                "success": success,
            },
        )
        if serverapp_th:
            if server_app_thread_has_exception.is_set():
                raise RuntimeError("Exception in ServerApp thread")

    log(DEBUG, "Stopping Simulation Engine now.")
    return updated_context


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def _run_simulation(
    num_supernodes: int,
    exit_event: EventType,
    backend_name: str = "ray",
    backend_config: BackendConfig | None = None,
    client_app_attr: str | None = None,
    server_app_attr: str | None = None,
    server_app_context: Context | None = None,
    app_dir: str = "",
    run: Run | None = None,
    enable_tf_gpu_growth: bool = False,
) -> Context:
    """Launch the Simulation Engine."""
    # Exit early if the `ray` dependency is missing
    if backend_name == "ray":
        if importlib.util.find_spec("ray") is None:
            flwr_exit(
                code=ExitCode.SIMULATION_MISSING_EXTRA,
                message=(
                    "`ray` backend selected for simulation, but `ray` is not "
                    "installed."
                ),
                event_type=exit_event,
                event_details={"success": False},
            )

    # Initialization of backend config to enable GPU growth globally when set
    backend_config.setdefault("actor", {"tensorflow": 0})

    if enable_tf_gpu_growth:
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config.get("actor", {}).get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config["actor"]["tensorflow"] = True

    # Convert config to original JSON-stream format
    backend_config_stream = json.dumps(backend_config)

    args = (
        num_supernodes,
        backend_name,
        backend_config_stream,
        app_dir,
        enable_tf_gpu_growth,
        run,
        exit_event,
        client_app_attr,
        server_app_attr,
        server_app_context,
    )
    updated_context = _main_loop(*args)
    return updated_context
