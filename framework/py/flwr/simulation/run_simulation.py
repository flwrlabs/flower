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


import asyncio
import importlib
import json
import logging
import platform
import threading
import traceback
from logging import DEBUG, ERROR, INFO, WARNING
from queue import Empty, Queue
from typing import Any, cast

from flwr.app.user_config import UserConfig
from flwr.cli.utils import get_sha256_hash
from flwr.clientapp import ClientApp
from flwr.common import Context, EventType, RecordDict, event, log, now
from flwr.common.constant import RUN_ID_NUM_BYTES, Status
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import (
    set_logger_propagation,
    update_console_handler,
    warn_deprecated_feature,
    warn_deprecated_feature_with_example,
)
from flwr.common.typing import Run, RunStatus
from flwr.server.grid import Grid, InMemoryGrid
from flwr.server.run_serverapp import run as _run
from flwr.server.server_app import ServerApp
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.linkstate.in_memory_linkstate import RunRecord
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.ray_transport.utils import (
    enable_tf_gpu_growth as enable_gpu_growth,
)
from flwr.supercore.constant import (
    DEFAULT_SIMULATION_CONFIG,
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION,
)
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


def _check_ray_support(backend_name: str) -> None:
    if backend_name == "ray":
        if platform.system() == "Windows":
            log(
                WARNING,
                "Ray support on Windows is experimental "
                "and may not work as expected. "
                "On Windows, Flower Simulations run best in WSL2: "
                "https://learn.microsoft.com/en-us/windows/wsl/about",
            )


# Entry point from Python session (script or notebook)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_simulation(
    server_app: ServerApp,
    client_app: ClientApp,
    num_supernodes: int,
    backend_name: str = "ray",
    backend_config: BackendConfig | None = None,
    enable_tf_gpu_growth: bool = False,
    verbose_logging: bool = False,
) -> None:
    r"""Run a Flower App using the Simulation Engine.

    Parameters
    ----------
    server_app : ServerApp
        The `ServerApp` to be executed. It will send messages to different `ClientApp`
        instances running on different (virtual) SuperNodes.

    client_app : ClientApp
        The `ClientApp` to be executed by each of the SuperNodes. It will receive
        messages sent by the `ServerApp`.

    num_supernodes : int
        Number of nodes that run a ClientApp. They can be sampled by a Grid in the
        ServerApp and receive a Message describing what the ClientApp should perform.

    backend_name : str (default: ray)
        A simulation backend that runs `ClientApp` objects.

    backend_config : Optional[BackendConfig]
        'A dictionary to configure a backend. Separate dictionaries to configure
        different elements of backend. Supported top-level keys are `init_args`
        for values parsed to initialisation of backend, `client_resources`
        to define the resources for clients, and `actor` to define the actor
        parameters. Values supported in <value> are those included by
        `flwr.common.typing.ConfigRecordValues`.

    enable_tf_gpu_growth : bool (default: False)
        A boolean to indicate whether to enable GPU growth on the main thread. This is
        desirable if you make use of a TensorFlow model on your `ServerApp` while
        having your `ClientApp` running on the same GPU. Without enabling this, you
        might encounter an out-of-memory error because TensorFlow, by default, allocates
        all GPU memory. Read more about how `tf.config.experimental.set_memory_growth()`
        works in the TensorFlow documentation: https://www.tensorflow.org/api/stable.

    verbose_logging : bool (default: False)
        When disabled, only INFO, WARNING and ERROR log messages will be shown. If
        enabled, DEBUG-level logs will be displayed.
    """
    warn_deprecated_feature(
        "The `run_simulation` function is deprecated and will be removed in a future "
        "version of Flower. Please use `flwr run` in the CLI instead to run your "
        "simulation. Refer to the Flower Tutorials "
        "for more details: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html",
    )
    event(
        EventType.PYTHON_API_RUN_SIMULATION_ENTER,
        event_details={"backend": backend_name, "num-supernodes": num_supernodes},
    )

    if enable_tf_gpu_growth:
        warn_deprecated_feature_with_example(
            "Passing `enable_tf_gpu_growth=True` is deprecated.",
            example_message="Instead, set the `TF_FORCE_GPU_ALLOW_GROWTH` environment "
            "variable to true.",
            code_example='import os;os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"'
            "\n\tflwr.simulation.run_simulationt(...)",
        )

    _check_ray_support(backend_name)

    _ = _run_simulation(
        num_supernodes=num_supernodes,
        client_app=client_app,
        server_app=server_app,
        backend_name=backend_name,
        backend_config=backend_config,
        enable_tf_gpu_growth=enable_tf_gpu_growth,
        verbose_logging=verbose_logging,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_serverapp_th(
    server_app_attr: str | None,
    server_app: ServerApp | None,
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
        _server_app_attr: str | None,
        _server_app: ServerApp | None,
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
                loaded_server_app=_server_app,
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
            server_app,
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
    is_app: bool,
    enable_tf_gpu_growth: bool,
    run: Run,
    exit_event: EventType,
    client_app: ClientApp | None = None,
    client_app_attr: str | None = None,
    server_app: ServerApp | None = None,
    server_app_attr: str | None = None,
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
            server_app=server_app,
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
            client_app=client_app,
            backend_name=backend_name,
            backend_config_json_stream=backend_config_stream,
            app_dir=app_dir,
            is_app=is_app,
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
            serverapp_th.join(timeout=5)
            if server_app_thread_has_exception.is_set():
                raise RuntimeError("Exception in ServerApp thread")

    log(DEBUG, "Stopping Simulation Engine now.")
    return updated_context


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def _run_simulation(
    num_supernodes: int,
    exit_event: EventType,
    client_app: ClientApp | None = None,
    server_app: ServerApp | None = None,
    backend_name: str = "ray",
    backend_config: BackendConfig | None = None,
    client_app_attr: str | None = None,
    server_app_attr: str | None = None,
    server_app_context: Context | None = None,
    app_dir: str = "",
    run: Run | None = None,
    enable_tf_gpu_growth: bool = False,
    verbose_logging: bool = False,
    is_app: bool = False,
) -> Context:
    """Launch the Simulation Engine."""
    if backend_config is None:
        backend_config = {}
    elif backend_config:
        # Backend config internally operates with `_` not with `-`
        backend_config = cast(
            BackendConfig, _replace_keys(backend_config, match="-", target="_")
        )
        log(DEBUG, "backend_config: %s", backend_config)

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

    # Set default init_args if not passed
    backend_config.setdefault("init_args", {})
    # Set default client_resources if not passed
    backend_config.setdefault(
        "client_resources",
        {
            "num_cpus": DEFAULT_SIMULATION_CONFIG.client_resources_num_cpus,
            "num_gpus": DEFAULT_SIMULATION_CONFIG.client_resources_num_gpus,
        },
    )
    # Initialization of backend config to enable GPU growth globally when set
    backend_config.setdefault("actor", {"tensorflow": 0})

    # Set logging level
    logger = logging.getLogger("flwr")
    if verbose_logging:
        update_console_handler(level=DEBUG, timestamps=True, colored=True)
    else:
        init_args = backend_config["init_args"]
        init_args.setdefault(
            "logging_level", DEFAULT_SIMULATION_CONFIG.init_args_logging_level
        )
        init_args.setdefault(
            "log_to_driver", DEFAULT_SIMULATION_CONFIG.init_args_log_to_driver
        )

    if enable_tf_gpu_growth:
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config.get("actor", {}).get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config["actor"]["tensorflow"] = True

    # Convert config to original JSON-stream format
    backend_config_stream = json.dumps(backend_config)

    # If no `Run` object is set, create one
    if run is None:
        run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
        run = Run.create_empty(run_id=run_id)
        run.federation = NOOP_FEDERATION

    args = (
        num_supernodes,
        backend_name,
        backend_config_stream,
        app_dir,
        is_app,
        enable_tf_gpu_growth,
        run,
        exit_event,
        client_app,
        client_app_attr,
        server_app,
        server_app_attr,
        server_app_context,
    )
    # Detect if there is an Asyncio event loop already running.
    # If yes, disable logger propagation. In environmnets
    # like Jupyter/Colab notebooks, it's often better to do this.
    asyncio_loop_running = False
    try:
        _ = (
            asyncio.get_running_loop()
        )  # Raises RuntimeError if no event loop is present
        log(DEBUG, "Asyncio event loop already running.")

        asyncio_loop_running = True

    except RuntimeError:
        pass

    finally:
        if asyncio_loop_running:
            # Set logger propagation to False to prevent duplicated log output in Colab.
            logger = set_logger_propagation(logger, False)

        updated_context = _main_loop(*args)
    return updated_context
