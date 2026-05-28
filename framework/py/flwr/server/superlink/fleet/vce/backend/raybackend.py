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
"""Ray backend for the Fleet API using the Simulation Engine."""


import os
import sys
from collections.abc import Callable
from logging import DEBUG, ERROR
from typing import Any

import ray

from flwr.app.message import Context, Message
from flwr.clientapp.client_app import ClientApp
from flwr.common.constant import PARTITION_ID_KEY
from flwr.common.logger import log
from flwr.simulation.ray_transport.ray_actor import BasicActorPool, ClientAppActor
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from .backend import Backend, BackendConfig

ClientResourcesDict = dict[str, int | float]
ActorArgsDict = dict[str, int | float | Callable[[], None]]


class RayBackend(Backend):
    """A backend that submits jobs to a `BasicActorPool`."""

    def __init__(
        self,
        backend_config: BackendConfig,
    ) -> None:
        """Prepare RayBackend by initialising Ray and creating the ActorPool."""
        log(DEBUG, "Initialising: %s", self.__class__.__name__)
        log(DEBUG, "Backend config: %s", backend_config)

        # Initialise ray
        self.init_args_key = "init_args"
        self.init_ray(backend_config)

        # Validate client resources
        self.client_resources_key = "client_resources"
        self.client_resources = self._validate_client_resources(config=backend_config)

        # Valide actor resources
        self.actor_kwargs = self._validate_actor_arguments(config=backend_config)
        self.pool: BasicActorPool | None = None

        self.app_fn: Callable[[], ClientApp] | None = None

    def _validate_client_resources(self, config: BackendConfig) -> ClientResourcesDict:
        client_resources_config = config.get(self.client_resources_key)
        client_resources: ClientResourcesDict = {}
        valid_types = (int, float)
        if client_resources_config:
            for k, v in client_resources_config.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"client resources keys are expected to be `str` but you used "
                        f"{type(k)} for `{k}`"
                    )
                if not isinstance(v, valid_types):
                    raise ValueError(
                        f"client resources are expected to be of type {valid_types} "
                        f"but found `{type(v)}` for key `{k}`",
                    )
                client_resources[k] = v

        else:
            client_resources = {"num_cpus": 2, "num_gpus": 0.0}
            log(
                DEBUG,
                "`%s` not specified in backend config. Applying default setting: %s",
                self.client_resources_key,
                client_resources,
            )

        return client_resources

    def _validate_actor_arguments(self, config: BackendConfig) -> ActorArgsDict:
        actor_args_config = config.get("actor", False)
        actor_args: ActorArgsDict = {}
        if actor_args_config:
            use_tf = actor_args.get("tensorflow", False)
            if use_tf:
                actor_args["on_actor_init_fn"] = enable_tf_gpu_growth
        return actor_args

    def init_ray(self, backend_config: BackendConfig) -> None:
        """Intialises Ray if not already initialised."""
        if not ray.is_initialized():
            ray_init_args: dict[str, Any] = {}

            if backend_config.get(self.init_args_key):
                for k, v in backend_config[self.init_args_key].items():
                    ray_init_args[k] = v
            ray_init_args.setdefault("include_dashboard", False)

            # Ray subprocesses inherit environment variables, but not this
            # driver's in-memory sys.path changes from runtime dependency setup.
            pythonpath = os.pathsep.join(sys.path)
            if os.environ.get("PYTHONPATH"):
                pythonpath = f"{pythonpath}{os.pathsep}{os.environ['PYTHONPATH']}"
            os.environ["PYTHONPATH"] = pythonpath

            ray.init(
                runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
                **ray_init_args,
            )

    @property
    def num_workers(self) -> int:
        """Return number of actors in pool."""
        return self.pool.num_actors if self.pool else 0

    def is_worker_idle(self) -> bool:
        """Report whether the pool has idle actors."""
        return self.pool.is_actor_available() if self.pool else False

    def build(self, app_fn: Callable[[], ClientApp]) -> None:
        """Build pool of Ray actors that this backend will submit jobs to."""
        # Create Actor Pool
        try:
            self.pool = BasicActorPool(
                actor_type=ClientAppActor,
                client_resources=self.client_resources,
                actor_kwargs=self.actor_kwargs,
            )
        except Exception as ex:
            raise ex

        self.pool.add_actors_to_pool(self.pool.actors_capacity)
        # Set ClientApp callable that ray actors will use
        self.app_fn = app_fn
        log(DEBUG, "Constructed ActorPool with: %i actors", self.pool.num_actors)

    def process_message(
        self,
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        partition_id = context.node_config[PARTITION_ID_KEY]

        if self.pool is None:
            raise ValueError("The actor pool is empty, unfit to process messages.")

        if self.app_fn is None:
            raise ValueError(
                "Unspecified function to load a `ClientApp`. "
                "Call the backend's `build()` method before processing messages."
            )

        future = None
        try:
            # Submit a task to the pool
            future = self.pool.submit(
                lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
                (self.app_fn, message, str(partition_id), context),
            )

            # Fetch result
            (
                out_mssg,
                updated_context,
            ) = self.pool.fetch_result_and_return_actor_to_pool(future)

            return out_mssg, updated_context

        except Exception as ex:
            log(
                ERROR,
                "An exception was raised when processing a message by %s",
                self.__class__.__name__,
            )
            # add actor back into pool
            if future is not None:
                self.pool.add_actor_back_to_pool(future)
            raise ex

    def terminate(self) -> None:
        """Terminate all actors in actor pool."""
        if self.pool:
            self.pool.terminate_all_actors()
        ray.shutdown()
        log(DEBUG, "Terminated %s", self.__class__.__name__)
