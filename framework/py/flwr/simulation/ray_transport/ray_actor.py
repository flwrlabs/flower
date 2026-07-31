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
"""Ray-based Flower Actor and ActorPool implementation."""


import threading
from abc import ABC
from collections.abc import Callable
from logging import DEBUG, ERROR, WARNING
from typing import Any

import ray
from ray import ObjectRef
from ray.util.actor_pool import ActorPool

from flwr.app import Context, Message
from flwr.clientapp.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.common.logger import log

ClientAppFn = Callable[[], ClientApp]


class VirtualClientEngineActor(ABC):
    """Abstract base class for VirtualClientEngine Actors."""

    def terminate(self) -> None:
        """Manually terminate Actor object."""
        log(WARNING, "Manually terminating %s", self.__class__.__name__)
        ray.actor.exit_actor()  # type: ignore[no-untyped-call]

    def run(
        self,
        client_app_fn: ClientAppFn,
        message: Message,
        cid: str,
        context: Context,
    ) -> tuple[str, Message, Context]:
        """Run a client run."""
        # Pass message through ClientApp and return a message
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            # Load app
            app: ClientApp = client_app_fn()

            # Handle task message
            out_message = app(message=message, context=context)

        except LoadClientAppError as load_ex:
            raise load_ex

        except Exception as ex:
            raise ClientAppException(str(ex)) from ex

        return cid, out_message, context


@ray.remote
class ClientAppActor(VirtualClientEngineActor):
    """A Ray Actor class that runs client runs.

    Parameters
    ----------
    on_actor_init_fn: Optional[Callable[[], None]] (default: None)
        A function to execute upon actor initialization.
    """

    def __init__(self, on_actor_init_fn: Callable[[], None] | None = None) -> None:
        super().__init__()
        if on_actor_init_fn:
            on_actor_init_fn()


def pool_size_from_resources(client_resources: dict[str, int | float]) -> int:
    """Calculate number of Actors that fit in the cluster.

    For this we consider the resources available on each node and those required per
    client.
    """
    total_num_actors = 0

    # We calculate the number of actors that fit in a node per node basis. This is
    # the right way of doing it otherwise situations like the following arise: imagine
    # each client needs 3 CPUs and Ray has w nodes (one with 2 CPUs and another with 4)
    # if we don't follow a per-node estimation of actors, we'll be creating an actor
    # pool with 2 Actors. This, however, doesn't fit in the cluster since only one of
    # the nodes can fit one Actor.
    nodes = ray.nodes()  # type: ignore[no-untyped-call]
    for node in nodes:
        node_resources = node["Resources"]

        # If a node has detached, it is still in the list of nodes
        # however, its resources will be empty.
        if not node_resources:
            continue

        # Fallback to zero when resource quantity is not configured on the ray node
        # e.g.: node without GPU; head node set up not to run tasks (zero resources)
        num_cpus = node_resources.get("CPU", 0)
        num_gpus = node_resources.get("GPU", 0)
        num_actors = int(num_cpus / client_resources["num_cpus"])

        # If a GPU is present and client resources do require one
        if "num_gpus" in client_resources.keys() and client_resources["num_gpus"] > 0.0:
            if num_gpus:
                # If there are gpus in the cluster
                num_actors = min(
                    num_actors, int(num_gpus / client_resources["num_gpus"])
                )
            else:
                num_actors = 0
        total_num_actors += num_actors

    if total_num_actors == 0:
        log(
            WARNING,
            "The ActorPool is empty. The system (CPUs=%s, GPUs=%s) "
            "does not meet the criteria to host at least one client with resources:"
            " %s. Lowering these resources could help.",
            num_cpus,
            num_gpus,
            client_resources,
        )
        raise ValueError(
            "ActorPool is empty. Stopping Simulation. "
            "Check `num_cpus` and/or `num_gpus` passed to the simulation engine"
        )

    return total_num_actors


class VirtualClientEngineActorPool(ActorPool):
    """A pool of VirtualClientEngine Actors.

    Parameters
    ----------
    create_actor_fn : Callable[[], Type[VirtualClientEngineActor]]
        A function that returns an actor that can be added to the pool.

    client_resources : Dict[str, Union[int, float]]
        A dictionary specifying the system resources that each
        actor should have access. This will be used to calculate
        the number of actors that fit in your cluster. Supported keys
        are `num_cpus` and `num_gpus`. E.g. {`num_cpus`: 2, `num_gpus`: 0.5}
        would allocate two Actors per GPU in your system assuming you have
        enough CPUs. To understand the GPU utilization caused by `num_gpus`,
        as well as using custom resources, please consult the Ray documentation.

    actor_lists: List[VirtualClientEngineActor] (default: None)
        This argument should not be used. It's only needed for serialization purposes
        (see the `__reduce__` method). Each time it is executed, we want to retain
        the same list of actors.
    """

    def __init__(
        self,
        create_actor_fn: Callable[[], type[VirtualClientEngineActor]],
        client_resources: dict[str, int | float],
        actor_list: list[type[VirtualClientEngineActor]] | None = None,
    ):
        self.client_resources = client_resources
        self.create_actor_fn = create_actor_fn

        if actor_list is None:
            # Figure out how many actors can be created given the cluster resources
            # and the resources the user indicates each VirtualClient will need
            num_actors = pool_size_from_resources(client_resources)
            actors = [create_actor_fn() for _ in range(num_actors)]
        else:
            # When __reduce__ is executed, we don't want to created
            # a new list of actors again.
            actors = actor_list

        super().__init__(actors)

        # A dict that maps cid to another dict containing: a reference to the remote job
        # and its status (i.e. whether it is ready or not)
        self._cid_to_future: dict[
            str, dict[str, bool | ObjectRef[Any] | tuple[Any, ...] | None]
        ] = {}
        self.actor_to_remove: set[str] = set()  # a set
        self.num_actors = len(actors)

        self.lock = threading.RLock()

    def __reduce__(self):  # type: ignore
        """Make this class serializable (needed due to lock)."""
        return VirtualClientEngineActorPool, (
            self.create_actor_fn,
            self.client_resources,
            self._idle_actors,  # Pass existing actors to avoid killing/re-creating
        )

    def add_actors_to_pool(self, num_actors: int) -> None:
        """Add actors to the pool.

        This expands the pool after it has been created iif new resources are added to
        your Ray cluster (e.g. you add a new node).
        """
        with self.lock:
            new_actors = [self.create_actor_fn() for _ in range(num_actors)]
            self._idle_actors.extend(new_actors)
            self.num_actors += num_actors

    def submit(self, fn: Any, value: tuple[ClientAppFn, Message, str, Context]) -> None:
        """Take an idle actor and assign it to run a client app and Message.

        Submit a job to an actor by first removing it from the list of idle actors, then
        check if this actor was flagged to be removed from the pool.
        """
        app_fn, mssg, cid, context = value
        actor = self._idle_actors.pop()
        if self._check_and_remove_actor_from_pool(actor):
            future = fn(actor, app_fn, mssg, cid, context)
            future_key = tuple(future) if isinstance(future, list) else future
            self._future_to_actor[future_key] = (self._next_task_index, actor, cid)
            self._next_task_index += 1

            # Update with future
            self._cid_to_future[cid]["future"] = future_key

    def submit_client_job(
        self, actor_fn: Any, job: tuple[ClientAppFn, Message, str, Context]
    ) -> None:
        """Submit a job while tracking client ids."""
        _, _, cid, _ = job

        # We need to put this behind a lock since .submit() involves
        # removing and adding elements from a dictionary. Which creates
        # issues in multi-threaded settings
        with self.lock:
            # Create cid to future mapping
            self._reset_cid_to_future_dict(cid)
            if self._idle_actors:
                # Submit job since there is an Actor that's available
                self.submit(actor_fn, job)
            else:
                # No actors are available, append to list of jobs to run later
                self._pending_submits.append((actor_fn, job))

    def _flag_future_as_ready(self, cid: str) -> None:
        """Flag future for VirtualClient with cid=cid as ready."""
        self._cid_to_future[cid]["ready"] = True

    def _reset_cid_to_future_dict(self, cid: str) -> None:
        """Reset cid:future mapping info."""
        if cid not in self._cid_to_future:
            self._cid_to_future[cid] = {}

        self._cid_to_future[cid]["future"] = None
        self._cid_to_future[cid]["ready"] = False

    def _is_future_ready(self, cid: str) -> bool:
        """Return status of future associated to the given client id (cid)."""
        if cid not in self._cid_to_future:
            # With the current ClientProxy<-->ActorPool interaction
            # we should never be hitting this condition.
            log(WARNING, "This shouldn't be happening")
            return False

        return self._cid_to_future[cid]["ready"]  # type: ignore

    def _fetch_future_result(self, cid: str) -> tuple[Message, Context]:
        """Fetch result and updated context for a VirtualClient from Object Store.

        The job submitted by the ClientProxy interfacing with client with cid=cid is
        ready. Here we fetch it from the object store and return.
        """
        try:
            future: ObjectRef[Any] = self._cid_to_future[cid]["future"]  # type: ignore
            res_cid, out_mssg, updated_context = ray.get(
                future
            )  # type: (str, Message, Context)
        except ray.exceptions.RayActorError as ex:
            log(ERROR, ex)
            if hasattr(ex, "actor_id"):
                # RayActorError only contains the actor_id attribute
                # if the actor won't be restarted again.
                self._flag_actor_for_removal(ex.actor_id)
            raise ex

        # Sanity check: was the result fetched generated by a client with cid=cid?
        assert res_cid == cid, log(
            ERROR, "The VirtualClient %s got result from client %s", cid, res_cid
        )

        # Reset mapping
        self._reset_cid_to_future_dict(cid)

        return out_mssg, updated_context

    def _flag_actor_for_removal(self, actor_id_hex: str) -> None:
        """Flag actor that should be removed from pool."""
        with self.lock:
            self.actor_to_remove.add(actor_id_hex)
            log(WARNING, "Actor(%s) will be remove from pool.", actor_id_hex)

    def _check_and_remove_actor_from_pool(
        self, actor: VirtualClientEngineActor
    ) -> bool:
        """Check if actor in set of those that should be removed.

        Remove the actor if so.
        """
        with self.lock:
            actor_id = (
                actor._actor_id.hex()  # type: ignore # pylint: disable=protected-access
            )

            if actor_id in self.actor_to_remove:
                # The actor should be removed
                self.actor_to_remove.remove(actor_id)
                self.num_actors -= 1
                log(WARNING, "REMOVED actor %s from pool", actor_id)
                log(WARNING, "Pool size: %s", self.num_actors)
                return False

            return True

    def _check_actor_fits_in_pool(self) -> bool:
        """Determine if available resources haven't changed.

        If true, allow the actor to be added back to the pool. Else don't allow it
        (effectively reducing the size of the pool).
        """
        num_actors_updated = pool_size_from_resources(self.client_resources)

        if num_actors_updated < self.num_actors:
            log(
                WARNING,
                "Cluster resources have changed. Number of actors in the pool should be"
                " reduced from %s down to %s. This"
                " might take several intermediate steps",
                self.num_actors,
                num_actors_updated,
            )
            # We are preventing one actor from being added back to the queue, so we just
            # decrease the number of actors by one. Eventually `self.num_actors`
            # should be equal to what `pool_size_from_resources(self.resources)` returns
            self.num_actors -= 1
            return False

        return True

    def process_unordered_future(self, timeout: float | None = None) -> None:
        """Similar to parent's get_next_unordered() but without final ray.get()."""
        if not self.has_next():  # type: ignore
            raise StopIteration("No more results to get")

        # Block until one result is ready
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)

        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")

        with self.lock:
            # Get actor that completed a job
            _, actor, cid = self._future_to_actor.pop(future, (None, None, -1))
            if actor is not None:
                # Still space in queue? (no if a node in the cluster died)
                if self._check_actor_fits_in_pool():
                    if self._check_and_remove_actor_from_pool(actor):
                        self._return_actor(actor)  # type: ignore
                    # Flag future as ready so ClientProxy with cid
                    # can break from the while loop (in `get_client_result()`)
                    # and fetch its result
                    self._flag_future_as_ready(cid)
                else:
                    # The actor doesn't fit in the pool anymore.
                    # Manually terminate the actor
                    actor.terminate.remote()

    def get_client_result(
        self, cid: str, timeout: float | None
    ) -> tuple[Message, Context]:
        """Get result from VirtualClient with specific cid."""
        # Loop until all jobs submitted to the pool are completed. Break early
        # if the result for the ClientProxy calling this method is ready
        while self.has_next() and not self._is_future_ready(cid):  # type: ignore
            try:
                self.process_unordered_future(timeout=timeout)
            except StopIteration:
                # There are no pending jobs in the pool
                break

        # Fetch result belonging to the VirtualClient calling this method
        # Return both result from tasks and (potentially) updated run context
        return self._fetch_future_result(cid)


class BasicActorPool:
    """A basic actor pool."""

    def __init__(
        self,
        actor_type: type[VirtualClientEngineActor],
        client_resources: dict[str, int | float],
        actor_kwargs: dict[str, Any],
    ):
        self.client_resources = client_resources

        # Queue of idle actors
        self.pool: list[VirtualClientEngineActor] = []
        self.num_actors = 0

        # Resolve arguments to pass during actor init
        actor_args = {} if actor_kwargs is None else actor_kwargs

        # A function that creates an actor
        self.create_actor_fn = lambda: actor_type.options(  # type: ignore
            **client_resources
        ).remote(**actor_args)

        # Figure out how many actors can be created given the cluster resources
        # and the resources the user indicates each VirtualClient will need
        self.actors_capacity = pool_size_from_resources(client_resources)
        self._future_to_actor: dict[Any, VirtualClientEngineActor] = {}

    def is_actor_available(self) -> bool:
        """Return true if there is an idle actor."""
        return len(self.pool) > 0

    def add_actors_to_pool(self, num_actors: int) -> None:
        """Add actors to the pool.

        This method may be executed also if new resources are added to your Ray cluster
        (e.g. you add a new node).
        """
        for _ in range(num_actors):
            self.pool.append(self.create_actor_fn())  # type: ignore
        self.num_actors += num_actors

    def terminate_all_actors(self) -> None:
        """Terminate actors in pool."""
        num_terminated = 0
        for actor in self.pool:
            actor.terminate.remote()  # type: ignore
            num_terminated += 1

        log(DEBUG, "Terminated %i actors", num_terminated)

    def submit(
        self, actor_fn: Any, job: tuple[ClientAppFn, Message, str, Context]
    ) -> Any:
        """On idle actor, submit job and return future."""
        # Remove idle actor from pool
        actor = self.pool.pop()
        # Submit job to actor
        app_fn, mssg, cid, context = job
        future = actor_fn(actor, app_fn, mssg, cid, context)
        # Keep track of future:actor (so we can fetch the actor upon job completion
        # and add it back to the pool)
        self._future_to_actor[future] = actor
        return future

    def add_actor_back_to_pool(self, future: Any) -> None:
        """Ad actor assigned to run future back into the pool."""
        actor = self._future_to_actor.pop(future)
        self.pool.append(actor)

    def fetch_result_and_return_actor_to_pool(
        self, future: Any
    ) -> tuple[Message, Context]:
        """Pull result given a future and add actor back to pool."""
        # Retrieve result for object store
        # Instead of doing ray.get(future) we await it
        _, out_mssg, updated_context = ray.get(future)
        # Get actor that ran job
        self.add_actor_back_to_pool(future)
        return out_mssg, updated_context
