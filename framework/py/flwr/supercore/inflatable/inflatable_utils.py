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
"""InflatableObject utilities."""


import concurrent.futures
import os
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from queue import Queue
from typing import TypeVar

from flwr.common.exit import add_exit_handler
from flwr.common.message import Message
from flwr.common.record import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    RecordDict,
)
from flwr.common.record.arraychunk import ArrayChunk
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611

from ..constant import (
    FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS,
    FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES,
    HEAD_BODY_DIVIDER,
    HEAD_VALUE_DIVIDER,
    PULL_BACKOFF_CAP,
    PULL_INITIAL_BACKOFF,
    PULL_MAX_TIME,
    PULL_MAX_TRIES_PER_OBJECT,
)
from .inflatable_object import (
    InflatableObject,
    UnexpectedObjectContentError,
    _get_object_head,
    get_object_head_values_from_object_content,
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)

# Helper registry that maps names of classes to their type
inflatable_class_registry: dict[str, type[InflatableObject]] = {
    ArrayChunk.__qualname__: ArrayChunk,
    Array.__qualname__: Array,
    ArrayRecord.__qualname__: ArrayRecord,
    ConfigRecord.__qualname__: ConfigRecord,
    Message.__qualname__: Message,
    MetricRecord.__qualname__: MetricRecord,
    RecordDict.__qualname__: RecordDict,
}

T = TypeVar("T", bound=InflatableObject)


# Allow thread pool executors to be shut down gracefully
_thread_pool_executors: set[concurrent.futures.ThreadPoolExecutor] = set()
_lock = threading.Lock()


def _shutdown_thread_pool_executors() -> None:
    """Shutdown all thread pool executors gracefully."""
    with _lock:
        for executor in _thread_pool_executors:
            executor.shutdown(wait=False, cancel_futures=True)
        _thread_pool_executors.clear()


def _track_executor(executor: concurrent.futures.ThreadPoolExecutor) -> None:
    """Track a thread pool executor for graceful shutdown."""
    with _lock:
        _thread_pool_executors.add(executor)


def _untrack_executor(executor: concurrent.futures.ThreadPoolExecutor) -> None:
    """Untrack a thread pool executor."""
    with _lock:
        _thread_pool_executors.discard(executor)


add_exit_handler(_shutdown_thread_pool_executors)


class ObjectUnavailableError(Exception):
    """Exception raised when an object has been pre-registered but is not yet
    available."""

    def __init__(self, object_id: str):
        super().__init__(f"Object with ID '{object_id}' is not yet available.")


class ObjectIdNotPreregisteredError(Exception):
    """Exception raised when an object ID is not pre-registered."""

    def __init__(self, object_id: str):
        super().__init__(f"Object with ID '{object_id}' could not be found.")


def get_num_workers(max_concurrent: int) -> int:
    """Get number of workers based on the number of CPU cores and the maximum
    allowed."""
    num_cores = os.cpu_count() or 1
    return min(max_concurrent, num_cores)


def push_objects(
    objects: dict[str, InflatableObject],
    push_object_fn: Callable[[str, bytes], None],
    *,
    object_ids_to_push: set[str] | None = None,
    keep_objects: bool = False,
    max_concurrent_pushes: int = FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES,
) -> None:
    """Push multiple objects to the servicer.

    Parameters
    ----------
    objects : dict[str, InflatableObject]
        A dictionary of objects to push, where keys are object IDs and values are
        `InflatableObject` instances.
    push_object_fn : Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes
        it to the servicer. This function should raise `ObjectIdNotPreregisteredError`
        if the object ID is not pre-registered.
    object_ids_to_push : Optional[set[str]] (default: None)
        A set of object IDs to push. If not provided, all objects will be pushed.
    keep_objects : bool (default: False)
        If `True`, the original objects will be kept in the `objects` dictionary
        after pushing. If `False`, they will be removed from the dictionary to avoid
        high memory usage.
    max_concurrent_pushes : int (default: FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES)
        The maximum number of concurrent pushes to perform.
    """
    lock = threading.Lock()

    def iter_dict_items() -> Iterator[tuple[str, bytes]]:
        """Iterate over the dictionary items."""
        for obj_id in list(objects.keys()):
            # Skip the object if no need to push it
            if object_ids_to_push is not None and obj_id not in object_ids_to_push:
                continue

            # Deflate the object content
            object_content = objects[obj_id].deflate()
            if not keep_objects:
                with lock:
                    del objects[obj_id]

            yield obj_id, object_content

    push_object_contents_from_iterable(
        iter_dict_items(),
        push_object_fn,
        max_concurrent_pushes=max_concurrent_pushes,
    )


def push_object_contents_from_iterable(
    object_contents: Iterable[tuple[str, bytes]],
    push_object_fn: Callable[[str, bytes], None],
    *,
    max_concurrent_pushes: int = FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES,
) -> None:
    """Push multiple object contents to the servicer.

    Parameters
    ----------
    object_contents : Iterable[tuple[str, bytes]]
        An iterable of `(object_id, object_content)` pairs.
        `object_id` is the object ID, and `object_content` is the object content.
    push_object_fn : Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes
        it to the servicer. This function should raise `ObjectIdNotPreregisteredError`
        if the object ID is not pre-registered.
    max_concurrent_pushes : int (default: FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES)
        The maximum number of concurrent pushes to perform.
    """
    error_event = threading.Event()
    err_queue: Queue[Exception] = Queue()

    def push(args: tuple[str, bytes]) -> None:
        """Push a single object."""
        if error_event.is_set():
            return
        obj_id, obj_content = args
        # Push the object using the provided function
        try:
            push_object_fn(obj_id, obj_content)
        except Exception as err:  # pylint: disable=broad-except
            # Unexpected error during pushing
            error_event.set()
            err_queue.put(err)

    # Push all object contents concurrently
    num_workers = get_num_workers(max_concurrent_pushes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Ensure that the thread pool executors are tracked for graceful shutdown
        _track_executor(executor)

        # Submit push tasks for each object content
        executor.map(push, object_contents)  # Non-blocking map

        # The context manager will block until all submitted tasks have completed

    # Remove the executor from the list of tracked executors
    _untrack_executor(executor)

    # If an error occurred during pushing, raise it
    if not err_queue.empty():
        raise err_queue.get()


def pull_objects(  # pylint: disable=too-many-arguments,too-many-locals
    object_ids: list[str],
    pull_object_fn: Callable[[str], bytes],
    *,
    max_concurrent_pulls: int = FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS,
    max_time: float | None = PULL_MAX_TIME,
    max_tries_per_object: int | None = PULL_MAX_TRIES_PER_OBJECT,
    initial_backoff: float = PULL_INITIAL_BACKOFF,
    backoff_cap: float = PULL_BACKOFF_CAP,
) -> dict[str, bytes]:
    """Pull multiple objects from the servicer.

    Parameters
    ----------
    object_ids : list[str]
        A list of object IDs to pull.
    pull_object_fn : Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
        The function should raise `ObjectUnavailableError` if the object is not yet
        available, or `ObjectIdNotPreregisteredError` if the object ID is not
        pre-registered.
    max_concurrent_pulls : int (default: FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS)
        The maximum number of concurrent pulls to perform.
    max_time : Optional[float] (default: PULL_MAX_TIME)
        The maximum time to wait for all pulls to complete. If `None`, waits
        indefinitely.
    max_tries_per_object : Optional[int] (default: PULL_MAX_TRIES_PER_OBJECT)
        The maximum number of attempts to pull each object. If `None`, pulls
        indefinitely until the object is available.
    initial_backoff : float (default: PULL_INITIAL_BACKOFF)
        The initial backoff time in seconds for retrying pulls after an
        `ObjectUnavailableError`.
    backoff_cap : float (default: PULL_BACKOFF_CAP)
        The maximum backoff time in seconds. Backoff times will not exceed this value.

    Returns
    -------
    dict[str, bytes]
        A dictionary where keys are object IDs and values are the pulled
        object contents.
    """
    if max_tries_per_object is None:
        max_tries_per_object = int(1e9)
    if max_time is None:
        max_time = float("inf")

    results: dict[str, bytes] = {}
    results_lock = threading.Lock()
    err_queue: Queue[Exception] = Queue()
    early_stop = threading.Event()
    start = time.monotonic()

    def stop_on_error(err: Exception) -> None:
        early_stop.set()
        err_queue.put(err)

    def pull_with_retries(object_id: str) -> None:
        """Attempt to pull a single object with retry and backoff."""
        tries = 0
        delay = initial_backoff

        while not early_stop.is_set():
            try:
                object_content = pull_object_fn(object_id)
                with results_lock:
                    results[object_id] = object_content
                return

            except ObjectUnavailableError as err:
                tries += 1
                if (
                    tries >= max_tries_per_object
                    or time.monotonic() - start >= max_time
                ):
                    # Stop all work if one object exhausts retries
                    stop_on_error(err)
                    return

                # Apply exponential backoff with ±20% jitter
                sleep_time = delay * (1 + random.uniform(-0.2, 0.2))
                early_stop.wait(sleep_time)
                delay = min(delay * 2, backoff_cap)

            except ObjectIdNotPreregisteredError as err:
                # Permanent failure: object ID is invalid
                stop_on_error(err)
                return

            except Exception as err:  # pylint: disable=broad-except
                # Permanent failure: unexpected error
                stop_on_error(err)
                return

    # Submit all pull tasks concurrently
    num_workers = get_num_workers(max_concurrent_pulls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Ensure that the thread pool executors are tracked for graceful shutdown
        _track_executor(executor)

        # Submit pull tasks for each object ID
        executor.map(pull_with_retries, object_ids)  # Non-blocking map

        # The context manager will block until all submitted tasks have completed

    # Remove the executor from the list of tracked executors
    _untrack_executor(executor)

    # If an error occurred during pulling, raise it
    if not err_queue.empty():
        raise err_queue.get()

    return results


def inflate_object_from_contents(
    object_id: str,
    object_contents: dict[str, bytes],
    *,
    keep_object_contents: bool = False,
    objects: dict[str, InflatableObject] | None = None,
) -> InflatableObject:
    """Inflate an object from object contents.

    Parameters
    ----------
    object_id : str
        The ID of the object to inflate.
    object_contents : dict[str, bytes]
        A dictionary mapping object IDs to their contents as bytes.
        All descendant objects must be present in this dictionary.
    keep_object_contents : bool (default: False)
        If `True`, the object content will be kept in the `object_contents`
        dictionary after inflation. If `False`, the object content will be
        removed from the dictionary to save memory.
    objects : Optional[dict[str, InflatableObject]] (default: None)
        No need to provide this parameter. A dictionary to store already
        inflated objects, mapping object IDs to their corresponding
        `InflatableObject` instances.

    Returns
    -------
    InflatableObject
        The inflated object.
    """
    if objects is None:
        # Initialize objects dictionary
        objects = {}

    if object_id in objects:
        # If the object is already in the objects dictionary, return it
        return objects[object_id]

    # Extract object class and object_ids of children
    object_content = object_contents[object_id]
    obj_type, children_obj_ids, _ = get_object_head_values_from_object_content(
        object_content=object_contents[object_id]
    )

    # Remove the object content from the dictionary to save memory
    if not keep_object_contents:
        del object_contents[object_id]

    # Resolve object class
    cls_type = inflatable_class_registry[obj_type]

    # Inflate all children objects
    children: dict[str, InflatableObject] = {}
    for child_obj_id in children_obj_ids:
        children[child_obj_id] = inflate_object_from_contents(
            child_obj_id,
            object_contents,
            keep_object_contents=keep_object_contents,
            objects=objects,
        )

    # Inflate object passing its children
    obj = cls_type.inflate(object_content, children=children)
    del object_content  # Free memory after inflation
    objects[object_id] = obj
    return obj


def validate_object_content(content: bytes) -> None:
    """Validate the deflated content of an InflatableObject."""
    try:
        # Check if there is a head-body divider
        index = content.find(HEAD_BODY_DIVIDER)
        if index == -1:
            raise ValueError(
                "Unexpected format for object content. Head and body "
                "could not be split."
            )

        head = _get_object_head(content)

        # check if the head has three parts:
        # <object_type> <children_ids> <object_body_len>
        head_decoded = head.decode(encoding="utf-8")
        head_parts = head_decoded.split(HEAD_VALUE_DIVIDER)

        if len(head_parts) != 3:
            raise ValueError("Unexpected format for object head.")

        obj_type, children_str, body_len = head_parts

        # Check that children IDs are valid IDs
        children = children_str.split(",")
        for children_id in children:
            if children_id and not is_valid_sha256_hash(children_id):
                raise ValueError(
                    f"Detected invalid object ID ({children_id}) in children."
                )

        # Check that object type is recognized
        if obj_type not in inflatable_class_registry:
            if obj_type != "CustomDataClass":  # to allow for the class in tests
                raise ValueError(f"Object of type {obj_type} is not supported.")

        # Check if the body length in the head matches that of the body
        actual_body_len = len(content) - len(head) - len(HEAD_BODY_DIVIDER)
        if actual_body_len != int(body_len):
            raise ValueError(
                f"Object content length expected {body_len} bytes but got "
                f"{actual_body_len} bytes."
            )

    except ValueError as err:
        raise UnexpectedObjectContentError(
            object_id=get_object_id(content), reason=str(err)
        ) from err


def pull_and_inflate_object_from_tree(  # pylint: disable=R0913
    object_tree: ObjectTree,
    pull_object_fn: Callable[[str], bytes],
    confirm_object_received_fn: Callable[[str], None],
    *,
    return_type: type[T] = InflatableObject,  # type: ignore
    max_concurrent_pulls: int = FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS,
    max_time: float | None = PULL_MAX_TIME,
    max_tries_per_object: int | None = PULL_MAX_TRIES_PER_OBJECT,
    initial_backoff: float = PULL_INITIAL_BACKOFF,
    backoff_cap: float = PULL_BACKOFF_CAP,
) -> T:
    """Pull and inflate the head object from the provided object tree.

    Parameters
    ----------
    object_tree : ObjectTree
        The object tree containing the object ID and its descendants.
    pull_object_fn : Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
    confirm_object_received_fn : Callable[[str], None]
        A function to confirm that the object has been received.
    return_type : type[T] (default: InflatableObject)
        The type of the object to return. Must be a subclass of `InflatableObject`.
    max_concurrent_pulls : int (default: FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS)
        The maximum number of concurrent pulls to perform.
    max_time : Optional[float] (default: PULL_MAX_TIME)
        The maximum time to wait for all pulls to complete. If `None`, waits
        indefinitely.
    max_tries_per_object : Optional[int] (default: PULL_MAX_TRIES_PER_OBJECT)
        The maximum number of attempts to pull each object. If `None`, pulls
        indefinitely until the object is available.
    initial_backoff : float (default: PULL_INITIAL_BACKOFF)
        The initial backoff time in seconds for retrying pulls after an
        `ObjectUnavailableError`.
    backoff_cap : float (default: PULL_BACKOFF_CAP)
        The maximum backoff time in seconds. Backoff times will not exceed this value.

    Returns
    -------
    T
        An instance of the specified return type containing the inflated object.
    """
    # Pull the main object and all its descendants
    pulled_object_contents = pull_objects(
        [tree.object_id for tree in iterate_object_tree(object_tree)],
        pull_object_fn,
        max_concurrent_pulls=max_concurrent_pulls,
        max_time=max_time,
        max_tries_per_object=max_tries_per_object,
        initial_backoff=initial_backoff,
        backoff_cap=backoff_cap,
    )

    # Confirm that all objects were pulled
    confirm_object_received_fn(object_tree.object_id)

    # Inflate the main object
    inflated_object = inflate_object_from_contents(
        object_tree.object_id, pulled_object_contents, keep_object_contents=False
    )

    # Check if the inflated object is of the expected type
    if not isinstance(inflated_object, return_type):
        raise TypeError(
            f"Expected object of type {return_type.__name__}, "
            f"but got {type(inflated_object).__name__}."
        )

    return inflated_object
