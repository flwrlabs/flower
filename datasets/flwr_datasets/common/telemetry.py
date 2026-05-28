# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower telemetry."""


import datetime
import json
import logging
import os
import platform
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from pathlib import Path
from typing import Any, cast

from flwr_datasets.common.version import package_name, package_version

FLWR_TELEMETRY_ENABLED = os.getenv("FLWR_TELEMETRY_ENABLED", "1")
FLWR_TELEMETRY_LOGGING = os.getenv("FLWR_TELEMETRY_LOGGING", "0")

TELEMETRY_EVENTS_URL = "https://telemetry.flower.ai/api/v1/event"

LOGGER_NAME = "flwr-datasets-telemetry"
LOGGER_LEVEL = logging.DEBUG


def _configure_logger(log_level: int) -> None:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
        )
    )

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)


_configure_logger(LOGGER_LEVEL)


def log(msg: str | Exception) -> None:
    """Log message using logger at DEBUG level."""
    logging.getLogger(LOGGER_NAME).log(LOGGER_LEVEL, msg)


def _get_home() -> Path:
    return Path().home()


def _get_source_id() -> str:
    """Get existing or new source ID."""
    source_id = "unavailable"
    # Check if .flwr in home exists
    try:
        home = _get_home()
    except RuntimeError:
        # If the home directory can’t be resolved, RuntimeError is raised.
        return source_id

    flwr_dir = home.joinpath(".flwr")
    # Create .flwr directory if it does not exist yet.
    try:
        flwr_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return source_id

    source_file = flwr_dir.joinpath("source")

    # If no source_file exists create one and write it
    if not source_file.exists():
        try:
            source_file.touch(exist_ok=True)
            source_file.write_text(str(uuid.uuid4()), encoding="utf-8")
        except PermissionError:
            return source_id

    source_id = source_file.read_text(encoding="utf-8").strip()

    try:
        uuid.UUID(source_id)
    except ValueError:
        source_id = "invalid"

    return source_id


# Using str as first base type to make it JSON serializable as
# otherwise the following exception will be thrown when serializing
# the event dict:
# TypeError: Object of type EventType is not JSON serializable
class EventType(str, Enum):
    """Types of telemetry events."""

    # This method combined with auto() will set the property value to
    # the property name e.g.
    # `START_CLIENT = auto()` becomes `START_CLIENT = "START_CLIENT"`
    # The type signature is not compatible with mypy and pylint
    # so each of those needs to be disabled for this line.
    # pylint: disable-next=no-self-argument,arguments-differ,line-too-long
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> Any:  # type: ignore # noqa: E501
        return name

    PING = auto()

    LOAD_PARTITION_CALLED = auto()
    LOAD_SPLIT_CALLED = auto()
    PLOT_LABEL_DISTRIBUTION_CALLED = auto()
    PLOT_COMPARISON_LABEL_DISTRIBUTION_CALLED = auto()


# Use the ThreadPoolExecutor with max_workers=1 to have a queue
# and also ensure that telemetry calls are not blocking.
state: dict[str, str | None | ThreadPoolExecutor | None] = {
    # Will be assigned ThreadPoolExecutor(max_workers=1)
    # in event() the first time it's required
    "executor": None,
    "source": None,
    "cluster": None,
}


# In Python 3.7 pylint will throw an error stating that
# "Value 'Future' is unsubscriptable".
# This pylint disable line can be remove when dropping support
# for Python 3.7
# pylint: disable-next=unsubscriptable-object
def event(
    event_type: EventType,
    event_details: dict[str, Any] | None = None,
) -> Future:  # type: ignore
    """Submit create_event to ThreadPoolExecutor to avoid blocking."""
    if state["executor"] is None:
        state["executor"] = ThreadPoolExecutor(max_workers=1)

    executor: ThreadPoolExecutor = cast(ThreadPoolExecutor, state["executor"])

    result = executor.submit(create_event, event_type, event_details)
    return result


def create_event(event_type: EventType, event_details: dict[str, Any] | None) -> str:
    """Create telemetry event."""
    if state["source"] is None:
        state["source"] = _get_source_id()

    if state["cluster"] is None:
        state["cluster"] = str(uuid.uuid4())

    if event_details is None:
        event_details = {}

    date = datetime.datetime.now(tz=datetime.UTC).isoformat()
    context = {
        "source": state["source"],
        "cluster": state["cluster"],
        "date": date,
        "package": {
            "package_name": package_name,
            "package_version": package_version,
        },
        "hw": {
            "cpu_count": os.cpu_count(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "architecture": platform.architecture(),
            "version": platform.uname().version,
        },
    }
    payload = {
        "event_type": event_type,
        "event_details": event_details,
        "context": context,
    }
    payload_json = json.dumps(payload)
    if FLWR_TELEMETRY_LOGGING == "1":
        log(" - ".join([date, "POST", payload_json]))

    # If telemetry is not disabled with setting FLWR_TELEMETRY_ENABLED=0
    # create a request and send it to the telemetry backend
    if FLWR_TELEMETRY_ENABLED == "1":
        request = urllib.request.Request(
            url=TELEMETRY_EVENTS_URL,
            data=payload_json.encode("utf-8"),
            headers={
                "User-Agent": f"{package_name}/{package_version}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = response.read()

            response_json: str = result.decode("utf-8")

            return response_json
        except urllib.error.URLError as ex:
            if FLWR_TELEMETRY_LOGGING == "1":
                log(ex)

    return "disabled"
