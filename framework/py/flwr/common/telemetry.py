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
from typing import Any, cast

from flwr.supercore.utils import get_flwr_home
from flwr.supercore.version import package_name, package_version

FLWR_TELEMETRY_ENABLED = os.getenv("FLWR_TELEMETRY_ENABLED", "1")
FLWR_TELEMETRY_LOGGING = os.getenv("FLWR_TELEMETRY_LOGGING", "0")

TELEMETRY_EVENTS_URL = "https://telemetry.flower.ai/api/v1/event"

LOGGER_NAME = "flwr-telemetry"
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


def _get_partner_id() -> str:
    """Get partner ID."""
    partner_id = os.getenv("FLWR_TELEMETRY_PARTNER_ID")
    if not partner_id:
        return "unavailable"
    try:
        uuid.UUID(partner_id)
    except ValueError:
        partner_id = "invalid"
    return partner_id


def _get_source_id() -> str:
    """Get existing or new source ID."""
    source_id = "unavailable"
    # Check if .flwr in home exists
    try:
        flwr_home = get_flwr_home()
        # Create .flwr directory if it does not exist yet.
        flwr_home.mkdir(parents=True, exist_ok=True)
    except (RuntimeError, PermissionError):
        # If the directory can’t be resolved or created, an error is raised.
        return source_id

    source_file = flwr_home.joinpath("source")

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

    # Ping
    PING = auto()

    # --- LEGACY FUNCTIONS -------------------------------------------------------------

    # Legacy: `start_client` function
    START_CLIENT_ENTER = auto()
    START_CLIENT_LEAVE = auto()

    # Legacy: `start_server` function
    START_SERVER_ENTER = auto()
    START_SERVER_LEAVE = auto()

    # Legacy: `start_simulation` function
    START_SIMULATION_ENTER = auto()
    START_SIMULATION_LEAVE = auto()

    # --- `flwr` CLI -------------------------------------------------------------------

    # Not yet implemented

    # --- `flwr-*` commands ------------------------------------------------------------

    # CLI: flwr-simulation
    FLWR_SIMULATION_RUN_ENTER = auto()
    FLWR_SIMULATION_RUN_LEAVE = auto()

    # CLI: flwr-serverapp
    FLWR_SERVERAPP_RUN_ENTER = auto()
    FLWR_SERVERAPP_RUN_LEAVE = auto()

    # CLI: flwr-clientapp
    FLWR_CLIENTAPP_RUN_ENTER = auto()
    FLWR_CLIENTAPP_RUN_LEAVE = auto()

    # --- Simulation Engine ------------------------------------------------------------

    # Python API: `run_simulation`
    PYTHON_API_RUN_SIMULATION_ENTER = auto()
    PYTHON_API_RUN_SIMULATION_LEAVE = auto()

    # --- Deployment Engine ------------------------------------------------------------

    # CLI: `flower-superlink`
    RUN_SUPERLINK_ENTER = auto()
    RUN_SUPERLINK_LEAVE = auto()

    # CLI: `flower-supernode`
    RUN_SUPERNODE_ENTER = auto()
    RUN_SUPERNODE_LEAVE = auto()

    # CLI: `flower-superexec`
    RUN_SUPEREXEC_ENTER = auto()
    RUN_SUPEREXEC_LEAVE = auto()


# Use the ThreadPoolExecutor with max_workers=1 to have a queue
# and also ensure that telemetry calls are not blocking.
state: dict[str, str | None | ThreadPoolExecutor | None] = {
    # Will be assigned ThreadPoolExecutor(max_workers=1)
    # in event() the first time it's required
    "executor": None,
    "source": None,
    "cluster": None,
    "partner": None,
}


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

    if state["partner"] is None:
        state["partner"] = _get_partner_id()

    if event_details is None:
        event_details = {}

    date = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    context = {
        "partner": state["partner"],
        "source": state["source"],
        "cluster": state["cluster"],
        "date": date,
        "flower": {
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
