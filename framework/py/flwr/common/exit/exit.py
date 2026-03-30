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
"""Unified exit function."""


import os
import re
import sys
import threading
import time
from logging import ERROR, INFO
from typing import Any, NoReturn

from flwr.common import EventType, event
from flwr.supercore.constant import FORCE_EXIT_TIMEOUT_SECONDS
from flwr.supercore.version import package_version

from ..logger import log
from .exit_code import EXIT_CODE_HELP
from .exit_handler import trigger_exit_handlers


def _get_code_url(code: int) -> str:
    """Get the help URL for a given exit code."""
    if not (match := re.match(r"\d+\.\d+", package_version)):
        doc_pth = f"ref-exit-codes/{code}.html"  # Fallback for non-standard versions
    else:
        doc_pth = f"{match.group(0)}/en/ref-exit-codes/{code}.html"
    return f"https://flower.ai/docs/framework/{doc_pth}"


def flwr_exit(
    code: int,
    message: str | None = None,
    event_type: EventType | None = None,
    event_details: dict[str, Any] | None = None,
) -> NoReturn:
    """Handle application exit with an optional message.

    The exit message logged and displayed will follow this structure::

        Exit Code: <code>
        <message>
        <short-help-message>

        For more information, visit: <help-page-url>

    - `<code>`: The unique exit code representing the termination reason.
    - `<message>`: Optional context or additional information about the exit.
    - `<short-help-message>`: A brief explanation for the given exit code.
    - `<help-page-url>`: A URL providing detailed documentation and resolution steps.

    Notes
    -----
    This function MUST be called from the main thread.
    """
    is_error = not 0 <= code < 100  # 0-99 are success exit codes

    # Construct exit message
    exit_message = f"Exit Code: {code}\n" if is_error else ""
    exit_message += message or ""
    if short_help_message := EXIT_CODE_HELP.get(code, ""):
        exit_message += f"\n{short_help_message}"

    # Set log level and system exit code
    log_level = ERROR if is_error else INFO
    sys_exit_code = 1 if is_error else 0

    # Add help URL for non-successful/graceful exits
    if is_error:
        exit_message += f"\n\nFor more information, visit: <{_get_code_url(code)}>"

    # Telemetry event
    event_type = event_type or _try_obtain_telemetry_event()
    if event_type:
        event_details = event_details or {}
        event_details["exit_code"] = code
        event(event_type, event_details).result()

    # Log the exit message
    log(log_level, exit_message)

    # Start a daemon thread to force exit if graceful exit fails
    def force_exit() -> None:
        time.sleep(FORCE_EXIT_TIMEOUT_SECONDS)
        os._exit(sys_exit_code)

    threading.Thread(target=force_exit, daemon=True).start()

    # Trigger exit handlers
    trigger_exit_handlers()

    # Exit
    sys.exit(sys_exit_code)


# pylint: disable-next=too-many-return-statements
def _try_obtain_telemetry_event() -> EventType | None:
    """Try to obtain a telemetry event."""
    if sys.argv[0].endswith("flower-superlink"):
        return EventType.RUN_SUPERLINK_LEAVE
    if sys.argv[0].endswith("flower-supernode"):
        return EventType.RUN_SUPERNODE_LEAVE
    if sys.argv[0].endswith("flwr-serverapp"):
        return EventType.FLWR_SERVERAPP_RUN_LEAVE
    if sys.argv[0].endswith("flwr-clientapp"):
        return None  # Not yet implemented
    if sys.argv[0].endswith("flwr-simulation"):
        return EventType.FLWR_SIMULATION_RUN_LEAVE
    return None
