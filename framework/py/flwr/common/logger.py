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
"""Compatibility exports for Flower logging utilities."""

from flwr.supercore.logger import FLOWER_LOGGER as FLOWER_LOGGER
from flwr.supercore.logger import LOGGER_NAME as LOGGER_NAME
from flwr.supercore.logger import ConsoleHandler as ConsoleHandler
from flwr.supercore.logger import CustomHTTPHandler as CustomHTTPHandler
from flwr.supercore.logger import configure as configure
from flwr.supercore.logger import (
    configure_superlink_log_file as configure_superlink_log_file,
)
from flwr.supercore.logger import console_handler as console_handler
from flwr.supercore.logger import flush_logs as flush_logs
from flwr.supercore.logger import log as log
from flwr.supercore.logger import mirror_output_to_queue as mirror_output_to_queue
from flwr.supercore.logger import print_json_error as print_json_error
from flwr.supercore.logger import redirect_output as redirect_output
from flwr.supercore.logger import restore_output as restore_output
from flwr.supercore.logger import set_logger_propagation as set_logger_propagation
from flwr.supercore.logger import start_log_uploader as start_log_uploader
from flwr.supercore.logger import stop_log_uploader as stop_log_uploader
from flwr.supercore.logger import update_console_handler as update_console_handler
from flwr.supercore.logger import warn_deprecated_feature as warn_deprecated_feature
from flwr.supercore.logger import (
    warn_deprecated_feature_with_example as warn_deprecated_feature_with_example,
)
from flwr.supercore.logger import warn_preview_feature as warn_preview_feature
from flwr.supercore.logger import warn_unsupported_feature as warn_unsupported_feature

__all__ = [
    "ConsoleHandler",
    "CustomHTTPHandler",
    "FLOWER_LOGGER",
    "LOGGER_NAME",
    "configure",
    "configure_superlink_log_file",
    "console_handler",
    "flush_logs",
    "log",
    "mirror_output_to_queue",
    "print_json_error",
    "redirect_output",
    "restore_output",
    "set_logger_propagation",
    "start_log_uploader",
    "stop_log_uploader",
    "update_console_handler",
    "warn_deprecated_feature",
    "warn_deprecated_feature_with_example",
    "warn_preview_feature",
    "warn_unsupported_feature",
]
