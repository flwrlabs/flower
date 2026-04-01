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
"""Constants for Flower infrastructure."""


from __future__ import annotations

import os
from enum import Enum

from flwr.common.constant import FLWR_DIR, NOOP_ACCOUNT_NAME
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611

# Constants for Inflatable
HEAD_BODY_DIVIDER = b"\x00"
HEAD_VALUE_DIVIDER = " "

# Constants for object pushing and pulling
FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES = int(
    os.getenv("FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES", "2")
)  # Default maximum number of concurrent pushes
FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS = int(
    os.getenv("FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PULLS", "2")
)  # Default maximum number of concurrent pulls
PULL_MAX_TIME = 7200  # Default maximum time to wait for pulling objects
PULL_MAX_TRIES_PER_OBJECT = 500  # Default maximum number of tries to pull an object
PULL_INITIAL_BACKOFF = 1  # Initial backoff time for pulling objects
PULL_BACKOFF_CAP = 10  # Maximum backoff time for pulling objects

# Top-level key in YAML config for exec plugin settings
EXEC_PLUGIN_SECTION = "exec_plugin"

# Flower in-memory Python-based database name
FLWR_IN_MEMORY_DB_NAME = ":flwr-in-memory:"

# Flower in-memory SQLite database URL
FLWR_IN_MEMORY_SQLITE_DB_URL = "sqlite:///:memory:"

# Constants for Hub
APP_ID_PATTERN = r"^@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
APP_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"
PLATFORM_API_URL = "https://api.flower.ai/v1"

# Constants for Flower CLI update check
FLWR_DISABLE_UPDATE_CHECK = "FLWR_DISABLE_UPDATE_CHECK"
FLWR_UPDATE_CHECK_URL = f"{PLATFORM_API_URL}/update-check/flwr"
FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS = 1
FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS = 2
FLWR_UPDATE_CHECK_CACHE_DIR = ".cache"
FLWR_UPDATE_CHECK_CACHE_FILENAME = "update-check.json"
FLWR_UPDATE_CHECK_SHOW_INTERVAL_SECONDS = 12 * 60 * 60

# SuperGrid constants
SUPERGRID_ADDRESS = "supergrid.flower.ai"

# Specification for app publishing
APP_PUBLISH_ALLOWED_LICENSE_FILES = ("LICENSE", "LICENSE.md")
APP_PUBLISH_INCLUDE_PATTERNS = (
    "**/*.py",
    "**/*.toml",
    "**/*.md",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
    "**/*.jsonl",
    "/.gitignore",
    "**/.editorconfig",
    "/LICENSE",
    "/LICENSE.md",
)
APP_PUBLISH_EXCLUDE_PATTERNS = (
    f"{FLWR_DIR}/**",  # Exclude the .flwr directory
    "**/__pycache__/**",
)
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_FILE_COUNT = 1000
MAX_DIR_DEPTH = 10  # relative depth (number of parts in relpath)
UTF8 = "utf-8"
MIME_MAP = {
    ".py": "text/x-python; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".toml": "application/toml; charset=utf-8",
}

# Constants for federations
NOOP_FEDERATION = f"@{NOOP_ACCOUNT_NAME}/default"
NOOP_FEDERATION_DESCRIPTION = "A federation for testing and development purposes."
DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    num_supernodes=10,
    client_resources_num_cpus=2,
    client_resources_num_gpus=0.0,
    backend="ray",
    verbose=False,
    init_args_num_cpus=None,
    init_args_num_gpus=None,
    init_args_logging_level="WARNING",
    init_args_log_to_driver=True,
)

# Constants for exit handling
FORCE_EXIT_TIMEOUT_SECONDS = 5  # Used in `flwr_exit` function

# Constants for message processing timing
MESSAGE_TIME_ENTRY_MAX_AGE_SECONDS = 3600


# System message type
SYSTEM_MESSAGE_TYPE = "system"

# SQLite PRAGMA settings for optimal performance and correctness
SQLITE_PRAGMAS = (
    ("busy_timeout", "5000"),  # Retry lock acquisition for up to 5s before SQLITE_BUSY
    ("journal_mode", "WAL"),  # Enable Write-Ahead Logging for better concurrency
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("cache_size", "-64000"),  # 64MB cache
    ("temp_store", "MEMORY"),  # In-memory temp tables
    ("mmap_size", "268435456"),  # 256MB memory-mapped I/O
)


class NodeStatus:
    """Event log writer types."""

    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"
    UNREGISTERED = "unregistered"

    def __new__(cls) -> NodeStatus:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class InvitationStatus(str, Enum):
    """Status of a federation invitation."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVOKED = "revoked"
    EXPIRED = "expired"


class RunType(str, Enum):
    """Supported run types."""

    SERVER_APP = "serverapp"
    SIMULATION = "simulation"


class RunTime(str, Enum):
    """Supported runtimes."""

    DEPLOYMENT = "deployment"
    SIMULATION = "simulation"


class ActionType(str, Enum):
    """Supported control action types."""

    START_RUN = "start_run"
    CREATE_FEDERATION = "create_federation"
