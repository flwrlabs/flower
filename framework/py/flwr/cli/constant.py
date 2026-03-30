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
"""Constants for CLI commands."""


import os

from flwr.supercore.constant import SUPERGRID_ADDRESS

# General help message for config overrides
CONFIG_HELP_MESSAGE = (
    "Override {0} values using one of the following formats:\n\n"
    "--{1} '<k1>=<v1> <k2>=<v2>' | --{1} '<k1>=<v1>' --{1} '<k2>=<v2>'{2}\n\n"
    "When providing key-value pairs, values can be of any type supported by TOML "
    "(e.g., bool, int, float, string).{3}"
)

# The help message for `--run-config` option
RUN_CONFIG_HELP_MESSAGE = CONFIG_HELP_MESSAGE.format(
    "run configuration",
    "run-config",
    " | --run-config <path/to/your/toml>",
    "The specified keys (<k1> and <k2> in the example) must exist in the "
    "run configuration under the `[tool.flwr.app.config]` section of "
    "`pyproject.toml` to be overridden. Alternatively, provide a TOML file "
    "containing key-value pair overrides.",
)

# The help message for `--federation-config` option
FEDERATION_CONFIG_HELP_MESSAGE = CONFIG_HELP_MESSAGE.format(
    "federation configuration",
    "federation-config",
    "",
    "",
)


class SuperLinkConnectionTomlKey:
    """TOML keys for SuperLink connection configuration."""

    SUPERLINK = "superlink"
    DEFAULT = "default"
    ADDRESS = "address"
    ROOT_CERTIFICATES = "root-certificates"
    INSECURE = "insecure"
    FEDERATION = "federation"
    OPTIONS = "options"


class SuperLinkSimulationOptionsTomlKey:
    """TOML keys for SuperLinkSimulationOptions."""

    NUM_SUPERNODES = "num-supernodes"
    BACKEND = "backend"
    VERBOSE = "verbose"


class SimulationClientResourcesTomlKey:
    """TOML keys for SimulationClientResources."""

    NUM_CPUS = "num-cpus"
    NUM_GPUS = "num-gpus"


class SimulationInitArgsTomlKey:
    """TOML keys for SimulationInitArgs."""

    NUM_CPUS = "num-cpus"
    NUM_GPUS = "num-gpus"
    LOGGING_LEVEL = "logging-level"
    LOG_TO_DRIVER = "log-to-driver"


class SimulationBackendConfigTomlKey:
    """TOML keys for SimulationBackendConfig."""

    CLIENT_RESOURCES = "client-resources"
    INIT_ARGS = "init-args"
    NAME = "name"


# Local SuperLink configuration
LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE = ":local:"
LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE_IN_MEMORY = ":local-in-memory:"
LOCAL_CONTROL_API_PORT = os.environ.get("FLWR_LOCAL_CONTROL_API_PORT", "39093")
LOCAL_CONTROL_API_ADDRESS = f"127.0.0.1:{LOCAL_CONTROL_API_PORT}"
LOCAL_SUPERLINK_STARTUP_TIMEOUT = 15.0
CONTROL_API_PROBE_TIMEOUT = 0.4
CONTROL_API_PROBE_INTERVAL = 0.2

# CLI connection configuration file name
FLOWER_CONFIG_FILE = "config.toml"

# The default configuration for the Flower config file
DEFAULT_FLOWER_CONFIG_TOML = f"""[superlink]
default = "local"

[superlink.supergrid]
address = "{SUPERGRID_ADDRESS}"

[superlink.local]
address = "{LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE}"
"""

# Keys for storing account auth credentials in the credential store
AUTHN_TYPE_STORE_KEY = "flower.account-auth.%s.authn-type"
ACCESS_TOKEN_STORE_KEY = "flower.account-auth.%s.oidc-access-token"
REFRESH_TOKEN_STORE_KEY = "flower.account-auth.%s.oidc-refresh-token"
