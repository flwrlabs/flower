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
"""Exit codes."""


from __future__ import annotations


class ExitCode:
    """Exit codes for Flower components."""

    # Success exit codes (0-99)
    SUCCESS = 0  # Successful exit without any errors or signals
    GRACEFUL_EXIT_SIGINT = 1  # Graceful exit triggered by SIGINT
    GRACEFUL_EXIT_SIGQUIT = 2  # Graceful exit triggered by SIGQUIT
    GRACEFUL_EXIT_SIGTERM = 3  # Graceful exit triggered by SIGTERM

    # SuperLink-specific exit codes (100-199)
    SUPERLINK_THREAD_CRASH = 100
    SUPERLINK_LICENSE_INVALID = 101
    SUPERLINK_LICENSE_MISSING = 102
    SUPERLINK_LICENSE_URL_INVALID = 103
    SUPERLINK_INVALID_ARGS = 104
    SUPERLINK_DATABASE_SCHEMA_MISMATCH = 105

    # ServerApp-specific exit codes (200-299)
    SERVERAPP_STRATEGY_PRECONDITION_UNMET = 200
    SERVERAPP_EXCEPTION = 201
    SERVERAPP_STRATEGY_AGGREGATION_ERROR = 202
    SERVERAPP_RUN_START_REJECTED = 203

    # SuperNode-specific exit codes (300-399)
    SUPERNODE_REST_ADDRESS_INVALID = 300
    # SUPERNODE_NODE_AUTH_KEYS_REQUIRED = 301 --- DELETED ---
    SUPERNODE_NODE_AUTH_KEY_INVALID = 302
    SUPERNODE_STARTED_WITHOUT_TLS_BUT_NODE_AUTH_ENABLED = 303
    SUPERNODE_INVALID_TRUSTED_ENTITIES = 304

    # SuperExec-specific exit codes (400-499)
    SUPEREXEC_INVALID_PLUGIN_CONFIG = 400

    # FlowerCLI-specific exit codes (500-599)
    FLWRCLI_NODE_AUTH_PUBLIC_KEY_INVALID = 500

    # Common exit codes (600-699)
    COMMON_ADDRESS_INVALID = 600
    COMMON_MISSING_EXTRA_REST = 601
    COMMON_TLS_NOT_SUPPORTED = 602

    # Simulation exit codes (700-799)
    SIMULATION_EXCEPTION = 700
    SIMULATION_MISSING_EXTRA = 701

    def __new__(cls) -> ExitCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


# All short help messages for exit codes
EXIT_CODE_HELP = {
    # Success exit codes (0-99)
    ExitCode.SUCCESS: "",
    ExitCode.GRACEFUL_EXIT_SIGINT: "",
    ExitCode.GRACEFUL_EXIT_SIGQUIT: "",
    ExitCode.GRACEFUL_EXIT_SIGTERM: "",
    # SuperLink-specific exit codes (100-199)
    ExitCode.SUPERLINK_THREAD_CRASH: "An important background thread has crashed.",
    ExitCode.SUPERLINK_LICENSE_INVALID: (
        "The license is invalid or has expired. "
        "Please contact `hello@flower.ai` for assistance."
    ),
    ExitCode.SUPERLINK_LICENSE_MISSING: (
        "The license is missing. Please specify the license key by setting the "
        "environment variable `FLWR_LICENSE_KEY`."
    ),
    ExitCode.SUPERLINK_LICENSE_URL_INVALID: (
        "The license URL is invalid. Please ensure that the `FLWR_LICENSE_URL` "
        "environment variable is set to a valid URL."
    ),
    ExitCode.SUPERLINK_INVALID_ARGS: (
        "Invalid arguments provided to SuperLink. Use `--help` check for the correct "
        "usage. Alternatively, check the documentation."
    ),
    ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH: (
        "The database schema does not match the expected schema for this version of "
        "SuperLink. Please refer to the documentation for guidance on how to resolve "
        "this issue."
    ),
    # ServerApp-specific exit codes (200-299)
    ExitCode.SERVERAPP_STRATEGY_PRECONDITION_UNMET: (
        "The strategy received replies that cannot be aggregated. Please ensure all "
        "replies returned by ClientApps have one `ArrayRecord` (none when replies are "
        "from a round of federated evaluation, i.e. when message type is "
        "`MessageType.EVALUATE`) and one `MetricRecord`. The records in all replies "
        "must use identical keys. In addition, if the strategy expects a key to "
        "perform weighted average (e.g. in FedAvg) please ensure the returned "
        "MetricRecord from ClientApps do include this key."
    ),
    ExitCode.SERVERAPP_EXCEPTION: "An unhandled exception occurred in the ServerApp.",
    ExitCode.SERVERAPP_STRATEGY_AGGREGATION_ERROR: (
        "The strategy encountered an error during aggregation. Please check the logs "
        "for more details."
    ),
    ExitCode.SERVERAPP_RUN_START_REJECTED: (
        "The SuperLink rejected the request to start the run. This may occur if the "
        "run has been stopped, the run ID or FAB is invalid, or the run failed to "
        "start within the allowed time."
    ),
    # SuperNode-specific exit codes (300-399)
    ExitCode.SUPERNODE_REST_ADDRESS_INVALID: (
        "When using the REST API, please provide `https://` or "
        "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
    ),
    ExitCode.SUPERNODE_NODE_AUTH_KEY_INVALID: (
        "Node authentication requires elliptic curve private key. "
        "Please ensure that the file path points to a valid private key "
        "file and try again."
    ),
    ExitCode.SUPERNODE_STARTED_WITHOUT_TLS_BUT_NODE_AUTH_ENABLED: (
        "The private key for SuperNode authentication was provided, but TLS is not "
        "enabled. Node authentication can only be used when TLS is enabled."
    ),
    ExitCode.SUPERNODE_INVALID_TRUSTED_ENTITIES: (
        "Failed to read the trusted entities YAML file. "
        "Please ensure that a valid file is provided using "
        "the `--trusted-entities` option."
    ),
    # SuperExec-specific exit codes (400-499)
    ExitCode.SUPEREXEC_INVALID_PLUGIN_CONFIG: (
        "The YAML configuration for the SuperExec plugin is invalid."
    ),
    # FlowerCLI-specific exit codes (500-599)
    ExitCode.FLWRCLI_NODE_AUTH_PUBLIC_KEY_INVALID: (
        "Node authentication requires a valid elliptic curve public key in the "
        "SSH format and following a NIST standard elliptic curve (e.g. SECP384R1). "
        "Please ensure that the file path points to a valid public key "
        "file and try again."
    ),
    # Common exit codes (600-699)
    ExitCode.COMMON_ADDRESS_INVALID: (
        "Please provide a valid URL, IPv4 or IPv6 address."
    ),
    ExitCode.COMMON_MISSING_EXTRA_REST: """
Extra dependencies required for using the REST-based Fleet API are missing.

To use the REST API, install `flwr` with the `rest` extra:

    `pip install "flwr[rest]"`.
""",
    ExitCode.COMMON_TLS_NOT_SUPPORTED: "Please use the '--insecure' flag.",
    # Simulation exit codes (700-799)
    ExitCode.SIMULATION_EXCEPTION: (
        "An unhandled exception occurred when running the simulation."
    ),
    ExitCode.SIMULATION_MISSING_EXTRA: """
Extra dependencies required for simulation are missing.

To use simulation with the Ray backend, install `flwr` with the `simulation` extra:

    `pip install "flwr[simulation]"`.
""",
}
