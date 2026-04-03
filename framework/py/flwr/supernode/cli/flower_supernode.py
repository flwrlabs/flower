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
"""`flower-supernode` command."""


import argparse
from logging import DEBUG, INFO, WARN
from pathlib import Path

import yaml
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.serialization import load_ssh_private_key
from cryptography.hazmat.primitives.serialization.ssh import load_ssh_public_key

from flwr.common import EventType, event
from flwr.common.args import try_obtain_root_certificates
from flwr.common.config import parse_config_args
from flwr.common.constant import (
    CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.supercore.grpc_health import add_args_health
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.version import package_version
from flwr.supernode.start_client_internal import start_client_internal


def flower_supernode() -> None:
    """Run Flower SuperNode."""
    warn_if_flwr_update_available(process_name="flower-supernode")

    args = _parse_args_run_supernode().parse_args()

    log(INFO, "Starting Flower SuperNode")

    event(EventType.RUN_SUPERNODE_ENTER)

    trusted_entities = _try_obtain_trusted_entities(args.trusted_entities)
    if trusted_entities:
        _validate_public_keys_ed25519(trusted_entities)
    root_certificates = try_obtain_root_certificates(args, args.superlink)
    authentication_keys = _try_setup_client_authentication(args)

    # Warn if authentication keys are provided but transport is not grpc-rere
    if authentication_keys is not None and args.transport != TRANSPORT_TYPE_GRPC_RERE:
        log(
            WARN,
            "SuperNode Authentication is only supported with the grpc-rere transport.",
        )

    log(DEBUG, "Isolation mode: %s", args.isolation)

    start_client_internal(
        server_address=args.superlink,
        transport=args.transport,
        root_certificates=root_certificates,
        insecure=args.insecure,
        authentication_keys=authentication_keys,
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
        node_config=parse_config_args(
            [args.node_config] if args.node_config else args.node_config
        ),
        isolation=args.isolation,
        clientappio_api_address=args.clientappio_api_address,
        health_server_address=args.health_server_address,
        trusted_entities=trusted_entities,
    )


def _parse_args_run_supernode() -> argparse.ArgumentParser:
    """Parse flower-supernode command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperNode",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Flower version: {package_version}",
    )
    _parse_args_common(parser)
    parser.add_argument(
        "--isolation",
        default=ISOLATION_MODE_SUBPROCESS,
        required=False,
        choices=[
            ISOLATION_MODE_SUBPROCESS,
            ISOLATION_MODE_PROCESS,
        ],
        help="Isolation mode when running a `ClientApp` (`subprocess` by default, "
        "possible values: `subprocess`, `process`). Use `subprocess` to configure "
        "SuperNode to run a `ClientApp` in a subprocess. Use `process` to indicate "
        "that a separate independent process gets created outside of SuperNode.",
    )
    parser.add_argument(
        "--clientappio-api-address",
        default=CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
        help="ClientAppIo API (gRPC) server address (IPv4, IPv6, or a domain name). "
        f"By default, it is set to {CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS}.",
    )
    parser.add_argument(
        "--trusted-entities",
        type=Path,
        default=None,
        metavar="YAML_FILE",
        help=(
            "Path to a YAML file defining trusted entities. "
            "The file must map public key IDs to public keys. "
            "Example: { fpk_UUID1: 'ssh-ed25519 <key1> [comment1]', "
            "fpk_UUID2: 'ssh-ed25519 <key2> [comment2]' }"
        ),
    )
    add_args_health(parser)

    return parser


def _parse_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the client without HTTPS. By default, the client runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    ex_group = parser.add_mutually_exclusive_group()
    ex_group.add_argument(
        "--grpc-rere",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_GRPC_RERE,
        default=TRANSPORT_TYPE_GRPC_RERE,
        help="Use grpc-rere as a transport layer for the client.",
    )
    ex_group.add_argument(
        "--grpc-adapter",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_GRPC_ADAPTER,
        help="Use grpc-adapter as a transport layer for the client.",
    )
    ex_group.add_argument(
        "--rest",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_REST,
        help="Use REST as a transport layer for the client.",
    )
    parser.add_argument(
        "--root-certificates",
        metavar="ROOT_CERT",
        type=str,
        help="Specifies the path to the PEM-encoded root certificate file for "
        "establishing secure HTTPS connections.",
    )
    parser.add_argument(
        "--superlink",
        default=FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
        help="SuperLink Fleet API address (IPv4, IPv6, or a domain name). If using the "
        "REST (experimental) transport, ensure your address is in the form "
        "`http://...` or `https://...` when TLS is enabled.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="The maximum number of times the client will try to reconnect to the"
        "SuperLink before giving up in case of a connection error. By default,"
        "it is set to None, meaning there is no limit to the number of tries.",
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=None,
        help="The maximum duration before the client stops trying to"
        "connect to the SuperLink in case of connection error. By default, it"
        "is set to None, meaning there is no limit to the total time.",
    )
    parser.add_argument(
        "--auth-supernode-private-key",
        type=str,
        help="Path to the SuperNode's private key to enable authentication.",
    )
    parser.add_argument(
        "--auth-supernode-public-key",
        type=str,
        help="This argument is deprecated and will be removed in a future release.",
    )
    parser.add_argument(
        "--node-config",
        type=str,
        help="A space separated list of key/value pairs (separated by `=`) to "
        "configure the SuperNode. "
        "E.g. --node-config 'key1=\"value1\" partition-id=0 num-partitions=100'",
    )


def _try_setup_client_authentication(
    args: argparse.Namespace,
) -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None:
    if not args.auth_supernode_private_key:
        return None

    try:
        ssh_private_key = load_ssh_private_key(
            Path(args.auth_supernode_private_key).expanduser().read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        flwr_exit(
            ExitCode.SUPERNODE_NODE_AUTH_KEY_INVALID,
            "Unable to parse the private key file.",
        )

    if args.auth_supernode_public_key:
        log(
            WARN,
            "The `--auth-supernode-public-key` flag is deprecated and will be "
            "removed in a future release. The public key is now derived from the "
            "private key provided by `--auth-supernode-private-key`.",
        )
    return ssh_private_key, ssh_private_key.public_key()


def _try_obtain_trusted_entities(
    trusted_entities_path: Path | None,
) -> dict[str, str] | None:
    """Validate and return the trust entities."""
    if not trusted_entities_path:
        return None
    trusted_entities_path = trusted_entities_path.expanduser()
    if not trusted_entities_path.is_file():
        flwr_exit(
            ExitCode.SUPERNODE_INVALID_TRUSTED_ENTITIES,
            "Path argument `--trusted-entities` does not point to a file.",
        )
    try:
        with trusted_entities_path.open("r", encoding="utf-8") as f:
            trusted_entities = yaml.safe_load(f)
        if not isinstance(trusted_entities, dict):
            raise ValueError("Invalid trusted entities format.")
    except (yaml.YAMLError, ValueError) as e:
        flwr_exit(
            ExitCode.SUPERNODE_INVALID_TRUSTED_ENTITIES,
            f"Failed to read YAML file '{trusted_entities_path}': {e}",
        )
    return trusted_entities


def _validate_public_keys_ed25519(trusted_entities: dict[str, str]) -> None:
    """Validate public keys for the trust entities are Ed25519."""
    for public_key_id in trusted_entities.keys():
        verifier_public_key = load_ssh_public_key(
            trusted_entities[public_key_id].encode("utf-8")
        )
        if not isinstance(verifier_public_key, ed25519.Ed25519PublicKey):
            flwr_exit(
                ExitCode.SUPERNODE_INVALID_TRUSTED_ENTITIES,
                "The provided public key associated with "
                f"trusted entity {public_key_id} is not Ed25519.",
            )
