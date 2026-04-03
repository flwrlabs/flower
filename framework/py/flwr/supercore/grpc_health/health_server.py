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
"""Health servers."""


import argparse
from logging import INFO

import grpc
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log

from .simple_health_servicer import SimpleHealthServicer


def run_health_server_grpc_no_tls(address: str) -> grpc.Server:
    """Run gRPC health server with no TLS."""
    health_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            SimpleHealthServicer(),
            add_HealthServicer_to_server,
        ),
        server_address=address,
        certificates=None,
    )
    log(INFO, "Starting gRPC health server on %s", health_server.bound_address)
    health_server.start()
    return health_server


def add_args_health(parser: argparse.ArgumentParser) -> None:
    """Add arguments for health server."""
    parser.add_argument(
        "--health-server-address",
        type=str,
        default=None,
        help="Health service gRPC server address (IPv4, IPv6, or a domain name) "
        "with no TLS. If not set, the health server will not be started.",
    )
