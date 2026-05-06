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
"""ServerAppIo gRPC API."""


from logging import INFO, WARNING

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.proto.appio_pb2 import CreateTaskRequest, CreateTaskResponse  # pylint: disable=E0611
from flwr.proto.appio_pb2_grpc import add_AppIoServicer_to_server  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import (  # pylint: disable=E0611
    add_ServerAppIoServicer_to_server,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.interceptors import (
    create_serverappio_runtime_version_server_interceptor,
    create_serverappio_superexec_auth_server_interceptor,
    create_serverappio_token_auth_server_interceptor,
)
from flwr.supercore.object_store import ObjectStoreFactory

from .serverappio_servicer import ServerAppIoServicer


def run_serverappio_api_grpc(  # pylint: disable=R0913,R0917
    address: str,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
    superexec_auth_secret: bytes | None = None,
) -> grpc.Server:
    """Run ServerAppIo API (gRPC, request-response)."""
    if superexec_auth_secret is not None and certificates is None:
        log(
            WARNING,
            "SuperExec auth is enabled on insecure ServerAppIo transport. "
            "Request metadata confidentiality is not guaranteed without TLS.",
        )

    # Create ServerAppIo API gRPC server
    serverappio_servicer = ServerAppIoServicer(
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
    )

    # Create interceptors
    interceptors = [
        create_serverappio_token_auth_server_interceptor(
            state_provider=state_factory.state
        )
    ]
    if superexec_auth_secret is not None:
        interceptors.append(
            create_serverappio_superexec_auth_server_interceptor(
                state_provider=state_factory.state,
                master_secret=superexec_auth_secret,
            )
        )
    interceptors.append(create_serverappio_runtime_version_server_interceptor())
    serverappio_add_servicer_to_server_fn = add_ServerAppIoServicer_to_server
    serverappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            serverappio_servicer,
            serverappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors,
    )
    add_AppIoServicer_to_server(serverappio_servicer, serverappio_grpc_server)
    _add_legacy_create_task_handler(serverappio_servicer, serverappio_grpc_server)

    address = serverappio_grpc_server.bound_address
    log(INFO, "Flower Deployment Runtime: Starting ServerAppIo API on %s", address)
    serverappio_grpc_server.start()

    return serverappio_grpc_server


def _add_legacy_create_task_handler(
    servicer: ServerAppIoServicer, server: grpc.Server
) -> None:
    """Register a compatibility handler for ServerAppIo.CreateTask."""
    rpc_method_handlers = {
        "CreateTask": grpc.unary_unary_rpc_method_handler(
            servicer.CreateTask,
            request_deserializer=CreateTaskRequest.FromString,
            response_serializer=CreateTaskResponse.SerializeToString,
        )
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "flwr.proto.ServerAppIo", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers("flwr.proto.ServerAppIo", rpc_method_handlers)
