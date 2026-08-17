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
"""Runtime gRPC API hosted by SuperLink."""


from logging import INFO, WARNING

import grpc

from flwr.common.logger import log
from flwr.proto.runtime_pb2_grpc import (  # pylint: disable=E0611
    add_RuntimeServicer_to_server,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.grpc import GRPC_MAX_MESSAGE_LENGTH, generic_create_grpc_server
from flwr.supercore.interceptors import (
    RpcErrorTranslationServerInterceptor,
    create_superlink_runtime_superexec_auth_server_interceptor,
    create_superlink_runtime_token_auth_server_interceptor,
    create_superlink_runtime_version_server_interceptor,
)
from flwr.supercore.object_store import ObjectStoreFactory

from .runtime_servicer import SuperLinkRuntimeServicer


def run_runtime_api_grpc(  # pylint: disable=R0913,R0917
    address: str,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
    superexec_auth_secret: bytes | None = None,
) -> grpc.Server:
    """Run the Runtime API (gRPC, request-response)."""
    if superexec_auth_secret is not None and certificates is None:
        log(
            WARNING,
            "SuperExec auth is enabled on insecure Runtime API transport. "
            "Request metadata confidentiality is not guaranteed without TLS.",
        )

    # Create Runtime API gRPC server
    runtime_servicer = SuperLinkRuntimeServicer(
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
    )

    # Create interceptors
    interceptors = [
        RpcErrorTranslationServerInterceptor(),
        create_superlink_runtime_token_auth_server_interceptor(
            state_provider=state_factory.state
        ),
    ]
    if superexec_auth_secret is not None:
        interceptors.append(
            create_superlink_runtime_superexec_auth_server_interceptor(
                state_provider=state_factory.state,
                master_secret=superexec_auth_secret,
            )
        )
    interceptors.append(create_superlink_runtime_version_server_interceptor())
    runtime_add_servicer_to_server_fn = add_RuntimeServicer_to_server
    runtime_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            runtime_servicer,
            runtime_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors,
    )

    address = runtime_grpc_server.bound_address
    log(INFO, "Flower Deployment Runtime: Starting Runtime API on %s", address)
    runtime_grpc_server.start()

    return runtime_grpc_server
