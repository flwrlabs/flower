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
"""Control API server."""


from logging import INFO

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.proto.control_pb2_grpc import add_ControlServicer_to_server
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import (
    ControlAuthnPlugin,
    ControlAuthzPlugin,
    NoOpControlAuthnPlugin,
)

from .control_account_auth_interceptor import ControlAccountAuthInterceptor
from .control_event_log_interceptor import ControlEventLogInterceptor
from .control_license_interceptor import ControlLicenseInterceptor
from .control_servicer import ControlServicer

try:
    from flwr.ee import get_license_plugin
except ImportError:

    def get_license_plugin() -> LicensePlugin | None:
        """Return the license plugin."""


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_control_api_grpc(
    address: str,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
    authn_plugin: ControlAuthnPlugin,
    authz_plugin: ControlAuthzPlugin,
    event_log_plugin: EventLogWriterPlugin | None = None,
    artifact_provider: ArtifactProvider | None = None,
    fleet_api_type: str | None = None,
) -> grpc.Server:
    """Run Control API (gRPC, request-response)."""
    license_plugin: LicensePlugin | None = get_license_plugin()
    if license_plugin and not license_plugin.check_license():
        flwr_exit(ExitCode.SUPERLINK_LICENSE_INVALID)

    control_servicer: grpc.Server = ControlServicer(
        linkstate_factory=state_factory,
        objectstore_factory=objectstore_factory,
        authn_plugin=authn_plugin,
        artifact_provider=artifact_provider,
        fleet_api_type=fleet_api_type,
    )
    interceptors = [ControlAccountAuthInterceptor(authn_plugin, authz_plugin)]
    if license_plugin is not None:
        interceptors.append(ControlLicenseInterceptor(license_plugin))
    # Event log interceptor must be added after account auth interceptor
    if event_log_plugin is not None:
        interceptors.append(ControlEventLogInterceptor(event_log_plugin))
        log(INFO, "Flower event logging enabled")
    control_add_servicer_to_server_fn = add_ControlServicer_to_server
    control_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(control_servicer, control_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors or None,
    )

    address = control_grpc_server.bound_address
    if isinstance(authn_plugin, NoOpControlAuthnPlugin):
        log(INFO, "Flower Deployment Runtime: Starting Control API on %s", address)
    else:
        log(
            INFO,
            "Flower Deployment Runtime: Starting Control API with account "
            "authentication on %s",
            address,
        )
    control_grpc_server.start()

    return control_grpc_server
