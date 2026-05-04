# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Token auth policy definitions for AppIo interfaces."""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodTokenPolicy:
    """Token requirement for a single unary RPC method."""

    requires_token: bool

    @staticmethod
    def no_auth() -> MethodTokenPolicy:
        """Return policy for methods that should remain unauthenticated."""
        return MethodTokenPolicy(requires_token=False)

    @staticmethod
    def token_required() -> MethodTokenPolicy:
        """Return policy for methods protected by App token auth."""
        return MethodTokenPolicy(requires_token=True)


# In a follow-up PR, create explicit method maps using a shared builder.
SERVERAPPIO_METHOD_AUTH_POLICY: dict[str, MethodTokenPolicy] = {
    "/flwr.proto.ServerAppIo/ListAppsToLaunch": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ServerAppIo/RequestToken": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ServerAppIo/PullPendingTasks": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ServerAppIo/ClaimTask": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ServerAppIo/GetRun": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ServerAppIo/SendAppHeartbeat": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/SendTaskHeartbeat": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PullAppInputs": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PushAppOutputs": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PushObject": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PullObject": MethodTokenPolicy.token_required(),
    # pylint: disable-next=line-too-long
    "/flwr.proto.ServerAppIo/ConfirmMessageReceived": MethodTokenPolicy.token_required(),  # noqa: E501
    "/flwr.proto.ServerAppIo/UpdateRunStatus": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PushLogs": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/GetFederationOptions": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PushMessages": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/PullMessages": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/GetNodes": MethodTokenPolicy.token_required(),
    "/flwr.proto.ServerAppIo/CreateTask": MethodTokenPolicy.token_required(),
}

CLIENTAPPIO_METHOD_AUTH_POLICY: dict[str, MethodTokenPolicy] = {
    "/flwr.proto.ClientAppIo/ListAppsToLaunch": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ClientAppIo/RequestToken": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ClientAppIo/PullPendingTasks": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ClientAppIo/ClaimTask": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ClientAppIo/GetRun": MethodTokenPolicy.no_auth(),
    "/flwr.proto.ClientAppIo/SendAppHeartbeat": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/SendTaskHeartbeat": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/PullAppInputs": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/PushAppOutputs": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/PushObject": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/PullObject": MethodTokenPolicy.token_required(),
    # pylint: disable-next=line-too-long
    "/flwr.proto.ClientAppIo/ConfirmMessageReceived": MethodTokenPolicy.token_required(),  # noqa: E501
    "/flwr.proto.ClientAppIo/PushMessage": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/PullMessage": MethodTokenPolicy.token_required(),
    "/flwr.proto.ClientAppIo/CreateTask": MethodTokenPolicy.token_required(),
}
