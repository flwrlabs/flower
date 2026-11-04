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
"""Flower ClientProxy implementation using gRPC bidirectional streaming."""


from flwr import common
from flwr.common import serde
from flwr.compat.server.grpc_bidi.grpc_bridge import GrpcBridge, InsWrapper, ResWrapper
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.server.client_proxy import ClientProxy


class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetPropertiesRes:
        """Request client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_properties_ins=get_properties_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parameters_ins=get_parameters_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parameters_res = serde.get_parameters_res_from_proto(
            client_msg.get_parameters_res
        )
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(evaluate_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(reconnect_ins=reconnect_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect
