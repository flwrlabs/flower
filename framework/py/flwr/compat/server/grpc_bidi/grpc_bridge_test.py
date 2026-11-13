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
"""Tests for GrpcBridge class."""


import time
from threading import Thread

from flwr.compat.server.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    GrpcBridgeClosed,
    InsWrapper,
    ResWrapper,
)
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)


def start_worker(
    rounds: int, bridge: GrpcBridge, results: list[ClientMessage]
) -> Thread:
    """Simulate processing loop with five calls."""

    def _worker() -> None:
        # Wait until the ClientMessage is available and extract
        # although here we do nothing with the return value
        for _ in range(rounds):
            try:
                res_wrapper = bridge.request(
                    InsWrapper(server_message=ServerMessage(), timeout=None)
                )
            except GrpcBridgeClosed:
                break

            results.append(res_wrapper.client_message)

    thread = Thread(target=_worker)
    thread.start()

    return thread


def test_workflow_successful() -> None:
    """Test full workflow."""
    # Prepare
    rounds = 5
    client_messages_received: list[ClientMessage] = []

    bridge = GrpcBridge()
    ins_wrapper_iterator = bridge.ins_wrapper_iterator()

    worker_thread = start_worker(rounds, bridge, client_messages_received)

    # Execute
    # Simulate remote client side
    for _ in range(rounds):
        try:
            _ = next(ins_wrapper_iterator)
            bridge.set_res_wrapper(ResWrapper(client_message=ClientMessage()))
        except Exception as exception:
            # pylint: disable-next=broad-exception-raised
            raise Exception from exception

    # Wait until worker_thread is finished
    worker_thread.join(timeout=1)

    # Assert
    assert len(client_messages_received) == rounds


def test_workflow_close() -> None:
    """Test interrupted workflow.

    Close bridge after setting three client messages.
    """
    # Prepare
    rounds = 5
    client_messages_received: list[ClientMessage] = []

    bridge = GrpcBridge()
    ins_wrapper_iterator = bridge.ins_wrapper_iterator()

    worker_thread = start_worker(rounds, bridge, client_messages_received)

    raised_error: GrpcBridgeClosed | StopIteration | None = None

    # Execute
    for i in range(rounds):
        try:
            _ = next(ins_wrapper_iterator)
            bridge.set_res_wrapper(ResWrapper(client_message=ClientMessage()))

            # Close the bridge after the third client message is set.
            # This might interrupt consumption of the message.
            if i == 2:
                # As the bridge is closed while ins_wrapper_iterator is not
                # waiting/blocking for next message it should raise StopIteration
                # on next invocation.
                bridge.close()

        except GrpcBridgeClosed as err:
            raised_error = err
            break
        except StopIteration as err:
            raised_error = err
            break

    # Wait for thread join before finishing the test
    worker_thread.join(timeout=1)

    # Assert
    assert len(client_messages_received) == 2
    assert isinstance(raised_error, StopIteration)


def test_ins_wrapper_iterator_close_while_blocking() -> None:
    """Test interrupted workflow.

    Close bridge while blocking for next ins_wrapper.
    """
    # Prepare
    rounds = 5
    client_messages_received: list[ClientMessage] = []

    bridge = GrpcBridge()
    ins_wrapper_iterator = bridge.ins_wrapper_iterator()

    worker_thread = start_worker(rounds, bridge, client_messages_received)

    raised_error: GrpcBridgeClosed | StopIteration | None = None

    def close_bridge_delayed(secs: int) -> None:
        """Close brige after {secs} second(s)."""
        time.sleep(secs)
        bridge.close()

    # Execute
    for i in range(rounds):
        try:
            # Close the bridge while the iterator is waiting/blocking
            # for a server message
            if i == 3:
                Thread(target=close_bridge_delayed, args=(1,)).start()

            _ = next(ins_wrapper_iterator)

            # Do not set a client message and wait until
            # the thread above closes the bridge
            if i < 2:
                bridge.set_res_wrapper(ResWrapper(ClientMessage()))

        except GrpcBridgeClosed as err:
            raised_error = err
            break
        except StopIteration as err:
            raised_error = err
            break

    # Wait for thread join before finishing the test
    worker_thread.join(timeout=1)

    # Assert
    assert len(client_messages_received) == 2
    assert isinstance(raised_error, GrpcBridgeClosed)
