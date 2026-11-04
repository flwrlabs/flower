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
"""Provides class GrpcBridge."""


from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from threading import Condition

from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)


@dataclass
class InsWrapper:
    """Instruction wrapper class for a single server message."""

    server_message: ServerMessage
    timeout: float | None


@dataclass
class ResWrapper:
    """Result wrapper class for a single client message."""

    client_message: ClientMessage


class GrpcBridgeClosed(Exception):
    """Error signaling that GrpcBridge is closed."""


class Status(Enum):
    """Status through which the bridge can transition."""

    AWAITING_INS_WRAPPER = 1
    INS_WRAPPER_AVAILABLE = 2
    AWAITING_RES_WRAPPER = 3
    RES_WRAPPER_AVAILABLE = 4
    CLOSED = 5


class GrpcBridge:
    """GrpcBridge holding res_wrapper and ins_wrapper.

    For understanding this class it is recommended to understand how
    the threading.Condition class works. See here:
    - https://docs.python.org/3/library/threading.html#condition-objects
    """

    def __init__(self) -> None:
        """Init bridge."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self._cv = Condition()  # cv stands for condition variable
        self._status = Status.AWAITING_INS_WRAPPER
        self._ins_wrapper: InsWrapper | None = None
        self._res_wrapper: ResWrapper | None = None

    def _is_closed(self) -> bool:
        """Return True if closed and False otherwise."""
        return self._status == Status.CLOSED

    def _raise_if_closed(self) -> None:
        if self._status == Status.CLOSED:
            raise GrpcBridgeClosed()

    def _transition(self, next_status: Status) -> None:
        """Validate status transition and set next status.

        The caller of the transition method will have to aquire conditional variable.
        """
        if next_status == Status.CLOSED:
            self._status = next_status
        elif (
            self._status == Status.AWAITING_INS_WRAPPER
            and next_status == Status.INS_WRAPPER_AVAILABLE
            and self._ins_wrapper is not None
            and self._res_wrapper is None
        ):
            self._status = next_status
        elif (
            self._status == Status.INS_WRAPPER_AVAILABLE
            and next_status == Status.AWAITING_RES_WRAPPER
            and self._ins_wrapper is None
            and self._res_wrapper is None
        ):
            self._status = next_status
        elif (
            self._status == Status.AWAITING_RES_WRAPPER
            and next_status == Status.RES_WRAPPER_AVAILABLE
            and self._ins_wrapper is None
            and self._res_wrapper is not None
        ):
            self._status = next_status
        elif (
            self._status == Status.RES_WRAPPER_AVAILABLE
            and next_status == Status.AWAITING_INS_WRAPPER
            and self._ins_wrapper is None
            and self._res_wrapper is None
        ):
            self._status = next_status
        else:
            raise ValueError(f"Invalid transition: {self._status} to {next_status}")

        self._cv.notify_all()

    def close(self) -> None:
        """Set bridge status to closed."""
        with self._cv:
            self._transition(Status.CLOSED)

    def request(self, ins_wrapper: InsWrapper) -> ResWrapper:
        """Set ins_wrapper and wait for res_wrapper."""
        # Set ins_wrapper and transition to INS_WRAPPER_AVAILABLE
        with self._cv:
            self._raise_if_closed()

            if self._status != Status.AWAITING_INS_WRAPPER:
                raise ValueError("This should not happen")

            self._ins_wrapper = ins_wrapper  # Write
            self._transition(Status.INS_WRAPPER_AVAILABLE)

        # Read res_wrapper and transition to AWAITING_INS_WRAPPER
        with self._cv:
            self._cv.wait_for(
                lambda: self._status in [Status.CLOSED, Status.RES_WRAPPER_AVAILABLE]
            )

            self._raise_if_closed()
            res_wrapper = self._res_wrapper  # Read
            self._res_wrapper = None  # Reset
            self._transition(Status.AWAITING_INS_WRAPPER)

        if res_wrapper is None:
            raise ValueError("ResWrapper can not be None")

        return res_wrapper

    def ins_wrapper_iterator(self) -> Iterator[InsWrapper]:
        """Return iterator over ins_wrapper objects."""
        while not self._is_closed():
            with self._cv:
                self._cv.wait_for(
                    lambda: self._status
                    in [Status.CLOSED, Status.INS_WRAPPER_AVAILABLE]
                )

                self._raise_if_closed()

                ins_wrapper = self._ins_wrapper  # Read
                self._ins_wrapper = None  # Reset

                # Transition before yielding as after the yield the execution of this
                # function is paused and will resume when next is called again.
                # Also release condition variable by exiting the context
                self._transition(Status.AWAITING_RES_WRAPPER)

            if ins_wrapper is None:
                raise ValueError("InsWrapper can not be None")

            yield ins_wrapper

    def set_res_wrapper(self, res_wrapper: ResWrapper) -> None:
        """Set res_wrapper for consumption."""
        with self._cv:
            self._raise_if_closed()

            if self._status != Status.AWAITING_RES_WRAPPER:
                raise ValueError("This should not happen")

            self._res_wrapper = res_wrapper  # Write
            self._transition(Status.RES_WRAPPER_AVAILABLE)
