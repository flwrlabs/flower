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
"""SuperLink utilities."""


import grpc

from flwr.common.constant import Status, SubStatus
from flwr.common.typing import RunStatus
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.object_store import ObjectStore

_STATUS_TO_MSG = {
    Status.PENDING: "Run is pending.",
    Status.STARTING: "Run is starting.",
    Status.RUNNING: "Run is running.",
    Status.FINISHED: "Run is finished.",
}


def check_abort(
    run_id: int,
    abort_status_list: list[str],
    state: LinkState,
    store: ObjectStore | None = None,
) -> str | None:
    """Check if the status of the provided `run_id` is in `abort_status_list`."""
    run_status: RunStatus = state.get_run_status({run_id})[run_id]

    if run_status.status in abort_status_list:
        msg = _STATUS_TO_MSG[run_status.status]
        if run_status.sub_status == SubStatus.STOPPED:
            msg += " Stopped by user."
        return msg

    # Clear the objects of the run from the store if the run is finished
    if store and run_status.status == Status.FINISHED:
        store.delete_objects_in_run(run_id)

    return None


def abort_grpc_context(msg: str | None, context: grpc.ServicerContext) -> None:
    """Abort context with statuscode PERMISSION_DENIED if `msg` is not None."""
    if msg is not None:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, msg)
