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
"""Tests for the ServerApp runtime."""


from queue import Queue
from unittest.mock import Mock, patch

from flwr.supercore.constant import EXIT_HANDLER_OUTPUT_TIMEOUT_SECONDS

from .run_serverapp import run_serverapp


# pylint: disable=protected-access
def test_exit_stops_background_workers_before_task_output() -> None:
    """Test that shutdown preserves time for final task output."""
    log_queue: Queue[str | None] = Queue()
    grid = Mock()
    grid._stub.PullTaskInput.return_value = Mock()
    heartbeat_sender = Mock(is_running=True)
    log_uploader = Mock()
    run = Mock(run_id=123)
    fab = Mock(content=b"fab")
    call_order = []

    with (
        patch("flwr.superlink.runtime.run_serverapp.GrpcGrid", return_value=grid),
        patch(
            "flwr.superlink.runtime.run_serverapp.HeartbeatSender",
            return_value=heartbeat_sender,
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.context_from_proto",
            return_value=None,
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.run_from_proto",
            return_value=run,
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.fab_from_proto",
            return_value=fab,
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.get_sha256_hash",
            return_value="hash",
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.start_log_uploader",
            return_value=log_uploader,
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.install_from_fab",
            side_effect=RuntimeError("stop after background workers start"),
        ),
        patch(
            "flwr.superlink.runtime.run_serverapp.register_signal_handlers"
        ) as register_signal_handlers,
        patch("flwr.superlink.runtime.run_serverapp.flwr_exit"),
        patch("flwr.superlink.runtime.run_serverapp.flush_logs") as flush_logs,
        patch(
            "flwr.superlink.runtime.run_serverapp.stop_log_uploader"
        ) as stop_log_uploader,
        patch("flwr.superlink.runtime.run_serverapp.cleanup_app_runtime_environment"),
        patch(
            "flwr.superlink.runtime.run_serverapp.time.monotonic",
            side_effect=[10.0, 10.5, 11.5, 12.0],
        ),
    ):
        flush_logs.side_effect = lambda *args, **kwargs: call_order.append("flush")
        stop_log_uploader.side_effect = lambda *args, **kwargs: call_order.append(
            "stop_logs"
        )
        heartbeat_sender.stop.side_effect = lambda *args, **kwargs: call_order.append(
            "stop_heartbeat"
        )
        grid._stub.PushTaskOutput.side_effect = lambda *args, **kwargs: (
            call_order.append("push_output")
        )

        run_serverapp(
            runtime_api_address="127.0.0.1:9091",
            log_queue=log_queue,
            token="token",
            insecure=True,
        )
        exit_handler = register_signal_handlers.call_args.kwargs["exit_handlers"][0]
        exit_handler()

    assert call_order == ["flush", "stop_logs", "stop_heartbeat", "push_output"]
    assert flush_logs.call_args.kwargs["timeout"] == 2.5
    assert stop_log_uploader.call_args.kwargs["timeout"] == 1.5
    assert heartbeat_sender.stop.call_args.kwargs["timeout"] == 1.0
    assert (
        grid._stub.PushTaskOutput.call_args.kwargs["timeout"]
        == EXIT_HANDLER_OUTPUT_TIMEOUT_SECONDS
    )
