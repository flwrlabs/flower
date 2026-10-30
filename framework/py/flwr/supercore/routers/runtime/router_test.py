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
"""Tests for shared Runtime file routes."""

from collections.abc import Callable
from types import ModuleType
from unittest.mock import Mock

import pytest

from flwr.proto import runtime_pb2
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.servicer.runtime import runtime_handlers as link_handlers
from flwr.supernode.servicer.runtime import runtime_handlers as node_handlers

from .router import delete_file, get_file, list_files, put_file


@pytest.mark.parametrize("handlers", [link_handlers, node_handlers])
@pytest.mark.parametrize("missing_run", [False, True])
@pytest.mark.parametrize(
    ("route", "request_name"),
    [
        (put_file, "PutFileRequest"),
        (get_file, "GetFileRequest"),
        (list_files, "ListFilesRequest"),
        (delete_file, "DeleteFileRequest"),
    ],
)
def test_file_routes_use_authenticated_series(
    handlers: ModuleType,
    missing_run: bool,
    route: Callable[..., object],
    request_name: str,
) -> None:
    """Resolve series scope through each host and reject unknown runs."""
    state = Mock()
    state.get_run_info.return_value = [] if missing_run else [Mock(series_id=456)]
    state.get_run.return_value = None if missing_run else Mock(series_id=456)
    state.put_file.return_value = 1
    state.get_file.return_value = None
    state.list_files.return_value = []
    request = getattr(runtime_pb2, request_name)()
    task = Task(task_id=123, run_id=789)
    operation = route.__name__

    if missing_run:
        with pytest.raises(FlowerError) as caught:
            route(request, state, handlers, task)
        assert caught.value.code == ApiErrorCode.RUN_ID_NOT_FOUND
        getattr(state, operation).assert_not_called()
    else:
        route(request, state, handlers, task)
        assert getattr(state, operation).call_args.kwargs["scope"] == "run-series-456"

    if handlers is link_handlers:
        state.get_run_info.assert_called_once_with(run_ids=[789])
    else:
        state.get_run.assert_called_once_with(789)
