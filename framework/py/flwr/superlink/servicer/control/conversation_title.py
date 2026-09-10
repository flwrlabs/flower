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
"""Generate a RunSeries title through a model task."""

from __future__ import annotations

import time
from logging import ERROR
from threading import Thread
from typing import cast

from flwr.common.constant import Status, SubStatus
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore import log
from flwr.supercore.constant import RUN_SERIES_DESCRIPTION_MAX_LENGTH, TaskType
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

_MODEL = "openai/gpt-5-nano"
_POLL_INTERVAL = 0.25
_TIMEOUT = 60.0
_INSTRUCTIONS = (
    "Create a concise title for this conversation. "
    "Return only the title, without quotes or Markdown, using at most four words."
)


def start_title_generation(
    state: LinkState,
    run_id: int,
    series_id: int,
    prompt: str,
) -> None:
    """Create a model task and persist its result in a daemon thread."""
    model_task_id = state.create_task(
        TaskType.MODEL,
        run_id,
        model_ref=_MODEL,
    )
    if model_task_id is None:
        log(ERROR, "Failed to create RunSeries title task for run %d", run_id)
        return

    request = ModelRequest(
        dst_task_id=model_task_id,
        input_=prompt,
        model=_MODEL,
        instructions=_INSTRUCTIONS,
        max_output_tokens=32,
        reasoning={"effort": "minimal"},
    )
    request.metadata.__dict__["_run_id"] = run_id
    request.metadata.src_task_id = model_task_id
    request.metadata.__dict__["_message_id"] = request.object_id
    if not state.store_task_message(request):
        state.finish_task(model_task_id, SubStatus.STOPPED, "Title request failed.")
        log(ERROR, "Failed to store RunSeries title request for run %d", run_id)
        return

    thread = Thread(
        target=_persist_title,
        args=(state, model_task_id, series_id),
        name="flwr-conversation-title",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:
        state.finish_task(model_task_id, SubStatus.STOPPED, "Title worker failed.")
        raise


def _persist_title(
    state: LinkState,
    model_task_id: int,
    series_id: int,
) -> None:
    """Wait briefly for a model response and persist its title."""
    deadline = time.monotonic() + _TIMEOUT
    try:
        while time.monotonic() < deadline:
            tasks = state.get_tasks(task_ids=[model_task_id])
            if not tasks:
                raise RuntimeError("Title model task ended without a response.")
            if tasks[0].status.status == Status.FINISHED:
                messages = state.get_task_message(
                    dst_task_ids=[model_task_id],
                    src_task_ids=[model_task_id],
                    limit=1,
                    order_by="created_at",
                )
                if not messages:
                    raise RuntimeError("Title model task ended without a response.")
                response = ModelResponse.from_message(messages[0]).payload
                output = cast(list[JSONObject], response["output"])
                content = cast(list[JSONObject], output[-1]["content"])
                title = cast(str, content[0]["text"])
                title = " ".join(title.strip().strip('"').strip("'").split())
                if len(title) > RUN_SERIES_DESCRIPTION_MAX_LENGTH:
                    title = (
                        f"{title[: RUN_SERIES_DESCRIPTION_MAX_LENGTH - 1].rstrip()}…"
                    )
                state.set_run_series_description(series_id, title)
                return
            time.sleep(_POLL_INTERVAL)

        state.finish_task(model_task_id, SubStatus.STOPPED, "Title request timed out.")
        raise TimeoutError("Title model task timed out.")
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to generate RunSeries title: %s", ex)
