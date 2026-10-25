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

import os
from logging import ERROR
from threading import Thread
from typing import cast

import requests

from flwr.server.superlink.linkstate import LinkState
from flwr.supercore import log
from flwr.supercore.constant import RUN_SERIES_DESCRIPTION_MAX_LENGTH
from flwr.supercore.task_process.model.provider import DEFAULT_MODEL_API_ENDPOINT
from flwr.supercore.typing import JSONObject

_MODEL = "openai/gpt-5-nano"
_MAX_PROMPT_LENGTH = 4096
_TIMEOUT = 60.0
_INSTRUCTIONS = (
    "Create a concise title for this conversation. "
    "Return only the title, without quotes or Markdown, using at most four words."
)


def start_title_generation(
    state: LinkState,
    series_id: int,
    prompt: str,
) -> None:
    """Generate and persist a title in a daemon thread."""
    try:
        Thread(
            target=_generate_and_persist_title,
            args=(state, series_id, prompt),
            name="flwr-conversation-title",
            daemon=True,
        ).start()
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to start RunSeries title generation: %s", ex)


def _generate_and_persist_title(
    state: LinkState,
    series_id: int,
    prompt: str,
) -> None:
    """Call the model provider and persist its title."""
    try:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
        endpoint = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip()
        if not endpoint:
            if not api_key:
                raise RuntimeError("Model API key is not set (FLWR_MODEL_API_KEY).")
            endpoint = DEFAULT_MODEL_API_ENDPOINT
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": _MODEL,
                "instructions": _INSTRUCTIONS,
                "input": prompt[:_MAX_PROMPT_LENGTH],
                "stream": False,
                "max_output_tokens": 32,
                "reasoning": {"effort": "minimal"},
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        output = cast(list[JSONObject], response.json()["output"])
        content = cast(list[JSONObject], output[-1]["content"])
        title = cast(str, content[0]["text"])
        title = " ".join(title.strip().strip('"').strip("'").split())
        if len(title) > RUN_SERIES_DESCRIPTION_MAX_LENGTH:
            title = f"{title[: RUN_SERIES_DESCRIPTION_MAX_LENGTH - 1].rstrip()}…"
        state.set_run_series_description(series_id, title)
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to generate RunSeries title: %s", ex)
