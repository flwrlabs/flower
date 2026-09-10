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
"""Generate short RunSeries descriptions from the initial user prompt."""

from __future__ import annotations

import os
from concurrent.futures import Future
from logging import ERROR
from threading import Thread
from typing import cast

import httpx

from flwr.supercore import log
from flwr.supercore.constant import RUN_SERIES_DESCRIPTION_MAX_LENGTH
from flwr.supercore.typing import JSONObject

_TITLE_DEFAULT = "New conversation"
_TITLE_MODEL = "openai/gpt-5-nano"
_TITLE_TIMEOUT = 300.0
_TITLE_INSTRUCTIONS = (
    "Create a concise title for this conversation. "
    "Return only the title, without quotes or Markdown, using at most four words."
)


def generate_series_description_in_background(prompt: str) -> Future[str]:
    """Generate a RunSeries description in a daemon thread."""
    future: Future[str] = Future()

    def generate() -> None:
        future.set_result(generate_series_description(prompt))

    Thread(
        target=generate,
        name="flwr-conversation-title",
        daemon=True,
    ).start()
    return future


def generate_series_description(prompt: str) -> str:
    """Generate a title, falling back to a prompt excerpt on failure."""
    fallback = " ".join(prompt.split()[:4]) or _TITLE_DEFAULT
    title = ""
    try:
        response = httpx.post(
            f"{os.environ['FLWR_RUNTIME_BASE_URL'].rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {os.environ['FLWR_RUNTIME_API_KEY']}"},
            json={
                "model": _TITLE_MODEL,
                "instructions": _TITLE_INSTRUCTIONS,
                "input": prompt,
                "stream": False,
                "max_output_tokens": 32,
                "reasoning": {"effort": "minimal"},
            },
            timeout=_TITLE_TIMEOUT,
        )
        response.raise_for_status()
        output = cast(list[JSONObject], response.json()["output"])
        content = cast(list[JSONObject], output[-1]["content"])
        title = cast(str, content[0]["text"])
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to generate RunSeries description: %s", ex)

    title = " ".join(title.strip().strip('"').strip("'").split()) or fallback
    if len(title) > RUN_SERIES_DESCRIPTION_MAX_LENGTH:
        title = f"{title[: RUN_SERIES_DESCRIPTION_MAX_LENGTH - 1].rstrip()}…"
    return title
