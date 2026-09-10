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

from concurrent.futures import Future
from logging import ERROR
from threading import Thread
from typing import TYPE_CHECKING, cast

from flwr.supercore import log
from flwr.supercore.constant import RUN_SERIES_DESCRIPTION_MAX_LENGTH
from flwr.supercore.typing import JSONObject

if TYPE_CHECKING:
    from .session import RuntimeAgentResponses

_TITLE_DEFAULT = "New conversation"
_TITLE_MODEL = "openai/gpt-5-nano"
_TITLE_INSTRUCTIONS = (
    "Create a concise title for this conversation. "
    "Return only the title, without quotes or Markdown, using at most four words."
)


def generate_series_description_in_background(
    responses: RuntimeAgentResponses, prompt: str
) -> Future[str]:
    """Generate a RunSeries description in a daemon thread."""
    future: Future[str] = Future()

    def generate() -> None:
        try:
            future.set_result(generate_series_description(responses, prompt))
        except Exception as ex:  # pylint: disable=broad-exception-caught
            future.set_exception(ex)

    Thread(
        target=generate,
        name="flwr-conversation-title",
        daemon=True,
    ).start()
    return future


def resolve_series_description(future: Future[str] | None) -> str | None:
    """Return the generated description if it is ready."""
    if future is None or not future.done():
        return None
    try:
        return future.result()
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to resolve RunSeries description: %s", ex)
        return None


def generate_series_description(responses: RuntimeAgentResponses, prompt: str) -> str:
    """Generate a title, falling back to a prompt excerpt on failure."""
    fallback = " ".join(prompt.split()[:4]) or _TITLE_DEFAULT
    title = ""
    try:
        response = responses.create(
            {
                "model": _TITLE_MODEL,
                "instructions": _TITLE_INSTRUCTIONS,
                "input": prompt,
                "stream": False,
                "max_output_tokens": 32,
                "reasoning": {"effort": "minimal"},
            }
        )
        output = cast(list[JSONObject], response["output"])
        content = cast(list[JSONObject], output[-1]["content"])
        title = cast(str, content[0]["text"])
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to generate RunSeries description: %s", ex)

    title = " ".join(title.strip().strip('"').strip("'").split()) or fallback
    if len(title) > RUN_SERIES_DESCRIPTION_MAX_LENGTH:
        title = f"{title[: RUN_SERIES_DESCRIPTION_MAX_LENGTH - 1].rstrip()}…"
    return title
