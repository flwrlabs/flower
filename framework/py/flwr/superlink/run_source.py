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
"""Run-start source attribution utilities."""

from typing import Literal, cast, get_args

RunStartSource = Literal["cli", "web_ui", "automation", "unknown"]
_RUN_START_SOURCES = frozenset(get_args(RunStartSource))


def resolve_run_start_source(value: str | None) -> RunStartSource:
    """Normalize a caller-provided source label for analytics.

    Source attribution is intentionally best effort. Callers can only affect
    the analytics label for their own request, so recognized values are
    trusted and invalid values fall back to ``unknown``. This value is not a
    security or authorization signal.
    """
    if value is None:
        return "unknown"
    if value not in _RUN_START_SOURCES:
        return "unknown"
    return cast(RunStartSource, value)
