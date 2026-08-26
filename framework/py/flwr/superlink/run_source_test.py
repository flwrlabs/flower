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
"""Tests for run-start source attribution utilities."""

from pytest import mark

from .run_source import RunStartSource, resolve_run_start_source


@mark.parametrize(
    ("value", "expected"),
    [
        (None, "unknown"),
        ("web_ui", "web_ui"),
        ("automation", "automation"),
        ("not-a-source", "unknown"),
    ],
)
def test_resolve_run_start_source(
    value: str | None, expected: RunStartSource
) -> None:
    """Normalize best-effort caller attribution without treating it as auth."""
    assert resolve_run_start_source(value) == expected
