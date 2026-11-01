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
"""Tests for the Control API run-source dependency."""

from .run_source import get_run_source


def test_get_run_source_defaults_to_unknown() -> None:
    """Return unknown when the request has no source header."""
    assert get_run_source() == "unknown"


def test_get_run_source_normalizes_header_value() -> None:
    """Normalize a caller-provided source header."""
    assert get_run_source("web_ui") == "web_ui"
