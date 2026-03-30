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
"""Tests for API error catalog consistency."""


from .base import ApiErrorCode
from .catalog import API_ERROR_MAP


def test_api_error_map_covers_all_api_error_codes() -> None:
    """Ensure every ApiErrorCode has an API_ERROR_MAP entry."""
    assert set(API_ERROR_MAP.keys()) == set(ApiErrorCode)
