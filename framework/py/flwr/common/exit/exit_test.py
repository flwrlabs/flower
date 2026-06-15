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
"""Tests for the exit function."""


from unittest.mock import patch

from parameterized import parameterized

from .exit import _get_code_url


@parameterized.expand(  # type: ignore
    [
        ("1.2.3", "1.2/en/"),
        ("0.9.0", "0.9/en/"),
        ("2.0.0-beta", "2.0/en/"),
        ("3.4.5.dev0", "3.4/en/"),
        ("non-standard-version", ""),  # Fallback case
    ]
)
def test_get_code_url(version: str, subdir: str) -> None:
    """Test that the correct URL is generated for a given exit code."""
    with patch("flwr.common.exit.exit.package_version", version):
        for code in [0, 42, 999]:
            actual_url = _get_code_url(code)
            expected_url = "https://flower.ai/docs/framework/"
            expected_url += f"{subdir}ref-exit-codes/{code}.html"
            assert actual_url == expected_url
