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
"""Tests for the shared HTTP user-authentication contract."""

import pytest

from flwr.supercore.error import ApiErrorCode, FlowerError

from .typing import AccountInfo
from .user_auth import get_bearer_token


def test_account_info_backward_compatible_defaults() -> None:
    """Keep existing two-field construction compatible."""
    account = AccountInfo("flower-aid", "alice")

    assert account.email is None
    assert account.roles == frozenset()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer access-token", "access-token"),
        ("bearer access-token", "access-token"),
    ],
)
def test_get_bearer_token(header: str, expected: str) -> None:
    """Accept a single case-insensitive Bearer scheme and token."""
    assert get_bearer_token(header) == expected


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Basic credentials",
        "Bearer",
        "Bearer ",
        "Bearer token with-spaces",
        "Bearer  token",
        "Bearer token ",
    ],
)
def test_get_bearer_token_rejects_missing_or_malformed_header(
    header: str | None,
) -> None:
    """Classify every malformed authorization header as authentication failure."""
    with pytest.raises(FlowerError) as exc_info:
        get_bearer_token(header)

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
