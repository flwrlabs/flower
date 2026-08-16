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
"""Test the no-op Control authentication plugin."""

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID

from .noop_auth_plugin import NoOpControlAuthnPlugin


def test_validate_tokens_returns_fresh_account_info() -> None:
    """Return isolated account information for every validation."""
    plugin = NoOpControlAuthnPlugin()

    _, first = plugin.validate_tokens_in_metadata([])
    assert first is not None
    first.account_name = "modified"

    _, second = plugin.validate_tokens_in_metadata([])
    assert second is not None
    assert second is not first
    assert second.flwr_aid == NOOP_FLWR_AID
    assert second.account_name == NOOP_ACCOUNT_NAME


def test_refresh_tokens_returns_fresh_account_info() -> None:
    """Return isolated account information for every token refresh."""
    plugin = NoOpControlAuthnPlugin()

    _, first = plugin.refresh_tokens([])
    assert first is not None
    first.account_name = "modified"

    _, second = plugin.refresh_tokens([])
    assert second is not None
    assert second is not first
    assert second.flwr_aid == NOOP_FLWR_AID
    assert second.account_name == NOOP_ACCOUNT_NAME
