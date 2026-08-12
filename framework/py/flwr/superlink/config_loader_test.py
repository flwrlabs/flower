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
"""Tests for SuperLink configuration loading."""


import pytest

from flwr.common.constant import AuthnType
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin

from . import config_loader


class FakeOidcControlAuthnPlugin(NoOpControlAuthnPlugin):
    """Test-only OIDC authentication plugin."""


def test_load_control_authn_plugin_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OIDC authentication is disabled when the flag is absent."""
    monkeypatch.delenv("FLWR_OIDC_ENABLED", raising=False)

    assert isinstance(config_loader.load_control_authn_plugin(), NoOpControlAuthnPlugin)


def test_load_control_authn_plugin_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the standard Flower enabled value selects the OIDC plugin."""
    monkeypatch.setenv("FLWR_OIDC_ENABLED", "1")
    monkeypatch.setattr(
        config_loader,
        "get_control_authn_ee_plugins",
        lambda: {AuthnType.OIDC: FakeOidcControlAuthnPlugin},
    )

    assert isinstance(
        config_loader.load_control_authn_plugin(), FakeOidcControlAuthnPlugin
    )
