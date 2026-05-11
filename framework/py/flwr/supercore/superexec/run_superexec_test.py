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
"""Tests for SuperExec runtime setup."""


from typing import Any
from unittest.mock import Mock

import pytest

from flwr.supercore.interceptors import (
    RuntimeVersionClientInterceptor,
    SuperExecAuthClientInterceptor,
)

from . import run_superexec as run_superexec_module


@pytest.mark.parametrize(
    ("superexec_auth_secret", "expected_interceptor_types"),
    [
        (None, (RuntimeVersionClientInterceptor,)),
        (
            b"superexec-secret",
            (RuntimeVersionClientInterceptor, SuperExecAuthClientInterceptor),
        ),
    ],
)
def test_run_superexec_adds_runtime_version_interceptor(
    monkeypatch: pytest.MonkeyPatch,
    superexec_auth_secret: bytes | None,
    expected_interceptor_types: tuple[type[object], ...],
) -> None:
    """SuperExec should attach runtime version metadata to AppIO calls."""
    channel = Mock()
    stub = Mock()
    stub.PullPendingTasks.side_effect = KeyboardInterrupt()
    captured: dict[str, Any] = {}

    def _create_channel(**kwargs: Any) -> Mock:
        captured.update(kwargs)
        return channel

    monkeypatch.setattr(run_superexec_module, "create_channel", _create_channel)
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(run_superexec_module, "wrap_stub", Mock())

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            stub_class=Mock(return_value=stub),
            appio_api_address="127.0.0.1:9091",
            insecure=True,
            superexec_auth_secret=superexec_auth_secret,
        )

    assert tuple(type(interceptor) for interceptor in captured["interceptors"]) == (
        expected_interceptor_types
    )
