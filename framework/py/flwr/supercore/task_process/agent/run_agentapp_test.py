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
"""Tests for the AgentApp process environment."""

import os

import pytest

from .run_agentapp import _set_runtime_environment


@pytest.mark.parametrize(("insecure", "scheme"), [(True, "http"), (False, "https")])
def test_set_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, insecure: bool, scheme: str
) -> None:
    """Expose the Runtime Responses base URL and AgentApp task token."""
    monkeypatch.delenv("FLWR_RUNTIME_BASE_URL", raising=False)
    monkeypatch.delenv("FLWR_RUNTIME_API_KEY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    _set_runtime_environment(
        "runtime.example:9092",
        "task-token",
        insecure=insecure,
        root_certificates_path="/path/to/runtime-ca.pem",
    )

    assert os.environ["FLWR_RUNTIME_BASE_URL"] == (
        f"{scheme}://runtime.example:9092/v1/runtime"
    )
    assert os.environ["FLWR_RUNTIME_API_KEY"] == "task-token"
    assert os.environ["SSL_CERT_FILE"] == "/path/to/runtime-ca.pem"
