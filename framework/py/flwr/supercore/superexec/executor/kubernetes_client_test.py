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
"""Tests for SuperExec Kubernetes client construction."""


import importlib
from unittest.mock import Mock

import pytest

from .kubernetes_executor import create_incluster_kubernetes_client


def test_create_incluster_kubernetes_client_loads_config_before_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test in-cluster auth loads before CoreV1Api construction."""
    core_v1_api = Mock()
    calls: list[str] = []
    client_module = Mock()
    config_module = Mock()

    def core_v1_api_factory() -> object:
        calls.append("CoreV1Api")
        return core_v1_api

    def load_incluster_config() -> None:
        calls.append("load_incluster_config")

    client_module.CoreV1Api.side_effect = core_v1_api_factory
    config_module.load_incluster_config.side_effect = load_incluster_config
    modules = {
        "kubernetes.client": client_module,
        "kubernetes.config": config_module,
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)

    client = create_incluster_kubernetes_client()

    assert client is core_v1_api
    assert calls == ["load_incluster_config", "CoreV1Api"]


def test_create_incluster_kubernetes_client_fails_if_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test missing Kubernetes package raises clear construction error."""

    def import_module(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'kubernetes'", name="kubernetes")

    monkeypatch.setattr(importlib, "import_module", import_module)

    with pytest.raises(
        RuntimeError,
        match="Kubernetes Python client package is required",
    ):
        create_incluster_kubernetes_client()


def test_create_incluster_kubernetes_client_fails_if_auth_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test unavailable in-cluster auth raises a clear construction error."""
    client_module = Mock()
    config_module = Mock()
    config_module.load_incluster_config.side_effect = RuntimeError("missing host")
    modules = {
        "kubernetes.client": client_module,
        "kubernetes.config": config_module,
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)

    with pytest.raises(
        RuntimeError,
        match="Failed to load in-cluster Kubernetes configuration",
    ):
        create_incluster_kubernetes_client()

    client_module.CoreV1Api.assert_not_called()
