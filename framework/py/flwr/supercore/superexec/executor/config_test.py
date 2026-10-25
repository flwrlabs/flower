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
"""Tests for SuperExec executor config parsing."""

from pathlib import Path
from textwrap import dedent

import pytest

from flwr.supercore.constant import ExecutorType

from .config import ExecutorConfigError, load_executor_config


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_load_executor_config_rejects_subprocess_config(tmp_path: Path) -> None:
    """Test subprocess executor config is rejected rather than ignored."""
    config_path = tmp_path / "executor.yaml"
    _write_yaml(config_path, "kubernetes: {}")

    with pytest.raises(ExecutorConfigError) as exc_info:
        load_executor_config(str(config_path), ExecutorType.SUBPROCESS)

    assert "subprocess executor does not support --executor-config" in str(
        exc_info.value
    )


def test_load_executor_config_rejects_empty_yaml(tmp_path: Path) -> None:
    """Test empty YAML files are invalid."""
    config_path = tmp_path / "executor.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ExecutorConfigError, match="file must not be empty"):
        load_executor_config(str(config_path), ExecutorType.KUBERNETES)
