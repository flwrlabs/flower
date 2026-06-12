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
"""Executor config loading for SuperExec."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from flwr.supercore.constant import ExecutorType

ExecutorConfig = dict[str, object]


class ExecutorConfigError(ValueError):
    """Raised when executor config loading fails."""

    def __init__(self, path: str | None, errors: Sequence[str]) -> None:
        message = "Failed to load executor config"
        if path is not None:
            message += f" from '{path}'"
        bullets = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"{message}:\n{bullets}")


def load_executor_config(path: str, executor_type: ExecutorType) -> ExecutorConfig:
    """Load executor config for the selected executor."""
    if executor_type == ExecutorType.SUBPROCESS:
        raise ExecutorConfigError(
            path,
            ["subprocess executor does not support --executor-config."],
        )
    raw_config = _load_yaml(path)
    if raw_config is None:
        raise ExecutorConfigError(path, ["file must not be empty."])
    if not isinstance(raw_config, Mapping):
        raise ExecutorConfigError(path, ["root must be a mapping."])

    return dict(raw_config)


def _load_yaml(path: str) -> object:
    try:
        with open(Path(path).expanduser(), encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ExecutorConfigError(path, ["file does not exist."]) from exc
    except OSError as exc:
        message = exc.strerror or str(exc)
        raise ExecutorConfigError(
            path, [f"file could not be read: {message}."]
        ) from exc
    except yaml.YAMLError as exc:
        raise ExecutorConfigError(path, [_yaml_error_message(exc)]) from exc


def _yaml_error_message(exc: yaml.YAMLError) -> str:
    problem = getattr(exc, "problem", None)
    mark = getattr(exc, "problem_mark", None)
    if problem is not None and mark is not None:
        return (
            "file must contain valid YAML near "
            f"line {mark.line + 1}, column {mark.column + 1}: {problem}."
        )
    return "file must contain valid YAML."
