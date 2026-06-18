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

from pathlib import Path

import yaml

from flwr.supercore.constant import ExecutorType

ExecutorConfig = dict[str, object]


class ExecutorConfigError(ValueError):
    """Raised when executor config loading fails."""


def load_executor_config(path: str, executor_type: ExecutorType) -> ExecutorConfig:
    """Load executor config for the selected executor."""
    message_prefix = f"Failed to load executor config from '{path}': "
    if executor_type == ExecutorType.SUBPROCESS:
        raise ExecutorConfigError(
            f"{message_prefix}subprocess executor does not support --executor-config."
        )

    try:
        with Path(path).expanduser().open(encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ExecutorConfigError(f"{message_prefix}file does not exist.") from exc
    except OSError as exc:
        message = exc.strerror or str(exc)
        raise ExecutorConfigError(
            f"{message_prefix}file could not be read: {message}."
        ) from exc
    except yaml.YAMLError as exc:
        raise ExecutorConfigError(
            f"{message_prefix}file must contain valid YAML."
        ) from exc

    if raw_config is None:
        raise ExecutorConfigError(f"{message_prefix}file must not be empty.")
    if not isinstance(raw_config, dict):
        raise ExecutorConfigError(f"{message_prefix}root must be a mapping.")

    return dict(raw_config)
