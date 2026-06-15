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
"""Executor factory for SuperExec TaskExecutor processes."""

from flwr.supercore.constant import ExecutorType

from .subprocess_executor import SubprocessExecutor
from .types import Executor


def get_executor(executor_type: ExecutorType) -> Executor:
    """Return the executor for the configured executor type."""
    if executor_type == ExecutorType.SUBPROCESS:
        return SubprocessExecutor()

    raise ValueError(f"Unsupported executor: {executor_type}")
