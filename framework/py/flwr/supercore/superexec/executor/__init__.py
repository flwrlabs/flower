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
"""Executor abstractions and implementations for SuperExec."""

from .factory import get_executor
from .kubernetes_executor import KubernetesExecutor, KubernetesExecutorConfig
from .subprocess_executor import SubprocessExecutor
from .types import ExecutionSpec, Executor, LaunchResult, LaunchResultStatus

__all__ = [
    "ExecutionSpec",
    "Executor",
    "KubernetesExecutor",
    "KubernetesExecutorConfig",
    "LaunchResult",
    "LaunchResultStatus",
    "SubprocessExecutor",
    "get_executor",
]
