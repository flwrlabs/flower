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
"""Environment helpers for SuperExec-launched task processes."""


import os

from flwr.supercore.constant import TaskType

FAB_BACKED_TASK_TYPES = frozenset(
    {
        TaskType.AGENT_APP,
        TaskType.CLIENT_APP,
        TaskType.SERVER_APP,
        TaskType.SIMULATION,
    }
)
SENSITIVE_PROVIDER_ENV_VARS = frozenset(
    {
        "BRAVE_API_KEY",
        "EXA_API_KEY",
        "FLWR_MODEL_API_KEY",
        "TAVILY_API_KEY",
    }
)


def task_process_env(task_type: TaskType) -> dict[str, str]:
    """Return the environment for a task process of the given type."""
    env = os.environ.copy()

    if task_type in FAB_BACKED_TASK_TYPES:
        for env_var in SENSITIVE_PROVIDER_ENV_VARS:
            env.pop(env_var, None)

    return env
