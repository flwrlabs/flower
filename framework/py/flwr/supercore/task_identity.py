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
"""Identity of the task running in the current process."""


class _TaskIdentityMeta(type):
    """Metaclass providing strict class-level identity properties."""

    @property
    def task_id(cls) -> int:
        """Return the current task ID."""
        if cls._task_id is None:
            raise RuntimeError("TaskIdentity.task_id is not set.")
        return cls._task_id

    @task_id.setter
    def task_id(cls, value: int) -> None:
        cls._task_id = value

    @property
    def run_id(cls) -> int:
        """Return the current run ID."""
        if cls._run_id is None:
            raise RuntimeError("TaskIdentity.run_id is not set.")
        return cls._run_id

    @run_id.setter
    def run_id(cls, value: int) -> None:
        cls._run_id = value

    @property
    def node_id(cls) -> int:
        """Return the current node ID."""
        if cls._node_id is None:
            raise RuntimeError("TaskIdentity.node_id is not set.")
        return cls._node_id

    @node_id.setter
    def node_id(cls, value: int) -> None:
        cls._node_id = value


class TaskIdentity(metaclass=_TaskIdentityMeta):
    """Identity of the task running in the current process."""

    _task_id: int | None = None
    _run_id: int | None = None
    _node_id: int | None = None
