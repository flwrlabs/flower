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
"""Stable error details shared by Runtime API implementations."""


RUN_ID_MISMATCH = "Run ID does not match the authenticated task."


def malformed_request(method: str, reason: str) -> str:
    """Return stable details for a malformed request."""
    return f"Malformed {method} request: {reason.rstrip('.')}."


def task_transition_failed(task_id: int) -> str:
    """Return stable details for a failed task-completion transition."""
    return f"Task {task_id} cannot transition to FINISHED."
