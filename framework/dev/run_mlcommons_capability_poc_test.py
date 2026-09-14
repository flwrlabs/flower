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
"""Integration test for the executable MLCommons capability POC harness."""

from dev.run_mlcommons_capability_poc import run_smoke


def test_capability_poc_allowed_and_denied_paths() -> None:
    """Exercise CLI submission, routing, Guardian allow, and fail-closed denial."""
    result = run_smoke()
    scenarios = result["scenarios"]

    assert isinstance(scenarios, dict)
    assert scenarios["allowed"]["task_created"]
    assert scenarios["allowed"]["fab_retrieved"]
    assert not scenarios["denied"]["task_created"]
    assert not scenarios["denied"]["fab_retrieved"]
