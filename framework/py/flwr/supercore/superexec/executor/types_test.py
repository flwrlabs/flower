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
"""Tests for SuperExec executor protocol types."""


from inspect import signature

from .types import Executor


def test_executor_protocol_defines_profileless_capacity_wait() -> None:
    """Test Executor exposes a profileless capacity wait hook."""
    wait_signature = signature(Executor.wait_for_capacity)

    assert list(wait_signature.parameters) == ["self"]
    assert wait_signature.return_annotation is None
