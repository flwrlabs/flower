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
"""Legacy Context tests."""

from flwr.app import Context, RecordDict

from .legacy_context import LegacyContext


def test_init_from_context_ignores_private_state() -> None:
    """LegacyContext should copy only public Context fields."""
    state = RecordDict()
    context = Context(
        run_id=1,
        node_id=2,
        node_config={"partition-id": 3},
        state=state,
        run_config={"rounds": 4},
        series_id=5,
    )

    legacy_context = LegacyContext(context)

    assert legacy_context.run_id == 1
    assert legacy_context.node_id == 2
    assert legacy_context.node_config == {"partition-id": 3}
    assert legacy_context.state is state
    assert legacy_context.run_config == {"rounds": 4}
    assert legacy_context.series_id == 5
