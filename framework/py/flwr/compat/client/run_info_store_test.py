# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Run info store tests."""


from typing import cast

from flwr.app import ConfigRecord, Context
from flwr.compat.client.run_info_store import DeprecatedRunInfoStore


def _run_dummy_task(context: Context) -> Context:
    counter_value: str = "1"
    if "counter" in context.state.config_records.keys():
        counter_value = cast(str, context.state.config_records["counter"]["count"])
        counter_value += "1"

    context.state.config_records["counter"] = ConfigRecord({"count": counter_value})

    return context


def test_multirun_in_run_info_store() -> None:
    """Test basic DeprecatedRunInfoStore logic."""
    # Tasks to perform
    run_ids = [0, 1, 1, 2, 3, 2, 1, 5]
    # the "task" is to count how many times each run is executed
    expected_values = {0: "1", 1: "1" * 3, 2: "1" * 2, 3: "1", 5: "1"}

    node_info_store = DeprecatedRunInfoStore(node_id=0, node_config={})

    for run_id in run_ids:

        # Register
        node_info_store.register_context(run_id=run_id)

        # Get run state
        context = node_info_store.retrieve_context(run_id=run_id)

        # Run "task"
        updated_state = _run_dummy_task(context)

        # Update run state
        node_info_store.update_context(run_id=run_id, context=updated_state)

    # Verify values
    for run_id, run_info in node_info_store.run_infos.items():
        assert (
            run_info.context.state.config_records["counter"]["count"]
            == expected_values[run_id]
        )
