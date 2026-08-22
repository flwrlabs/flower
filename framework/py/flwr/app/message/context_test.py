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
"""Context tests."""

from concurrent.futures import ThreadPoolExecutor
from time import sleep

from flwr.app import ConfigRecord, Context, RecordDict


def test_locked_guards_concurrent_context_mutations() -> None:
    """Context.locked should make read-modify-write operations atomic."""
    context = Context(0, 0, {}, RecordDict({"counter": ConfigRecord({"value": 0})}), {})

    def increment() -> None:
        with context.locked():
            counter = context.state.config_records["counter"]
            value = counter["value"]
            assert isinstance(value, int)
            sleep(0.001)
            counter["value"] = value + 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: increment(), range(40)))

    assert context.state.config_records["counter"]["value"] == 40


def test_locked_is_reentrant() -> None:
    """Context.locked should support nested framework and user operations."""
    context = Context(0, 0, {}, RecordDict(), {})

    with context.locked(), context.locked():
        context.state["config"] = ConfigRecord({"value": 1})

    assert context.state["config"]["value"] == 1
