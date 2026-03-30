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
"""The SecAgg+ protocol modifier tests."""


import unittest
from collections.abc import Callable
from itertools import product

from flwr.app.message_type import MessageType
from flwr.client.mod import make_ffn
from flwr.common import ConfigRecord, Context, Message, RecordDict
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.common.typing import ConfigRecordValues

from .secaggplus_mod import SecAggPlusState, check_configs, secaggplus_mod


def get_test_handler(
    ctxt: Context,
) -> Callable[[dict[str, ConfigRecordValues]], ConfigRecord]:
    """."""

    def empty_ffn(_msg: Message, _2: Context) -> Message:
        return Message(RecordDict(), reply_to=_msg)

    app = make_ffn(empty_ffn, [secaggplus_mod])

    def func(configs: dict[str, ConfigRecordValues]) -> ConfigRecord:
        in_msg = Message(
            RecordDict({RECORD_KEY_CONFIGS: ConfigRecord(configs)}),
            dst_node_id=123,
            message_type=MessageType.TRAIN,
        )
        out_msg = app(in_msg, ctxt)
        return out_msg.content.config_records[RECORD_KEY_CONFIGS]

    return func


def _make_ctxt() -> Context:
    cfg = ConfigRecord(SecAggPlusState().to_dict())
    return Context(
        run_id=234,
        node_id=123,
        node_config={},
        state=RecordDict({RECORD_KEY_STATE: cfg}),
        run_config={},
    )


def _make_set_state_fn(
    ctxt: Context,
) -> Callable[[str], None]:
    def set_stage(stage: str) -> None:
        state_dict = ctxt.state.config_records[RECORD_KEY_STATE]
        state = SecAggPlusState(**state_dict)
        state.current_stage = stage
        ctxt.state.config_records[RECORD_KEY_STATE] = ConfigRecord(state.to_dict())

    return set_stage


class TestSecAggPlusHandler(unittest.TestCase):
    """Test the SecAgg+ protocol handler."""

    def test_stage_transition(self) -> None:
        """Test stage transition."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        assert Stage.all() == (
            Stage.SETUP,
            Stage.SHARE_KEYS,
            Stage.COLLECT_MASKED_VECTORS,
            Stage.UNMASK,
        )

        valid_transitions = {
            # From one stage to the next stage
            (Stage.UNMASK, Stage.SETUP),
            (Stage.SETUP, Stage.SHARE_KEYS),
            (Stage.SHARE_KEYS, Stage.COLLECT_MASKED_VECTORS),
            (Stage.COLLECT_MASKED_VECTORS, Stage.UNMASK),
            # From any stage to the initial stage
            # Such transitions will log a warning.
            (Stage.SETUP, Stage.SETUP),
            (Stage.SHARE_KEYS, Stage.SETUP),
            (Stage.COLLECT_MASKED_VECTORS, Stage.SETUP),
        }

        invalid_transitions = set(product(Stage.all(), Stage.all())).difference(
            valid_transitions
        )

        # Test valid transitions
        # If the next stage is valid, the function should update the current stage
        # and then raise KeyError or other exceptions when trying to execute SA.
        for current_stage, next_stage in valid_transitions:
            set_stage(current_stage)

            with self.assertRaises(KeyError):
                handler({Key.STAGE: next_stage})

        # Test invalid transitions
        # If the next stage is invalid, the function should raise ValueError
        for current_stage, next_stage in invalid_transitions:
            set_stage(current_stage)

            with self.assertRaises(ValueError):
                handler({Key.STAGE: next_stage})

    def test_stage_setup_check(self) -> None:
        """Test content checking for the setup stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_key_type_pairs = [
            (Key.SAMPLE_NUMBER, int),
            (Key.SHARE_NUMBER, int),
            (Key.THRESHOLD, int),
            (Key.CLIPPING_RANGE, float),
            (Key.TARGET_RANGE, int),
            (Key.MOD_RANGE, int),
        ]

        type_to_test_value: dict[type, ConfigRecordValues] = {
            int: 10,
            bool: True,
            float: 1.0,
            str: "test",
            bytes: b"test",
        }

        valid_configs: dict[str, ConfigRecordValues] = {
            key: type_to_test_value[value_type]
            for key, value_type in valid_key_type_pairs
        }

        # Test valid configs
        try:
            check_configs(Stage.SETUP, ConfigRecord(valid_configs))
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_configs() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[Key.STAGE] = Stage.SETUP

        # Test invalid configs
        for key, value_type in valid_key_type_pairs:
            invalid_configs = valid_configs.copy()

            # Test wrong value type for the key
            for other_type, other_value in type_to_test_value.items():
                if other_type == value_type:
                    continue
                invalid_configs[key] = other_value

                set_stage(Stage.UNMASK)
                with self.assertRaises(TypeError):
                    handler(invalid_configs.copy())

            # Test missing key
            invalid_configs.pop(key)

            set_stage(Stage.UNMASK)
            with self.assertRaises(KeyError):
                handler(invalid_configs.copy())

    def test_stage_share_keys_check(self) -> None:
        """Test content checking for the share keys stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: dict[str, ConfigRecordValues] = {
            "1": [b"public key 1", b"public key 2"],
            "2": [b"public key 1", b"public key 2"],
            "3": [b"public key 1", b"public key 2"],
        }

        # Test valid configs
        try:
            check_configs(Stage.SHARE_KEYS, ConfigRecord(valid_configs))
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_configs() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[Key.STAGE] = Stage.SHARE_KEYS

        # Test invalid configs
        invalid_values: list[ConfigRecordValues] = [
            b"public key 1",
            [b"public key 1"],
            [b"public key 1", b"public key 2", b"public key 3"],
        ]

        for value in invalid_values:
            invalid_configs = valid_configs.copy()
            invalid_configs["1"] = value

            set_stage(Stage.SETUP)
            with self.assertRaises(TypeError):
                handler(invalid_configs.copy())

    def test_stage_collect_masked_vectors_check(self) -> None:
        """Test content checking for the collect masked vectors stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: dict[str, ConfigRecordValues] = {
            Key.CIPHERTEXT_LIST: [b"ctxt!", b"ctxt@", b"ctxt#", b"ctxt?"],
            Key.SOURCE_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid configs
        try:
            check_configs(Stage.COLLECT_MASKED_VECTORS, ConfigRecord(valid_configs))
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_configs() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[Key.STAGE] = Stage.COLLECT_MASKED_VECTORS

        # Test invalid configs
        # Test missing keys
        for key in list(valid_configs.keys()):
            if key == Key.STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs.pop(key)

            set_stage(Stage.SHARE_KEYS)
            with self.assertRaises(KeyError):
                handler(invalid_configs)

        # Test wrong value type for the key
        for key in valid_configs:
            if key == Key.STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs[key] = [3.1415926]

            set_stage(Stage.SHARE_KEYS)
            with self.assertRaises(TypeError):
                handler(invalid_configs)

    def test_stage_unmask_check(self) -> None:
        """Test content checking for the unmasking stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: dict[str, ConfigRecordValues] = {
            Key.ACTIVE_NODE_ID_LIST: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Key.DEAD_NODE_ID_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid configs
        try:
            check_configs(Stage.UNMASK, ConfigRecord(valid_configs))
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_configs() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[Key.STAGE] = Stage.UNMASK

        # Test invalid configs
        # Test missing keys
        for key in list(valid_configs.keys()):
            if key == Key.STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs.pop(key)

            set_stage(Stage.COLLECT_MASKED_VECTORS)
            with self.assertRaises(KeyError):
                handler(invalid_configs)

        # Test wrong value type for the key
        for key in valid_configs:
            if key == Key.STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs[key] = [True, False, True, False]

            set_stage(Stage.COLLECT_MASKED_VECTORS)
            with self.assertRaises(TypeError):
                handler(invalid_configs)
