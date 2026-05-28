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
"""Constants for the SecAgg/SecAgg+ protocol."""


from __future__ import annotations

RECORD_KEY_STATE = "secaggplus_state"
RECORD_KEY_CONFIGS = "secaggplus_configs"


class Stage:
    """Stages for the SecAgg+ protocol."""

    SETUP = "setup"
    SHARE_KEYS = "share_keys"
    COLLECT_MASKED_VECTORS = "collect_masked_vectors"
    UNMASK = "unmask"
    _stages = (SETUP, SHARE_KEYS, COLLECT_MASKED_VECTORS, UNMASK)

    @classmethod
    def all(cls) -> tuple[str, str, str, str]:
        """Return all stages."""
        return cls._stages

    def __new__(cls) -> Stage:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class Key:
    """Keys for the configs in the ConfigRecord."""

    STAGE = "stage"
    SAMPLE_NUMBER = "sample_num"
    SHARE_NUMBER = "share_num"
    THRESHOLD = "threshold"
    CLIPPING_RANGE = "clipping_range"
    TARGET_RANGE = "target_range"
    MOD_RANGE = "mod_range"
    MAX_WEIGHT = "max_weight"
    PUBLIC_KEY_1 = "pk1"
    PUBLIC_KEY_2 = "pk2"
    DESTINATION_LIST = "dsts"
    CIPHERTEXT_LIST = "ctxts"
    SOURCE_LIST = "srcs"
    PARAMETERS = "params"
    MASKED_PARAMETERS = "masked_params"
    ACTIVE_NODE_ID_LIST = "active_nids"
    DEAD_NODE_ID_LIST = "dead_nids"
    NODE_ID_LIST = "nids"
    SHARE_LIST = "shares"

    def __new__(cls) -> Key:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
