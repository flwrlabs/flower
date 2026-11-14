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
"""Tests for legacy client DP mods."""

from unittest.mock import Mock

import pytest

from flwr.app import Context, Message, MessageType
from flwr.clientapp.typing import Mod

from .centraldp_mods import adaptiveclipping_mod, fixedclipping_mod
from .localdp_mod import LocalDpMod


@pytest.mark.parametrize(
    "mod",
    [
        fixedclipping_mod,
        adaptiveclipping_mod,
        LocalDpMod(clipping_norm=1.0, sensitivity=1.0, epsilon=1.0, delta=0.01),
    ],
)
def test_non_training_message_bypasses_dp(mod: Mod) -> None:
    """Forward non-training messages without requiring legacy parameter records."""
    message = Mock(spec=Message, metadata=Mock(message_type=MessageType.QUERY))
    context = Mock(spec=Context)
    reply = Mock(spec=Message)
    call_next = Mock(return_value=reply)

    assert mod(message, context, call_next) is reply
    call_next.assert_called_once_with(message, context)
