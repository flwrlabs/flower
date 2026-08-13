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
"""Tests for protobuf HTTP translation helpers."""

from flwr.supercore.error import ApiErrorCode, FlowerError

from .translation import ProtobufTranslationMiddleware


def test_response_for_rejects_scalar_iterables() -> None:
    """Reject scalar byte and text values instead of treating them as streams."""
    for result in ("invalid", b"invalid", bytearray(b"invalid")):
        try:
            ProtobufTranslationMiddleware._response_for(  # pylint: disable=W0212
                result
            )
        except FlowerError as exc:
            assert exc.code == ApiErrorCode.INVALID_HANDLER_RESPONSE
        else:
            raise AssertionError("Expected invalid handler response error")
