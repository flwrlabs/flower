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
"""Tests for the temporary capability contracts."""

from .capability import capability_binding, safe_digest_prefix


def test_capability_binding_v1_golden_vector() -> None:
    """Lock the domain-separated canonical JSON binding contract."""
    binding = capability_binding(
        federation_id="@alice/research",
        fab_hash="0123456789abcdef" * 4,
    )

    assert binding == (
        "flwr-capability-binding-v1-sha256:"
        "903193ce7786ff344fdb2cd1263b6769634c5186e5310615f29ce1392e13518a"
    )


def test_safe_digest_prefix_uses_exactly_twelve_hex_characters() -> None:
    """Keep canonical values intact while shortening human-readable logs."""
    digest = "0123456789abcdef" * 4

    assert safe_digest_prefix(digest) == "0123456789ab"
    assert safe_digest_prefix(f"kind:{digest}") == "0123456789ab"
