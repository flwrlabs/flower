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
"""Temporary MLCommons capability contracts for the proof of concept."""

import hashlib
import json
import re

CAPABILITY_FILE_VERSION = "v1"
CAPABILITY_BINDING_DOMAIN = b"flwr-capability-binding-v1\n"
PARTICIPANT_ID_PREFIX = "flwr-p384-spki-pem-sha256:"
PARTICIPANT_ID_PATTERN = re.compile(
    rf"{re.escape(PARTICIPANT_ID_PREFIX)}[0-9a-f]{{64}}\Z"
)
CAPABILITY_LOG_PREFIX = "[CAPABILITY]"
STORY_LOG_PREFIX = "[STORY]"
LOG_DIGEST_PREFIX_LENGTH = 12


def safe_digest_prefix(value: str) -> str:
    """Return a presentation-only 12-character digest prefix."""
    return value.rsplit(":", maxsplit=1)[-1][:LOG_DIGEST_PREFIX_LENGTH]


def participant_id_from_public_key(public_key: bytes) -> str:
    """Return the POC participant ID for canonical registered key bytes."""
    return PARTICIPANT_ID_PREFIX + hashlib.sha256(public_key).hexdigest()


def capability_binding(federation_id: str, fab_hash: str) -> str:
    """Return the temporary v1 federation-and-FAB job binding."""
    canonical_json = json.dumps(
        {"fab_hash": fab_hash, "federation_id": federation_id},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(CAPABILITY_BINDING_DOMAIN + canonical_json).hexdigest()
    return "flwr-capability-binding-v1-sha256:" + digest
