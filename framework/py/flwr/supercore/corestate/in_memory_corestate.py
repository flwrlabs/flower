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
"""In-memory CoreState implementation."""

import secrets
from dataclasses import dataclass
from logging import DEBUG, WARNING
from threading import Lock

from flwr.common import now
from flwr.common.constant import (
    FLWR_APP_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_INITIAL_GRACE_PERIOD,
    HEARTBEAT_PATIENCE,
)
from flwr.common.logger import log
from flwr.supercore.utils import mask_string

from ..object_store import ObjectStore
from .corestate import CoreState


@dataclass
class TokenRecord:
    """Record containing token and heartbeat information."""

    token: str
    active_until: float


class InMemoryCoreState(CoreState):
    """In-memory CoreState implementation."""

    def __init__(self, object_store: ObjectStore) -> None:
        self._object_store = object_store
        # Store run ID to token mapping and token to run ID mapping
        self.token_store: dict[int, TokenRecord] = {}
        self.token_to_run_id: dict[str, int] = {}
        self.lock_token_store = Lock()

    @property
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""
        return self._object_store

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        with self.lock_token_store:
            if run_id in self.token_store:
                log(
                    DEBUG,
                    "ClientApp token request denied: run_id=%s already has a token",
                    run_id,
                )
                return None  # Token already created for this run ID

            self.token_store[run_id] = TokenRecord(
                token=token,
                active_until=now().timestamp() + HEARTBEAT_INITIAL_GRACE_PERIOD,
            )
            self.token_to_run_id[token] = run_id
        log(
            DEBUG,
            "ClientApp token created: run_id=%s token=%s initial_grace_s=%s",
            run_id,
            mask_string(token),
            HEARTBEAT_INITIAL_GRACE_PERIOD,
        )
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        self._cleanup_expired_tokens()
        with self.lock_token_store:
            record = self.token_store.get(run_id)
            return record is not None and record.token == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        with self.lock_token_store:
            record = self.token_store.pop(run_id, None)
            if record is not None:
                self.token_to_run_id.pop(record.token, None)
        if record is not None:
            log(
                DEBUG,
                "ClientApp token deleted after completion: run_id=%s token=%s",
                run_id,
                mask_string(record.token),
            )

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        self._cleanup_expired_tokens()
        with self.lock_token_store:
            return self.token_to_run_id.get(token)

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        # Clean up expired tokens
        self._cleanup_expired_tokens()

        with self.lock_token_store:
            # Return False if token is not found
            if token not in self.token_to_run_id:
                log(
                    WARNING,
                    "ClientApp heartbeat rejected: token=%s is unknown or expired",
                    mask_string(token),
                )
                return False

            # Get the run_id and update heartbeat info
            run_id = self.token_to_run_id[token]
            record = self.token_store[run_id]
            current = now().timestamp()
            record.active_until = (
                current + HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
            )
            log(
                DEBUG,
                "ClientApp heartbeat accepted: run_id=%s token=%s lease_s=%s",
                run_id,
                mask_string(token),
                HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL,
            )
            return True

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens and perform additional cleanup.

        This method is called before token operations to ensure integrity.
        Subclasses can override `_on_tokens_expired` to add custom cleanup logic.
        """
        with self.lock_token_store:
            current = now().timestamp()
            expired_records: list[tuple[int, float]] = []
            for run_id, record in list(self.token_store.items()):
                if record.active_until < current:
                    expired_records.append((run_id, record.active_until))
                    # Remove from both stores
                    del self.token_store[run_id]
                    self.token_to_run_id.pop(record.token, None)

            # Hook for subclasses
            if expired_records:
                log(
                    WARNING,
                    "Expired %s ClientApp token(s): run_ids=%s",
                    len(expired_records),
                    [run_id for run_id, _ in expired_records],
                )
                self._on_tokens_expired(expired_records)

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Handle cleanup of expired tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """
