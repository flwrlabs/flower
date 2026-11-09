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
"""Authenticated event replay, cursors, and current Space visibility."""

from uuid import UUID, uuid4

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Event,
    SpaceParticipant,
    StreamEventsRequest,
)
from flwr.server.superlink.linkstate import InMemoryLinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_handlers import stream_events


def test_stream_replays_the_account_cursor_and_shared_identity() -> None:
    """Resume using the feed seq and keep the shared event UUID stable on replay."""
    state = InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    state.store_event(
        Event(
            event_id=str(uuid4()), type="agent.updated", created_at=now().isoformat()
        ),
        ["alice"],
    )
    event = Event(
        event_id=str(uuid4()),
        type="agent.updated",
        created_at=now().isoformat(),
    )
    event.payload["name"] = "shared"
    state.store_event(event, ["alice", "bob"])
    active = True
    account = AccountInfo(flwr_aid="alice", account_name="Alice")
    stream = stream_events(StreamEventsRequest(), account, state, lambda: active, "1")
    result = next(stream)
    assert result.seq == 2 and result.event.event_id == event.event_id
    assert result.event.payload["name"] == "shared"
    active = False
    assert not list(stream)
    active = True
    replay = stream_events(
        StreamEventsRequest(after_seq=1), account, state, lambda: active
    )
    assert next(replay) == result
    replay.close()
    bob = stream_events(
        StreamEventsRequest(),
        AccountInfo(flwr_aid="bob", account_name="Bob"),
        state,
        lambda: active,
    )
    assert next(bob).seq == 1
    bob.close()


def test_stream_skips_spaces_the_user_can_no_longer_access() -> None:
    """Historical feed membership does not grant current access to a Space."""
    state = InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(flwr_aid="bob"),
        ]
    )
    state.space_store[UUID(space.id)].participants.remove(
        SpaceParticipant(flwr_aid="bob")
    )
    event = Event(
        event_id=str(uuid4()), type="agent.updated", created_at=now().isoformat()
    )
    state.store_event(event, ["bob"])
    stream = stream_events(
        StreamEventsRequest(),
        AccountInfo(flwr_aid="bob", account_name="Bob"),
        state,
        lambda: True,
    )
    assert next(stream).event.event_id == event.event_id
    stream.close()


@pytest.mark.parametrize("cursor", ["invalid", "-1", str(2**64)])
def test_bad_cursor_is_rejected_before_streaming(cursor: str) -> None:
    """Return an ordinary API error before starting an invalid stream."""
    state = InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    with pytest.raises(FlowerError) as error:
        stream_events(
            StreamEventsRequest(),
            AccountInfo(flwr_aid="alice", account_name="Alice"),
            state,
            lambda: True,
            cursor,
        )
    assert error.value.code == ApiErrorCode.INVALID_USER_EVENT_CURSOR
