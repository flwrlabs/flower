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
"""Conversation participant wire-format tests."""

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Conversation,
    ConversationParticipant,
    CreateConversationResponse,
    ListConversationsResponse,
)


def test_conversation_participants_roundtrip() -> None:
    """Preserve typed user and uint64 agent identities in conversation responses."""
    conversation = Conversation(
        conversation_id=2**64 - 1,
        federation_id="@workspace",
        participants=[
            ConversationParticipant(flwr_aid="alice"),
            ConversationParticipant(agent_id=2**64 - 1),
        ],
    )
    created = CreateConversationResponse.FromString(
        CreateConversationResponse(conversation=conversation).SerializeToString()
    )
    listed = ListConversationsResponse.FromString(
        ListConversationsResponse(conversations=[conversation]).SerializeToString()
    )
    assert created.conversation == conversation
    assert list(listed.conversations) == [conversation]
    assert created.conversation.participants[0].WhichOneof("participant") == "flwr_aid"
    assert created.conversation.participants[1].WhichOneof("participant") == "agent_id"
