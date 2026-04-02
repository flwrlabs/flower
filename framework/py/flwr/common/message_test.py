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
"""Message tests."""


import time
from collections import namedtuple
from collections.abc import Callable
from contextlib import ExitStack
from copy import copy
from itertools import product
from typing import Any

import pytest

from flwr.supercore.inflatable.inflatable_object import (
    get_object_body,
    get_object_children_ids_from_object_content,
    get_object_type_from_object_content,
)

from ..app.error import Error
from ..app.metadata import Metadata
from ..supercore.date import now
from . import ConfigRecord, RecordDict
from .constant import MESSAGE_TTL_TOLERANCE
from .message import (
    DEFAULT_TTL,
    Message,
    MessageInitializationError,
    make_message,
    remove_content_from_message,
)
from .serde import message_to_proto
from .serde_test import RecordMaker


@pytest.mark.parametrize(
    "content_fn, error_fn, context",
    [
        (
            lambda maker: maker.recorddict(1, 1, 1),
            None,
            None,
        ),  # check when only content is set
        (None, lambda code: Error(code=code), None),  # check when only error is set
        (
            lambda maker: maker.recorddict(1, 1, 1),
            lambda code: Error(code=code),
            pytest.raises(TypeError),
        ),  # check when both are set (ERROR)
        (None, None, pytest.raises(TypeError)),  # check when neither is set (ERROR)
    ],
)
def test_message_creation(
    content_fn: Callable[[RecordMaker], RecordDict],
    error_fn: Callable[[int], Error],
    context: Any,
) -> None:
    """Test Message creation attempting to pass content and/or error."""
    # Prepare
    maker = RecordMaker(state=2)
    current_time = time.time()

    with ExitStack() as stack:
        if context:
            stack.enter_context(context)

        metadata = maker.metadata()
        message = make_message(
            metadata=metadata,
            content=None if content_fn is None else content_fn(maker),
            error=None if error_fn is None else error_fn(0),
        )

        assert message.metadata.created_at > current_time
        assert message.metadata.created_at < time.time()


def create_message_with_content(ttl: float | None = None) -> Message:
    """Create a Message with content."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    if ttl:
        metadata.ttl = ttl
    return make_message(metadata=metadata, content=RecordDict())


def create_message_with_error(ttl: float | None = None) -> Message:
    """Create a Message with error."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    if ttl:
        metadata.ttl = ttl
    return make_message(metadata=metadata, error=Error(code=1))


@pytest.mark.parametrize(
    "message_creation_fn",
    [
        create_message_with_content,
        create_message_with_error,
    ],
)
def test_altering_message(
    message_creation_fn: Callable[[], Message],
) -> None:
    """Test that a message with content doesn't allow setting an error.

    And viceversa.
    """
    message = message_creation_fn()

    with pytest.raises(ValueError):
        if message.has_content():
            message.error = Error(code=123)
        if message.has_error():
            message.content = RecordDict()


@pytest.mark.parametrize(
    "message_creation_fn,ttl,reply_ttl",
    [
        (create_message_with_content, 1e6, None),
        (create_message_with_error, 1e6, None),
        (create_message_with_content, 1e6, 3600),
        (create_message_with_error, 1e6, 3600),
    ],
)
def test_create_reply(
    message_creation_fn: Callable[[float], Message],
    ttl: float,
    reply_ttl: float | None,
) -> None:
    """Test reply creation from message."""
    message: Message = message_creation_fn(ttl)

    time.sleep(0.1)

    if message.has_error():
        dummy_error = Error(code=0, reason="it crashed")
        reply_message = Message(dummy_error, reply_to=message, ttl=reply_ttl)
    else:
        reply_message = Message(RecordDict(), reply_to=message, ttl=reply_ttl)

    # Ensure reply has a higher timestamp
    assert message.metadata.created_at < reply_message.metadata.created_at
    if reply_ttl:
        # Ensure the TTL is the one specify upon reply creation
        assert reply_message.metadata.ttl == reply_ttl
    else:
        # Ensure reply ttl is lower (since it uses remaining time left)
        assert message.metadata.ttl > reply_message.metadata.ttl

    assert message.metadata.src_node_id == reply_message.metadata.dst_node_id
    assert message.metadata.dst_node_id == reply_message.metadata.src_node_id
    assert reply_message.metadata.reply_to_message_id == message.metadata.message_id


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (
            Metadata,
            {
                "run_id": 123,
                "message_id": "msg_456",
                "src_node_id": 1,
                "dst_node_id": 2,
                "reply_to_message_id": "reply_789",
                "group_id": "group_xyz",
                "created_at": 1234567890.0,
                "ttl": 10.0,
                "message_type": "query",
            },
        ),
        (Error, {"code": 1, "reason": "reason_098"}),
        (
            Message,
            {
                "metadata": RecordMaker(1).metadata(),
                "content": RecordMaker(1).recorddict(1, 1, 1),
            },
        ),
        (
            Message,
            {
                "metadata": RecordMaker(2).metadata(),
                "error": Error(0, "some reason"),
            },
        ),
    ],
)
def test_repr(cls: type, kwargs: dict[str, Any]) -> None:
    """Test string representations of Metadata/Message/Error."""
    # Prepare
    anon_cls = namedtuple(cls.__qualname__, kwargs.keys())  # type: ignore
    expected = anon_cls(**kwargs)
    actual = cls(**kwargs)

    # Assert
    assert str(actual) == str(expected)


@pytest.mark.parametrize(
    "message_creation_fn,initial_ttl,reply_ttl,expected_reply_ttl",
    [
        # Case where the reply_ttl is larger than the allowed TTL
        (create_message_with_content, 20, 30, 20),
        (create_message_with_error, 20, 30, 20),
        # Case where the reply_ttl is within the allowed range
        (create_message_with_content, 20, 10, 10),
        (create_message_with_error, 20, 10, 10),
    ],
)
def test_reply_ttl_limitation(
    message_creation_fn: Callable[[float], Message],
    initial_ttl: float,
    reply_ttl: float,
    expected_reply_ttl: float,
) -> None:
    """Test that the reply TTL does not exceed the allowed TTL."""
    message = message_creation_fn(initial_ttl)

    if message.has_error():
        dummy_error = Error(code=0, reason="test error")
        reply_message = Message(dummy_error, reply_to=message, ttl=reply_ttl)
    else:
        reply_message = Message(RecordDict(), reply_to=message, ttl=reply_ttl)

    assert reply_message.metadata.ttl - expected_reply_ttl <= MESSAGE_TTL_TOLERANCE, (
        f"Expected TTL to be <= {expected_reply_ttl}, "
        f"but got {reply_message.metadata.ttl}"
    )


@pytest.mark.parametrize(
    "ttl,group_id,use_keyword",
    product([None, 10.0, 10], [None, "group_xyz"], [True, False]),
)
def test_create_ins_message_success(
    ttl: float | None, group_id: str | None, use_keyword: bool
) -> None:
    """Test creating an instruction message with content."""
    # Prepare
    current_time = now().timestamp()
    kwargs = {k: v for k, v in [("ttl", ttl), ("group_id", group_id)] if v is not None}

    # Execute
    if use_keyword:
        msg = Message(
            content=RecordDict(),
            dst_node_id=123,
            message_type="query",
            **kwargs,  # type: ignore
        )
    else:
        msg = Message(RecordDict(), 123, "query", **kwargs)  # type: ignore

    # Assert
    assert msg.has_content() and msg.content == RecordDict()
    assert msg.metadata.dst_node_id == 123
    assert msg.metadata.message_type == "query"
    assert msg.metadata.ttl == (ttl or DEFAULT_TTL)
    assert msg.metadata.group_id == (group_id or "")
    assert current_time < msg.metadata.created_at < now().timestamp()
    assert msg.metadata.run_id == 0  # Should be unset
    assert msg.metadata.message_id == ""  # Should be unset
    assert msg.metadata.src_node_id == 0  # Should be unset
    assert msg.metadata.reply_to_message_id == ""  # Should be unset


@pytest.mark.parametrize(
    "content_or_error,ttl",
    product([RecordDict(), Error(0)], [None, 10.0, 20]),
)
def test_create_reply_message_success(
    content_or_error: RecordDict | Error, ttl: float | None
) -> None:
    """Test creating a reply message."""
    # Prepare
    msg = make_message(content=RecordDict(), metadata=RecordMaker(1).metadata())
    current_time = msg.metadata.created_at

    # Execute
    reply = Message(content_or_error, reply_to=msg, ttl=ttl)

    # Assert
    assert reply.metadata.run_id == msg.metadata.run_id
    assert reply.metadata.src_node_id == msg.metadata.dst_node_id
    assert reply.metadata.dst_node_id == msg.metadata.src_node_id
    assert reply.metadata.reply_to_message_id == msg.metadata.message_id
    assert reply.metadata.group_id == msg.metadata.group_id
    assert current_time < reply.metadata.created_at < now().timestamp()
    assert (
        reply.metadata.created_at + reply.metadata.ttl
        <= msg.metadata.created_at + msg.metadata.ttl
    )
    assert reply.metadata.message_type == msg.metadata.message_type
    assert reply.metadata.message_id == ""  # Should be unset
    if isinstance(content_or_error, RecordDict):
        assert reply.has_content()
    else:
        assert reply.has_error()


@pytest.mark.parametrize(
    "args,kwargs",
    [
        # Pass Error instead of content
        ((Error(0), 123, "query"), {}),
        # Too many positional args
        ((RecordDict(), 123, "query", 123), {}),
        # Too few positional args
        ((RecordDict(),), {}),
        ((RecordDict(), 123), {}),
        # Use keyword args when positional args are set
        ((RecordDict(), 123, "query"), {"content": RecordDict()}),
        ((RecordDict(), 123, "query"), {"dst_node_id": 123}),
        ((RecordDict(), 123, "query"), {"message_type": "query"}),
        # Use keyword args not allowed
        ((RecordDict(), 123, "query"), {"metadata": RecordMaker(1).metadata()}),
        ((RecordDict(), 123, "query"), {"error": Error(0)}),
        (
            (RecordDict(), 123, "query"),
            {"reply_to": Message(RecordDict(), 123, "query")},
        ),
        # Use invalid arg types
        (("wrong type", 123, "query"), {}),
        ((RecordDict(), "wrong type", "query"), {}),
        ((RecordDict(), 123, 456), {}),
        ((RecordDict(), 123, "query"), {"ttl": "wrong type"}),
        ((RecordDict(), 123, "query"), {"group_id": 123}),
        ((RecordDict(), 123, "query"), {"group_id": 123.0}),
    ],
)
def test_create_ins_message_failure(args: Any, kwargs: dict[str, Any]) -> None:
    """Test creating an instruction message with error."""
    # Execute
    with pytest.raises(MessageInitializationError):
        Message(*args, **kwargs)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        # Too many positional args
        ((RecordDict(), 123), {}),
        ((Error(0), 123), {}),
        # Too few positional args
        ((), {}),
        # Use keyword args when positional args are set
        ((RecordDict(),), {"content": RecordDict()}),
        ((Error(0),), {"error": Error(0)}),
        # Use keyword args not allowed
        ((RecordDict(),), {"metadata": RecordMaker(1).metadata()}),
        ((Error(0),), {"metadata": RecordMaker(1).metadata()}),
        ((RecordDict(),), {"dst_node_id": 123}),
        ((Error(0),), {"dst_node_id": 123}),
        ((RecordDict(),), {"message_type": "query"}),
        ((Error(0),), {"message_type": "query"}),
        ((RecordDict(),), {"group_id": "group_xyz"}),
        ((Error(0),), {"group_id": "group_xyz"}),
        # Use invalid arg types
        (("wrong type",), {}),
        ((123,), {}),
        ((RecordDict(),), {"ttl": "wrong type"}),
        ((Error(0),), {"ttl": "wrong type"}),
    ],
)
def test_create_reply_message_failure(args: Any, kwargs: dict[str, Any]) -> None:
    """Test creating a reply message with error."""
    # Prepare
    msg = make_message(content=RecordDict(), metadata=RecordMaker(1).metadata())

    # Execute
    with pytest.raises(MessageInitializationError):
        Message(*args, reply_to=msg, **kwargs)


def test_inflate_deflate_message_with_content() -> None:
    """Test inflation and deflation of a Message carrying a RecordDict."""
    # Prepare
    msg = make_message(content=RecordDict(), metadata=RecordMaker(1).metadata())

    # Assert
    # Expected children
    assert msg.children == {msg.content.object_id: msg.content}

    msg_b = msg.deflate()

    # Assert
    # Class name matches
    assert get_object_type_from_object_content(msg_b) == msg.__class__.__qualname__
    # Header contains the Object ID of the message content
    assert get_object_children_ids_from_object_content(msg_b) == [msg.content.object_id]
    # Body of deflated Message matches its direct protobuf serialization
    # when the content is removed
    msg_copy = copy(msg)
    msg_copy.content = None  # type: ignore
    msg_copy.metadata.__dict__["_message_id"] = ""
    assert message_to_proto(msg_copy).SerializeToString(
        deterministic=True
    ) == get_object_body(msg_b, Message)

    msg_ = Message.inflate(msg_b, children=msg.children)
    # Assert
    # Both objects are identical
    assert msg.object_id == msg_.object_id

    # Assert
    # Inflate but passing no children
    with pytest.raises(ValueError):
        Message.inflate(msg_b)
    # Inflate but passing children with wrong Object ID
    with pytest.raises(ValueError):
        Message.inflate(msg_b, children={"123": RecordDict()})


def test_inflate_deflate_message_with_error() -> None:
    """Test inflation and deflation of a Message carrying an Error."""
    # Prepare
    msg = make_message(error=Error(code=1), metadata=RecordMaker(1).metadata())

    # Assert
    # Expected children
    assert msg.children is None

    msg_b = msg.deflate()

    # Assert
    # Class name matches
    assert get_object_type_from_object_content(msg_b) == msg.__class__.__qualname__
    # Header contains the Object ID of the message content
    assert get_object_children_ids_from_object_content(msg_b) == []
    # Body of deflated Message matches its direct protobuf serialization
    msg_copy = copy(msg)
    msg_copy.metadata.__dict__["_message_id"] = ""
    assert message_to_proto(msg_copy).SerializeToString(
        deterministic=True
    ) == get_object_body(msg_b, Message)

    # Inflate without passing any children
    msg_ = Message.inflate(msg_b)
    # Assert
    # Both objects are identical
    assert msg.object_id == msg_.object_id

    # Assert
    # Inflate but passing children
    with pytest.raises(ValueError):
        Message.inflate(msg_b, children={"123": RecordDict()})


def test_object_id_excludes_message_id_in_metadata() -> None:
    """Test inflation and deflation of a Message carrying an Error."""
    # Prepare
    msg = make_message(error=Error(code=1), metadata=RecordMaker(1).metadata())
    object_id = msg.object_id

    # Modify message_id
    msg.metadata.__dict__["_message_id"] = "1234"

    # Assert
    assert object_id == msg.object_id


def test_remove_content_from_message() -> None:
    """Test remove_content method."""
    # Prepare message w/ content
    msg = make_message(
        content=RecordDict({"a": ConfigRecord()}), metadata=RecordMaker(1).metadata()
    )

    # Execute (expected content to be an empty RecordDict)
    msg_ = remove_content_from_message(msg)
    assert msg_.content == RecordDict()
    assert msg_.metadata == msg.metadata

    # Prepare message w/ error
    msg = make_message(error=Error(code=1), metadata=RecordMaker(1).metadata())
    # Execute (expected to have an identical message returned)
    msg_ = remove_content_from_message(msg)
    assert msg_.error == msg.error
    assert msg_.object_id == msg.object_id
