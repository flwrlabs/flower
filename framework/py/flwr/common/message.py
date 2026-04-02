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
"""Message."""


from __future__ import annotations

from logging import WARNING
from typing import Any, cast, overload

from flwr.common.logger import warn_deprecated_feature
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.proto.message_pb2 import Metadata as ProtoMetadata  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectIDs  # pylint: disable=E0611
from flwr.supercore.date import now
from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_descendant_object_ids,
    get_object_body,
    get_object_children_ids_from_object_content,
)

from ..app.error import Error
from ..app.metadata import Metadata
from .constant import MESSAGE_TTL_TOLERANCE
from .logger import log
from .record import RecordDict
from .serde_utils import (
    error_from_proto,
    error_to_proto,
    metadata_from_proto,
    metadata_to_proto,
)

DEFAULT_TTL = 43200  # This is 12 hours
MESSAGE_INIT_ERROR_MESSAGE = (
    "Invalid arguments for Message. Expected one of the documented "
    "signatures: Message(content: RecordDict, dst_node_id: int, message_type: str,"
    " *, [ttl: float, group_id: str]) or Message(content: RecordDict | error: Error,"
    " *, reply_to: Message, [ttl: float])."
)


class _WarningTracker:
    """A class to track warnings for deprecated properties."""

    def __init__(self) -> None:
        # These variables are used to ensure that the deprecation warnings
        # for the deprecated properties/class are logged only once.
        self.create_error_reply_logged = False
        self.create_reply_logged = False


_warning_tracker = _WarningTracker()


class MessageInitializationError(TypeError):
    """Error raised when initializing a message with invalid arguments."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or MESSAGE_INIT_ERROR_MESSAGE)


class Message(InflatableObject):
    """Represents a message exchanged between ClientApp and ServerApp.

    This class encapsulates the payload and metadata necessary for communication
    between a ClientApp and a ServerApp.

    Parameters
    ----------
    content : Optional[RecordDict] (default: None)
        Holds records either sent by another entity (e.g. sent by the server-side
        logic to a client, or vice-versa) or that will be sent to it.
    error : Optional[Error] (default: None)
        A dataclass that captures information about an error that took place
        when processing another message.
    dst_node_id : Optional[int] (default: None)
        An identifier for the node receiving this message.
    message_type : Optional[str] (default: None)
        A string that encodes the action to be executed on
        the receiving end.
    ttl : Optional[float] (default: None)
        Time-to-live (TTL) for this message in seconds. If `None` (default),
        the TTL is set to 43,200 seconds (12 hours).
    group_id : Optional[str] (default: None)
        An identifier for grouping messages. In some settings, this is used as
        the FL round.
    reply_to : Optional[Message] (default: None)
        The instruction message to which this message is a reply. This message does
        not retain the original message's content but derives its metadata from it.
    """

    @overload
    def __init__(  # pylint: disable=too-many-arguments
        self,
        content: RecordDict,
        dst_node_id: int,
        message_type: str,
        *,
        ttl: float | None = None,
        group_id: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self, content: RecordDict, *, reply_to: Message, ttl: float | None = None
    ) -> None: ...

    @overload
    def __init__(
        self, error: Error, *, reply_to: Message, ttl: float | None = None
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        dst_node_id: int | None = None,
        message_type: str | None = None,
        content: RecordDict | None = None,
        error: Error | None = None,
        ttl: float | None = None,
        group_id: str | None = None,
        reply_to: Message | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        # Set positional arguments
        content, error, dst_node_id, message_type = _extract_positional_args(
            *args,
            content=content,
            error=error,
            dst_node_id=dst_node_id,
            message_type=message_type,
        )
        _check_arg_types(
            dst_node_id=dst_node_id,
            message_type=message_type,
            content=content,
            error=error,
            ttl=ttl,
            group_id=group_id,
            reply_to=reply_to,
            metadata=metadata,
        )

        # Set metadata directly (This is for internal use only)
        if metadata is not None:
            # When metadata is set, all other arguments must be None,
            # except `content`, `error`, or `content_or_error`
            if any(
                x is not None
                for x in [dst_node_id, message_type, ttl, group_id, reply_to]
            ):
                raise MessageInitializationError(
                    f"Invalid arguments for {Message.__qualname__}. "
                    "Expected only `metadata` to be set when creating a message "
                    "with provided metadata."
                )

        # Create metadata for an instruction message
        elif reply_to is None:
            # Check arguments
            # `content`, `dst_node_id` and `message_type` must be set
            if not (
                isinstance(content, RecordDict)
                and isinstance(dst_node_id, int)
                and isinstance(message_type, str)
            ):
                raise MessageInitializationError()

            # Set metadata
            metadata = Metadata(
                run_id=0,  # Will be set before pushed
                message_id="",  # Will be set by the SuperLink
                src_node_id=0,  # Will be set before pushed
                dst_node_id=dst_node_id,
                # Instruction messages do not reply to any message
                reply_to_message_id="",
                group_id=group_id or "",
                created_at=now().timestamp(),
                ttl=ttl or DEFAULT_TTL,
                message_type=message_type,
            )

        # Create metadata for a reply message
        else:
            # Check arguments
            # `dst_node_id`, `message_type` and `group_id` must not be set
            if any(x is not None for x in [dst_node_id, message_type, group_id]):
                raise MessageInitializationError()

            # Set metadata
            current = now().timestamp()
            metadata = Metadata(
                run_id=reply_to.metadata.run_id,
                message_id="",  # Will be set by the SuperLink
                src_node_id=reply_to.metadata.dst_node_id,
                dst_node_id=reply_to.metadata.src_node_id,
                reply_to_message_id=reply_to.metadata.message_id,
                group_id=reply_to.metadata.group_id,
                created_at=current,
                ttl=_limit_reply_ttl(current, ttl, reply_to),
                message_type=reply_to.metadata.message_type,
            )

        metadata.delivered_at = ""  # Backward compatibility
        var_dict = {
            "_metadata": metadata,
            "_content": content,
            "_error": error,
        }
        self.__dict__.update(var_dict)

    @property
    def metadata(self) -> Metadata:
        """A dataclass including information about the message to be executed."""
        return cast(Metadata, self.__dict__["_metadata"])

    @property
    def content(self) -> RecordDict:
        """The content of this message."""
        if self.__dict__["_content"] is None:
            raise ValueError(
                "Message content is None. Use <message>.has_content() "
                "to check if a message has content."
            )
        return cast(RecordDict, self.__dict__["_content"])

    @content.setter
    def content(self, value: RecordDict) -> None:
        """Set content."""
        if self.__dict__["_error"] is None:
            self.__dict__["_content"] = value
        else:
            raise ValueError("A message with an error set cannot have content.")

    @property
    def error(self) -> Error:
        """Error captured by this message."""
        if self.__dict__["_error"] is None:
            raise ValueError(
                "Message error is None. Use <message>.has_error() "
                "to check first if a message carries an error."
            )
        return cast(Error, self.__dict__["_error"])

    @error.setter
    def error(self, value: Error) -> None:
        """Set error."""
        if self.has_content():
            raise ValueError("A message with content set cannot carry an error.")
        self.__dict__["_error"] = value

    def has_content(self) -> bool:
        """Return True if message has content, else False."""
        return self.__dict__["_content"] is not None

    def has_error(self) -> bool:
        """Return True if message has an error, else False."""
        return self.__dict__["_error"] is not None

    def create_error_reply(self, error: Error, ttl: float | None = None) -> Message:
        """Construct a reply message indicating an error happened.

        Parameters
        ----------
        error : Error
            The error that was encountered.
        ttl : Optional[float] (default: None)
            Time-to-live for this message in seconds. If unset, it will be set based
            on the remaining time for the received message before it expires. This
            follows the equation:

            ttl = msg.meta.ttl - (reply.meta.created_at - msg.meta.created_at)

        Returns
        -------
        message : Message
            A Message containing only the relevant error and metadata.
        """
        if not _warning_tracker.create_error_reply_logged:
            _warning_tracker.create_error_reply_logged = True
            warn_deprecated_feature(
                "`Message.create_error_reply` is deprecated. "
                "Instead of calling `some_message.create_error_reply(some_error, "
                "ttl=...)`, use `Message(some_error, reply_to=some_message, ttl=...)`."
            )
        if ttl is not None:
            return Message(error, reply_to=self, ttl=ttl)
        return Message(error, reply_to=self)

    def create_reply(self, content: RecordDict, ttl: float | None = None) -> Message:
        """Create a reply to this message with specified content and TTL.

        The method generates a new `Message` as a reply to this message.
        It inherits 'run_id', 'src_node_id', 'dst_node_id', and 'message_type' from
        this message and sets 'reply_to_message_id' to the ID of this message.

        Parameters
        ----------
        content : RecordDict
            The content for the reply message.
        ttl : Optional[float] (default: None)
            Time-to-live for this message in seconds. If unset, it will be set based
            on the remaining time for the received message before it expires. This
            follows the equation:

            ttl = msg.meta.ttl - (reply.meta.created_at - msg.meta.created_at)

        Returns
        -------
        Message
            A new `Message` instance representing the reply.
        """
        if not _warning_tracker.create_reply_logged:
            _warning_tracker.create_reply_logged = True
            warn_deprecated_feature(
                "`Message.create_reply` is deprecated. "
                "Instead of calling `some_message.create_reply(some_content, ttl=...)`"
                ", use `Message(some_content, reply_to=some_message, ttl=...)`."
            )
        if ttl is not None:
            return Message(content, reply_to=self, ttl=ttl)
        return Message(content, reply_to=self)

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join(
            [
                f"{k.lstrip('_')}={v!r}"
                for k, v in self.__dict__.items()
                if v is not None
            ]
        )
        return f"{self.__class__.__qualname__}({view})"

    @property
    def children(self) -> dict[str, InflatableObject] | None:
        """Return a dictionary of a single RecordDict with its Object IDs as key."""
        return {self.content.object_id: self.content} if self.has_content() else None

    def deflate(self) -> bytes:
        """Deflate message."""
        # Exclude message_id from serialization
        proto_metadata: ProtoMetadata = metadata_to_proto(self.metadata)
        proto_metadata.message_id = ""
        # Store message metadata and error in object body
        obj_body = ProtoMessage(
            metadata=proto_metadata,
            content=None,
            error=error_to_proto(self.error) if self.has_error() else None,
        ).SerializeToString(deterministic=True)

        return add_header_to_object_body(object_body=obj_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> Message:
        """Inflate an Message from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the Message.
        children : Optional[dict[str, InflatableObject]] (default: None)
            Dictionary of children InflatableObjects mapped to their Object IDs.
            These children enable the full inflation of the Message.

        Returns
        -------
        Message
            The inflated Message.
        """
        if children is None:
            children = {}

        # Get the children id from the deflated message
        children_ids = get_object_children_ids_from_object_content(object_content)

        # If the message had content, only one children is possible
        # If the message carried an error, the returned listed should be empty
        if children_ids != list(children.keys()):
            raise ValueError(
                f"Mismatch in children object IDs: expected {children_ids}, but "
                f"received {list(children.keys())}. The provided children must exactly "
                "match the IDs specified in the object head."
            )

        # Inflate content
        obj_body = get_object_body(object_content, cls)
        proto_message = ProtoMessage.FromString(obj_body)

        # Prepare content if error wasn't set in protobuf message
        if proto_message.HasField("error"):
            content = None
            error = error_from_proto(proto_message.error)
        else:
            content = cast(RecordDict, children[children_ids[0]])
            error = None
        # Return message
        return make_message(
            metadata=metadata_from_proto(proto_message.metadata),
            content=content,
            error=error,
        )


def make_message(
    metadata: Metadata, content: RecordDict | None = None, error: Error | None = None
) -> Message:
    """Create a message with the provided metadata, content, and error."""
    return Message(metadata=metadata, content=content, error=error)  # type: ignore


def remove_content_from_message(message: Message) -> Message:
    """Return a copy of the Message but with an empty RecordDict as content.

    If message has no content, it returns itself.
    """
    if message.has_error():
        return message

    return make_message(metadata=message.metadata, content=RecordDict())


def _limit_reply_ttl(
    current: float, reply_ttl: float | None, reply_to: Message
) -> float:
    """Limit the TTL of a reply message such that it does exceed the expiration time of
    the message it replies to."""
    # Calculate the maximum allowed TTL
    max_allowed_ttl = reply_to.metadata.created_at + reply_to.metadata.ttl - current

    if reply_ttl is not None and reply_ttl - max_allowed_ttl > MESSAGE_TTL_TOLERANCE:
        log(
            WARNING,
            "The reply TTL of %.2f seconds exceeded the "
            "allowed maximum of %.2f seconds. "
            "The TTL has been updated to the allowed maximum.",
            reply_ttl,
            max_allowed_ttl,
        )
        return max_allowed_ttl

    return reply_ttl or max_allowed_ttl


def _extract_positional_args(
    *args: Any,
    content: RecordDict | None,
    error: Error | None,
    dst_node_id: int | None,
    message_type: str | None,
) -> tuple[RecordDict | None, Error | None, int | None, str | None]:
    """Extract positional arguments for the `Message` constructor."""
    content_or_error = args[0] if args else None
    if len(args) > 1:
        if dst_node_id is not None:
            raise MessageInitializationError()
        dst_node_id = args[1]
    if len(args) > 2:
        if message_type is not None:
            raise MessageInitializationError()
        message_type = args[2]
    if len(args) > 3:
        raise MessageInitializationError()

    # One and only one of `content_or_error`, `content` and `error` must be set
    if sum(x is not None for x in [content_or_error, content, error]) != 1:
        raise MessageInitializationError()

    # Set `content` or `error` based on `content_or_error`
    if content_or_error is not None:  # This means `content` and `error` are None
        if isinstance(content_or_error, RecordDict):
            content = content_or_error
        elif isinstance(content_or_error, Error):
            error = content_or_error
        else:
            raise MessageInitializationError()
    return content, error, dst_node_id, message_type


def _check_arg_types(  # pylint: disable=too-many-arguments, R0917
    dst_node_id: int | None = None,
    message_type: str | None = None,
    content: RecordDict | None = None,
    error: Error | None = None,
    ttl: float | None = None,
    group_id: str | None = None,
    reply_to: Message | None = None,
    metadata: Metadata | None = None,
) -> None:
    """Check argument types for the `Message` constructor."""
    # pylint: disable=too-many-boolean-expressions
    if (
        (dst_node_id is None or isinstance(dst_node_id, int))
        and (message_type is None or isinstance(message_type, str))
        and (content is None or isinstance(content, RecordDict))
        and (error is None or isinstance(error, Error))
        and (ttl is None or isinstance(ttl, (int | float)))
        and (group_id is None or isinstance(group_id, str))
        and (reply_to is None or isinstance(reply_to, Message))
        and (metadata is None or isinstance(metadata, Metadata))
    ):
        return
    raise MessageInitializationError()


def get_message_to_descendant_id_mapping(message: Message) -> dict[str, ObjectIDs]:
    """Construct a mapping between message object_id and that of its descendants."""
    return {
        message.object_id: ObjectIDs(
            object_ids=list(get_descendant_object_ids(message))
        )
    }
