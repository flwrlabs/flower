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
"""Error."""


from __future__ import annotations

from typing import cast

MESSAGE_INIT_ERROR_MESSAGE = (
    "Invalid arguments for Message. Expected one of the documented "
    "signatures: Message(content: RecordDict, dst_node_id: int, message_type: str,"
    " *, [ttl: float, group_id: str]) or Message(content: RecordDict | error: Error,"
    " *, reply_to: Message, [ttl: float])."
)


class Error:
    """The class storing information about an error that occurred.

    Parameters
    ----------
    code : int
        An identifier for the error.
    reason : Optional[str]
        A reason for why the error arose (e.g. an exception stack-trace)
    """

    def __init__(self, code: int, reason: str | None = None) -> None:
        var_dict = {
            "_code": code,
            "_reason": reason,
        }
        self.__dict__.update(var_dict)

    @property
    def code(self) -> int:
        """Error code."""
        return cast(int, self.__dict__["_code"])

    @property
    def reason(self) -> str | None:
        """Reason reported about the error."""
        return cast(str | None, self.__dict__["_reason"])

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join([f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__qualname__}({view})"

    def __eq__(self, other: object) -> bool:
        """Compare two instances of the class."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.__dict__ == other.__dict__
