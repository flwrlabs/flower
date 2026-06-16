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
"""Typed dict base class for *Records."""


from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    MutableMapping,
    ValuesView,
)
from typing import Generic, Self, TypeVar, cast

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class TypedDict(MutableMapping[K, V], Generic[K, V]):
    """Typed dictionary."""

    def __init__(
        self, check_key_fn: Callable[[K], None], check_value_fn: Callable[[V], None]
    ):
        self.__dict__["_check_key_fn"] = check_key_fn
        self.__dict__["_check_value_fn"] = check_value_fn
        self.__dict__["_data"] = {}

    def __setitem__(self, key: K, value: V) -> None:
        """Set the given key to the given value after type checking."""
        # Check the types of key and value
        cast(Callable[[K], None], self.__dict__["_check_key_fn"])(key)
        cast(Callable[[V], None], self.__dict__["_check_value_fn"])(value)

        # Set key-value pair
        cast(dict[K, V], self.__dict__["_data"])[key] = value

    def __delitem__(self, key: K) -> None:
        """Remove the item with the specified key."""
        del cast(dict[K, V], self.__dict__["_data"])[key]

    def __getitem__(self, item: K) -> V:
        """Return the value for the specified key."""
        return cast(dict[K, V], self.__dict__["_data"])[item]

    def __iter__(self) -> Iterator[K]:
        """Yield an iterator over the keys of the dictionary."""
        return iter(cast(dict[K, V], self.__dict__["_data"]))

    def __repr__(self) -> str:
        """Return a string representation of the dictionary."""
        return cast(dict[K, V], self.__dict__["_data"]).__repr__()

    def __len__(self) -> int:
        """Return the number of items in the dictionary."""
        return len(cast(dict[K, V], self.__dict__["_data"]))

    def __contains__(self, key: object) -> bool:
        """Check if the dictionary contains the specified key."""
        return key in cast(dict[K, V], self.__dict__["_data"])

    def __eq__(self, other: object) -> bool:
        """Compare this instance to another dictionary or TypedDict."""
        data = cast(dict[K, V], self.__dict__["_data"])
        if isinstance(other, TypedDict):
            other_data = cast(dict[K, V], other.__dict__["_data"])
            return data == other_data
        if isinstance(other, dict):
            return data == other
        return NotImplemented

    def keys(self) -> KeysView[K]:
        """D.keys() -> a set-like object providing a view on D's keys."""
        return cast(dict[K, V], self.__dict__["_data"]).keys()

    def values(self) -> ValuesView[V]:
        """D.values() -> an object providing a view on D's values."""
        return cast(dict[K, V], self.__dict__["_data"]).values()

    def items(self) -> ItemsView[K, V]:
        """D.items() -> a set-like object providing a view on D's items."""
        return cast(dict[K, V], self.__dict__["_data"]).items()

    def copy(self) -> Self:
        """Return a shallow copy of the dictionary."""
        # Allocate instance without going through __init__
        new = self.__class__.__new__(type(self))
        # Copy internal state
        new.__dict__["_check_key_fn"] = self.__dict__["_check_key_fn"]
        new.__dict__["_check_value_fn"] = self.__dict__["_check_value_fn"]
        new.__dict__["_data"] = cast(dict[K, V], self.__dict__["_data"]).copy()
        return new
