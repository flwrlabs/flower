# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Conversion functions between an annotated class instance and a ConfigsRecord."""

import sys
from typing import Any, List, Type, TypeVar, cast, get_type_hints

import numpy as np

from flwr.common import ConfigsRecord
from flwr.common.parameter import bytes_to_ndarray, ndarray_to_bytes

if sys.version_info >= (3, 9):
    from typing import get_args, get_origin
else:

    def get_origin(tp: Any) -> Any:
        """Get the unsubscripted version of a type."""
        return getattr(tp, "__origin__", None)

    def get_args(tp: Any) -> Any:
        """Get type arguments with all substitutions performed."""
        return getattr(tp, "__args__", ())


T = TypeVar("T")


def to_configsrecord(instance: object) -> ConfigsRecord:
    """Convert the annotated class instance to a ConfigsRecord."""
    ret = ConfigsRecord()
    type_hints = get_type_hints(instance.__class__)

    for key, value in instance.__dict__.items():
        if key not in type_hints:
            raise ValueError(f"Field '{key}' is not annotated.")

        # Handle non-list types
        if isinstance(value, (int, float, str, bytes, bool)):
            ret[key] = value
        elif isinstance(value, np.ndarray):
            ret[key] = ndarray_to_bytes(value)
        # Handle list types
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], np.ndarray):
                ret[key] = [ndarray_to_bytes(arr) for arr in value]
            else:
                ret[key] = value
        # Handle dict
        elif isinstance(value, dict):
            dict_keys = list(value.keys())
            dict_values = list(value.values())
            if len(dict_values) > 0 and isinstance(dict_values[0], np.ndarray):
                dict_values = [ndarray_to_bytes(arr) for arr in dict_values]
            ret[f"{key}:K"] = dict_keys
            ret[f"{key}:V"] = dict_values
        else:
            raise ValueError(f"Value of field '{key}' is not supported.")
    return ret


def from_configsrecord(cls: Type[T], configs: ConfigsRecord) -> T:
    """Construct an annotated class instance from a ConfigsRecord."""
    ret = cls()
    type_hints = get_type_hints(cls)

    for key, value_type in type_hints.items():
        origin = get_origin(value_type)
        args = get_args(value_type)

        if key in configs:
            # Read NDArray
            if origin is np.ndarray:
                ret.__dict__[key] = bytes_to_ndarray(cast(bytes, configs[key]))
            # Read List[NDArray]
            elif origin is list and get_origin(args[0]) is np.ndarray:
                bytes_lst = cast(List[bytes], configs[key])
                ret.__dict__[key] = [bytes_to_ndarray(b) for b in bytes_lst]
            # Read common types (ConfigsRecordValues)
            else:
                ret.__dict__[key] = configs[key]
        # Read dict
        elif f"{key}:K" in configs and f"{key}:V" in configs and origin is dict:
            dict_keys = cast(List[Any], configs[f"{key}:K"])
            dict_values = cast(List[Any], configs[f"{key}:V"])
            # Check if Dict[Any, NDArray]
            if get_origin(args[1]) is np.ndarray:
                dict_values = [bytes_to_ndarray(cast(bytes, b)) for b in dict_values]
            ret.__dict__[key] = dict(zip(dict_keys, dict_values))
        else:
            raise ValueError(f"Field '{key}' not found in configs.")

    return ret
