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
"""ArrayRecord."""


from __future__ import annotations

import gc
import json
import sys
from collections import OrderedDict
from logging import WARN
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_object_body,
)

from ..constant import GC_THRESHOLD
from ..logger import log
from ..typing import NDArray
from .array import Array
from .typeddict import TypedDict

if TYPE_CHECKING:
    import torch


def _raise_array_record_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {ArrayRecord.__qualname__}. Expected either "
        "a list of NumPy ndarrays, a PyTorch state_dict, or a dictionary of Arrays. "
        "The `keep_input` argument is keyword-only."
    )


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: Array) -> None:
    if not isinstance(value, Array):
        raise TypeError(
            f"Value must be of type `{Array}` but `{type(value)}` was passed."
        )


class ArrayRecord(TypedDict[str, Array], InflatableObject):
    """Array record.

    A typed dictionary (``str`` to :class:`Array`) that can store named arrays,
    including model parameters, gradients, embeddings or non-parameter arrays.
    Internally, this behaves similarly to an ``dict[str, Array]``.
    An ``ArrayRecord`` can be viewed as an equivalent to PyTorch's ``state_dict``,
    but it holds arrays in a serialized form.

    This object is one of the record types supported by :class:`RecordDict` and can
    therefore be stored in the ``content`` of a :class:`Message` or the ``state``
    of a :class:`Context`.

    This class can be instantiated in multiple ways:

    1. By providing nothing (empty container).
    2. By providing a dictionary of :class:`Array` (via the ``array_dict`` argument).
    3. By providing a list of NumPy ``ndarray`` (via the ``numpy_ndarrays`` argument).
    4. By providing a PyTorch ``state_dict`` (via the ``torch_state_dict`` argument).

    Parameters
    ----------
    array_dict : Optional[dict[str, Array]] (default: None)
        An existing dictionary containing named :class:`Array` instances. If
        provided, these entries will be used directly to populate the record.
    numpy_ndarrays : Optional[list[NDArray]] (default: None)
        A list of NumPy arrays. Each array will be automatically converted
        into an :class:`Array` and stored in this record with generated keys.
    torch_state_dict : Optional[dict[str, torch.Tensor]] (default: None)
        A PyTorch ``state_dict`` (``str`` keys to ``torch.Tensor`` values). Each
        tensor will be converted into an :class:`Array` and stored in this record.
    keep_input : bool (default: True)
        If ``False``, entries from the input are removed after being added to
        this record to free up memory. If ``True``, the input remains unchanged.
        Regardless of this value, no duplicate memory is used if the input is a
        dictionary of :class:`Array`, i.e., ``array_dict``.

    Examples
    --------
    Initializing an empty ArrayRecord::

        record = ArrayRecord()

    Initializing with a dictionary of :class:`Array`::

        arr = Array("float32", [5, 5], "numpy.ndarray", b"serialized_data...")
        record = ArrayRecord({"weight": arr})

    Initializing with a list of NumPy arrays::

        import numpy as np
        arr1 = np.random.randn(3, 3)
        arr2 = np.random.randn(2, 2)
        record = ArrayRecord([arr1, arr2])

    Initializing with a PyTorch model state_dict::

        import torch.nn as nn
        model = nn.Linear(10, 5)
        record = ArrayRecord(model.state_dict())

    Initializing with a TensorFlow model weights (a list of NumPy arrays)::

        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
        record = ArrayRecord(model.get_weights())
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self, array_dict: dict[str, Array], *, keep_input: bool = True
    ) -> None: ...

    @overload
    def __init__(
        self, numpy_ndarrays: list[NDArray], *, keep_input: bool = True
    ) -> None: ...

    @overload
    def __init__(
        self,
        # `Any` is required for PyTorch state dict because they are not strongly typed
        torch_state_dict: dict[str, torch.Tensor] | dict[str, Any],
        *,
        keep_input: bool = True,
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        numpy_ndarrays: list[NDArray] | None = None,
        torch_state_dict: dict[str, torch.Tensor] | dict[str, Any] | None = None,
        array_dict: dict[str, Array] | None = None,
        keep_input: bool = True,
    ) -> None:
        super().__init__(_check_key, _check_value)

        # Determine the initialization method and validates input arguments.
        # Support the following initialization formats:
        # 1. cls(array_dict: dict[str, Array], keep_input: bool)
        # 2. cls(numpy_ndarrays: list[NDArray], keep_input: bool)
        # 3. cls(torch_state_dict: dict[str, torch.Tensor], keep_input: bool)

        # Init the argument
        if len(args) > 1:
            _raise_array_record_init_error()
        arg = args[0] if args else None
        init_method: str | None = None  # Track which init method is being used

        # Try to assign a value to arg if it's not already set.
        # If an initialization method is provided, update init_method.
        def _try_set_arg(_arg: Any, method: str) -> None:
            # Skip if _arg is None
            if _arg is None:
                return
            nonlocal arg, init_method
            # Raise an error if arg is already set
            if arg is not None:
                _raise_array_record_init_error()
            # Raise an error if a different initialization method is already set
            if init_method is not None:
                _raise_array_record_init_error()
            # Set init_method and arg
            if init_method is None:
                init_method = method
            arg = _arg

        # Try to set keyword arguments
        _try_set_arg(array_dict, "array_dict")
        _try_set_arg(numpy_ndarrays, "numpy_ndarrays")
        _try_set_arg(torch_state_dict, "state_dict")

        # If no arguments are provided, return and keep self empty
        if arg is None:
            return

        # Handle dictionary of Arrays
        if not init_method or init_method == "array_dict":
            # Type check the input
            if (
                isinstance(arg, dict)
                and all(isinstance(k, str) for k in arg.keys())
                and all(isinstance(v, Array) for v in arg.values())
            ):
                array_dict = cast(dict[str, Array], arg)
                converted = self.from_array_dict(array_dict, keep_input=keep_input)
                self.__dict__.update(converted.__dict__)
                return

        # Handle NumPy ndarrays
        if not init_method or init_method == "numpy_ndarrays":
            # Type check the input
            # pylint: disable-next=not-an-iterable
            if isinstance(arg, list) and all(isinstance(v, np.ndarray) for v in arg):
                numpy_ndarrays = cast(list[NDArray], arg)
                converted = self.from_numpy_ndarrays(
                    numpy_ndarrays, keep_input=keep_input
                )
                self.__dict__.update(converted.__dict__)
                return

        # Handle PyTorch state_dict
        if not init_method or init_method == "state_dict":
            # Type check the input
            if (
                (torch := sys.modules.get("torch")) is not None
                and isinstance(arg, dict)
                and all(isinstance(k, str) for k in arg.keys())
                and all(isinstance(v, torch.Tensor) for v in arg.values())
            ):
                torch_state_dict = cast(dict[str, torch.Tensor], arg)  # type: ignore
                converted = self.from_torch_state_dict(
                    torch_state_dict, keep_input=keep_input
                )
                self.__dict__.update(converted.__dict__)
                return

        _raise_array_record_init_error()

    @classmethod
    def from_array_dict(
        cls,
        array_dict: dict[str, Array],
        *,
        keep_input: bool = True,
    ) -> ArrayRecord:
        """Create ArrayRecord from a dictionary of :class:`Array`."""
        record = ArrayRecord()
        for k, v in array_dict.items():
            record[k] = Array(
                dtype=v.dtype, shape=tuple(v.shape), stype=v.stype, data=v.data
            )
        if not keep_input:
            array_dict.clear()
        return record

    @classmethod
    def from_numpy_ndarrays(
        cls,
        ndarrays: list[NDArray],
        *,
        keep_input: bool = True,
    ) -> ArrayRecord:
        """Create ArrayRecord from a list of NumPy ``ndarray``."""
        record = ArrayRecord()
        total_serialized_bytes = 0

        for i in range(len(ndarrays)):  # pylint: disable=C0200
            record[str(i)] = Array.from_numpy_ndarray(ndarrays[i])

            if not keep_input:
                # Remove the reference
                ndarrays[i] = None  # type: ignore
                total_serialized_bytes += len(record[str(i)].data)

                # If total serialized data exceeds the threshold, trigger GC
                if total_serialized_bytes > GC_THRESHOLD:
                    total_serialized_bytes = 0
                    gc.collect()

        if not keep_input:
            # Clear the entire list to remove all references and force GC
            ndarrays.clear()
            gc.collect()
        return record

    @classmethod
    def from_torch_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        keep_input: bool = True,
    ) -> ArrayRecord:
        """Create ArrayRecord from PyTorch ``state_dict``."""
        if "torch" not in sys.modules:
            raise RuntimeError(
                f"PyTorch is required to use {cls.from_torch_state_dict.__name__}"
            )

        record = ArrayRecord()

        for k in list(state_dict.keys()):
            v = state_dict[k] if keep_input else state_dict.pop(k)
            record[k] = Array.from_numpy_ndarray(v.detach().cpu().numpy())

        return record

    def to_numpy_ndarrays(self, *, keep_input: bool = True) -> list[NDArray]:
        """Return the ArrayRecord as a list of NumPy ``ndarray``."""
        if keep_input:
            return [v.numpy() for v in self.values()]

        # Clear the record and return the list of NumPy arrays
        ret: list[NDArray] = []
        total_serialized_bytes = 0
        for k in list(self.keys()):
            arr = self.pop(k)
            ret.append(arr.numpy())
            total_serialized_bytes += len(arr.data)
            del arr

            # If total serialized data exceeds the threshold, trigger GC
            if total_serialized_bytes > GC_THRESHOLD:
                total_serialized_bytes = 0
                gc.collect()

        if not keep_input:
            # Force GC
            gc.collect()
        return ret

    def to_torch_state_dict(
        self, *, keep_input: bool = True
    ) -> OrderedDict[str, torch.Tensor]:
        """Return the ArrayRecord as a PyTorch ``state_dict``."""
        if not (torch := sys.modules.get("torch")):
            raise RuntimeError(
                f"PyTorch is required to use {self.to_torch_state_dict.__name__}"
            )

        state_dict = OrderedDict()

        for k in list(self.keys()):
            arr = self[k] if keep_input else self.pop(k)
            state_dict[k] = torch.from_numpy(arr.numpy())

        return state_dict

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this object.

        Note that a small amount of Bytes might also be included in this counting that
        correspond to metadata of the serialized object (e.g. of NumPy array) needed for
        deseralization.
        """
        num_bytes = 0

        for k, v in self.items():
            num_bytes += len(v.data)

            # We also count the bytes footprint of the keys
            num_bytes += len(k)

        return num_bytes

    @property
    def children(self) -> dict[str, InflatableObject]:
        """Return a dictionary of Arrays with their Object IDs as keys."""
        return {arr.object_id: arr for arr in self.values()}

    def deflate(self) -> bytes:
        """Deflate the ArrayRecord."""
        # array_name: array_object_id mapping
        array_refs: dict[str, str] = {}

        for array_name, array in self.items():
            array_refs[array_name] = array.object_id

        # Serialize references dict
        object_body = json.dumps(array_refs).encode("utf-8")
        return add_header_to_object_body(object_body=object_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> ArrayRecord:
        """Inflate an ArrayRecord from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the ArrayRecord.
        children : Optional[dict[str, InflatableObject]] (default: None)
            Dictionary of children InflatableObjects mapped to their Object IDs.
            These children enable the full inflation of the ArrayRecord.

        Returns
        -------
        ArrayRecord
            The inflated ArrayRecord.
        """
        if children is None:
            children = {}

        # Inflate mapping of array_names (keys in the ArrayRecord) to Arrays' object IDs
        obj_body = get_object_body(object_content, cls)
        array_refs: dict[str, str] = json.loads(obj_body.decode(encoding="utf-8"))

        unique_arrays = set(array_refs.values())
        children_obj_ids = set(children.keys())
        if unique_arrays != children_obj_ids:
            raise ValueError(
                "Unexpected set of `children`. "
                f"Expected {unique_arrays} but got {children_obj_ids}."
            )

        # Ensure children are of type Array
        if not all(isinstance(arr, Array) for arr in children.values()):
            raise ValueError("`Children` are expected to be of type `Array`.")

        # Instantiate new ArrayRecord
        return ArrayRecord(
            {name: children[object_id] for name, object_id in array_refs.items()}
        )

    @property
    def object_id(self) -> str:
        """Get object ID."""
        ret = super().object_id
        self.is_dirty = False  # Reset dirty flag
        return ret

    @property
    def is_dirty(self) -> bool:
        """Check if the object is dirty after the last deflation."""
        if "_is_dirty" not in self.__dict__:
            self.__dict__["_is_dirty"] = True

        if not self.__dict__["_is_dirty"]:
            if any(v.is_dirty for v in self.values()):
                # If any Array is dirty, mark the record as dirty
                self.__dict__["_is_dirty"] = True
        return cast(bool, self.__dict__["_is_dirty"])

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        """Set the dirty flag."""
        self.__dict__["_is_dirty"] = value

    def __setitem__(self, key: str, value: Array) -> None:
        """Set item and mark the record as dirty."""
        self.is_dirty = True  # Mark as dirty when setting an item
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item and mark the record as dirty."""
        self.is_dirty = True  # Mark as dirty when deleting an item
        super().__delitem__(key)


class ParametersRecord(ArrayRecord):
    """Deprecated class ``ParametersRecord``, use ``ArrayRecord`` instead.

    This class exists solely for backward compatibility with legacy
    code that previously used ``ParametersRecord``. It has been renamed
    to ``ArrayRecord``.

    .. warning::
        ``ParametersRecord`` is deprecated and will be removed in a future release.
        Use ``ArrayRecord`` instead.

    Examples
    --------
    Legacy (deprecated) usage::

        from flwr.common import ParametersRecord

        record = ParametersRecord()

    Updated usage::

        from flwr.common import ArrayRecord

        record = ArrayRecord()
    """

    _warning_logged = False

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        if not ParametersRecord._warning_logged:
            ParametersRecord._warning_logged = True
            log(
                WARN,
                "The `ParametersRecord` class has been renamed to `ArrayRecord`. "
                "Support for `ParametersRecord` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(*args, **kwargs)
