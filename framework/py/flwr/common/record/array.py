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
"""Array."""


from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_object_body,
    get_object_children_ids_from_object_content,
)

from ..constant import FLWR_PRIVATE_MAX_ARRAY_CHUNK_SIZE, SType
from ..typing import NDArray
from .arraychunk import ArrayChunk

if TYPE_CHECKING:
    import torch


def _raise_array_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {Array.__qualname__}. Expected either a "
        "PyTorch tensor, a NumPy ndarray, or explicit"
        " dtype/shape/stype/data values."
    )


@dataclass
class Array(InflatableObject):
    """Array type.

    A dataclass containing serialized data from an array-like or tensor-like object
    along with metadata about it. The class can be initialized in one of three ways:

    1. By specifying explicit values for `dtype`, `shape`, `stype`, and `data`.
    2. By providing a NumPy ndarray (via the `ndarray` argument).
    3. By providing a PyTorch tensor (via the `torch_tensor` argument).

    In scenarios (2)-(3), the `dtype`, `shape`, `stype`, and `data` are automatically
    derived from the input. In scenario (1), these fields must be specified manually.

    Parameters
    ----------
    dtype : Optional[str] (default: None)
        A string representing the data type of the serialized object (e.g. `"float32"`).
        Only required if you are not passing in a ndarray or a tensor.

    shape : Optional[tuple[int, ...]] (default: None)
        A tuple representing the shape of the unserialized array-like object. Only
        required if you are not passing in a ndarray or a tensor.

    stype : Optional[str] (default: None)
        A string indicating the serialization mechanism used to generate the bytes in
        `data` from an array-like or tensor-like object. Only required if you are not
        passing in a ndarray or a tensor.

    data : Optional[bytes] (default: None)
        A buffer of bytes containing the data. Only required if you are not passing in
        a ndarray or a tensor.

    ndarray : Optional[NDArray] (default: None)
        A NumPy ndarray. If provided, the `dtype`, `shape`, `stype`, and `data`
        fields are derived automatically from it.

    torch_tensor : Optional[torch.Tensor] (default: None)
        A PyTorch tensor. If provided, it will be **detached and moved to CPU**
        before conversion, and the `dtype`, `shape`, `stype`, and `data` fields
        will be derived automatically from it.

    Examples
    --------
    Initializing by specifying all fields directly::

        arr1 = Array(
            dtype="float32",
            shape=[3, 3],
            stype="numpy.ndarray",
            data=b"serialized_data...",
        )

    Initializing with a NumPy ndarray::

        import numpy as np
        arr2 = Array(np.random.randn(3, 3))

    Initializing with a PyTorch tensor::

        import torch
        arr3 = Array(torch.randn(3, 3))
    """

    dtype: str
    shape: tuple[int, ...]
    stype: str
    data: bytes

    @overload
    def __init__(
        self, dtype: str, shape: tuple[int, ...], stype: str, data: bytes
    ) -> None: ...

    @overload
    def __init__(self, ndarray: NDArray) -> None: ...

    @overload
    def __init__(self, torch_tensor: torch.Tensor) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *args: Any,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
        stype: str | None = None,
        data: bytes | None = None,
        ndarray: NDArray | None = None,
        torch_tensor: torch.Tensor | None = None,
    ) -> None:
        # Determine the initialization method and validate input arguments.
        # Support three initialization formats:
        # 1. Array(dtype: str, shape: tuple[int, ...], stype: str, data: bytes)
        # 2. Array(ndarray: NDArray)
        # 3. Array(torch_tensor: torch.Tensor)

        # Initialize all arguments
        # If more than 4 positional arguments are provided, raise an error.
        if len(args) > 4:
            _raise_array_init_error()
        all_args = [None] * 4
        for i, arg in enumerate(args):
            all_args[i] = arg
        init_method: str | None = None  # Track which init method is being used

        # Try to assign a value to all_args[index] if it's not already set.
        # If an initialization method is provided, update init_method.
        def _try_set_arg(index: int, arg: Any, method: str) -> None:
            # Skip if arg is None
            if arg is None:
                return
            # Raise an error if all_args[index] is already set
            if all_args[index] is not None:
                _raise_array_init_error()
            # Raise an error if a different initialization method is already set
            nonlocal init_method
            if init_method is not None and init_method != method:
                _raise_array_init_error()
            # Set init_method and all_args[index]
            if init_method is None:
                init_method = method
            all_args[index] = arg

        # Try to set keyword arguments in all_args
        _try_set_arg(0, dtype, "direct")
        _try_set_arg(1, shape, "direct")
        _try_set_arg(2, stype, "direct")
        _try_set_arg(3, data, "direct")
        _try_set_arg(0, ndarray, "ndarray")
        _try_set_arg(0, torch_tensor, "torch_tensor")

        # Check if all arguments are correctly set
        all_args = [arg for arg in all_args if arg is not None]

        # Handle direct field initialization
        if not init_method or init_method == "direct":
            if (
                len(all_args) == 4  # pylint: disable=too-many-boolean-expressions
                and isinstance(all_args[0], str)
                and isinstance(all_args[1], tuple)
                and all(isinstance(i, int) for i in all_args[1])
                and isinstance(all_args[2], str)
                and isinstance(all_args[3], bytes)
            ):
                self.dtype, self.shape, self.stype, self.data = all_args
                return

        # Handle NumPy array
        if not init_method or init_method == "ndarray":
            if len(all_args) == 1 and isinstance(all_args[0], np.ndarray):
                self.__dict__.update(self.from_numpy_ndarray(all_args[0]).__dict__)
                return

        # Handle PyTorch tensor
        if not init_method or init_method == "torch_tensor":
            if (
                len(all_args) == 1
                and "torch" in sys.modules
                and isinstance(all_args[0], sys.modules["torch"].Tensor)
            ):
                self.__dict__.update(self.from_torch_tensor(all_args[0]).__dict__)
                return

        _raise_array_init_error()

    @classmethod
    def from_numpy_ndarray(cls, ndarray: NDArray) -> Array:
        """Create Array from NumPy ndarray."""
        assert isinstance(
            ndarray, np.ndarray
        ), f"Expected NumPy ndarray, got {type(ndarray)}"
        buffer = BytesIO()
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(buffer, ndarray, allow_pickle=False)
        data = buffer.getvalue()
        return Array(
            dtype=str(ndarray.dtype),
            shape=tuple(ndarray.shape),
            stype=SType.NUMPY,
            data=data,
        )

    @classmethod
    def from_torch_tensor(cls, tensor: torch.Tensor) -> Array:
        """Create Array from PyTorch tensor."""
        if not (torch := sys.modules.get("torch")):
            raise RuntimeError(
                f"PyTorch is required to use {cls.from_torch_tensor.__name__}"
            )

        assert isinstance(
            tensor, torch.Tensor
        ), f"Expected PyTorch Tensor, got {type(tensor)}"
        return cls.from_numpy_ndarray(tensor.detach().cpu().numpy())

    def numpy(self) -> NDArray:
        """Return the array as a NumPy array."""
        if self.stype != SType.NUMPY:
            raise TypeError(
                f"Unsupported serialization type for numpy conversion: '{self.stype}'"
            )
        bytes_io = BytesIO(self.data)
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
        return cast(NDArray, ndarray_deserialized)

    @property
    def children(self) -> dict[str, InflatableObject]:
        """Return a dictionary of ArrayChunks with their Object IDs as keys."""
        return dict(self.slice_array())

    def slice_array(self) -> list[tuple[str, InflatableObject]]:
        """Slice Array data and construct a list of ArrayChunks."""
        # Return cached chunks if they exist
        if "_chunks" in self.__dict__:
            return cast(list[tuple[str, InflatableObject]], self.__dict__["_chunks"])

        # Chunks are not children as some of them may be identical
        chunks: list[tuple[str, InflatableObject]] = []
        # memoryview allows for zero-copy slicing
        data_view = memoryview(self.data)
        for start in range(0, len(data_view), FLWR_PRIVATE_MAX_ARRAY_CHUNK_SIZE):
            end = min(start + FLWR_PRIVATE_MAX_ARRAY_CHUNK_SIZE, len(data_view))
            ac = ArrayChunk(data_view[start:end])
            chunks.append((ac.object_id, ac))

        # Cache the chunks for future use
        self.__dict__["_chunks"] = chunks
        return chunks

    def deflate(self) -> bytes:
        """Deflate the Array."""
        array_metadata: dict[str, str | tuple[int, ...] | list[int]] = {}

        # We want to record all object_id even if repeated
        # it can happend that chunks carry the exact same data
        # for example when the array has only zeros
        children_list = self.slice_array()
        # Let's not save the entire object_id but a mapping to those
        # that will be carried in the object head
        # (replace a long object_id with a single scalar)
        unique_children = list(self.children.keys())
        arraychunk_ids = [unique_children.index(ch_id) for ch_id, _ in children_list]

        # The deflated Array carries everything but the data
        # The `arraychunk_ids` will be used during Array inflation
        # to rematerialize the data from ArrayChunk objects.
        array_metadata = {
            "dtype": self.dtype,
            "shape": self.shape,
            "stype": self.stype,
            "arraychunk_ids": arraychunk_ids,
        }

        # Serialize metadata dict
        obj_body = json.dumps(array_metadata).encode("utf-8")
        return add_header_to_object_body(object_body=obj_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> Array:
        """Inflate an Array from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the Array.

        children : Optional[dict[str, InflatableObject]] (default: None)
            Must be ``None``. ``Array`` must have child objects.
            Providing no children will raise a ``ValueError``.

        Returns
        -------
        Array
            The inflated Array.
        """
        if children is None:
            children = {}

        obj_body = get_object_body(object_content, cls)

        # Extract children IDs from head
        children_ids = get_object_children_ids_from_object_content(object_content)
        # Decode the Array body
        array_metadata: dict[str, str | tuple[int, ...] | list[int]] = json.loads(
            obj_body.decode(encoding="utf-8")
        )

        # Verify children ids in body match those passed for inflation
        chunk_ids_indices = cast(list[int], array_metadata["arraychunk_ids"])
        # Convert indices back to IDs
        chunk_ids = [children_ids[i] for i in chunk_ids_indices]
        # Check consistency
        unique_arrayschunks = set(chunk_ids)
        children_obj_ids = set(children.keys())
        if unique_arrayschunks != children_obj_ids:
            raise ValueError(
                "Unexpected set of `children`. "
                f"Expected {unique_arrayschunks} but got {children_obj_ids}."
            )

        # Materialize Array with empty data
        array = cls(
            dtype=cast(str, array_metadata["dtype"]),
            shape=cast(tuple[int], tuple(array_metadata["shape"])),
            stype=cast(str, array_metadata["stype"]),
            data=b"",
        )

        # Now inject data from chunks
        buff = bytearray()
        for ch_id in chunk_ids:
            buff += cast(ArrayChunk, children[ch_id]).data

        array.data = bytes(buff)
        return array

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
        return cast(bool, self.__dict__["_is_dirty"])

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        """Set the dirty flag."""
        self.__dict__["_is_dirty"] = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with special handling for dirty state."""
        if name in ("dtype", "shape", "stype", "data"):
            # Mark as dirty if any of the main attributes are set
            self.is_dirty = True
            # Clear cached object ID
            self.__dict__.pop("_object_id", None)
            # Clear cached chunks if data is set
            if name == "data":
                self.__dict__.pop("_chunks", None)
        super().__setattr__(name, value)
