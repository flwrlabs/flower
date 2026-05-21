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
"""(De-)serialization tests."""


import random
import string
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest

from flwr.app.user_config import UserConfig
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.message import make_message

# pylint: disable=E0611
from flwr.proto import transport_pb2 as pb2
from flwr.proto.fab_pb2 import Fab as ProtoFab
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.recorddict_pb2 import Array as ProtoArray
from flwr.proto.recorddict_pb2 import ArrayRecord as ProtoArrayRecord
from flwr.proto.recorddict_pb2 import ConfigRecord as ProtoConfigRecord
from flwr.proto.recorddict_pb2 import MetricRecord as ProtoMetricRecord
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
from flwr.proto.run_pb2 import Run as ProtoRun
from flwr.supercore.constant import RunType
from flwr.supercore.date import now

from ..app.error import Error
from ..app.metadata import Metadata

# pylint: enable=E0611
from . import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
    typing,
)
from .serde import (
    array_from_proto,
    array_record_from_proto,
    array_record_to_proto,
    array_to_proto,
    config_record_from_proto,
    config_record_to_proto,
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    metric_record_from_proto,
    metric_record_to_proto,
    recorddict_from_proto,
    recorddict_to_proto,
    run_from_proto,
    run_to_proto,
    scalar_from_proto,
    scalar_to_proto,
    status_from_proto,
    status_to_proto,
)


def test_serialisation_deserialisation() -> None:
    """Test if the np.ndarray is identical after (de-)serialization."""
    # Prepare
    scalars = [True, b"bytestr", 3.14, 9000, "Hello", (1 << 63) + 1]

    for scalar in scalars:
        # Execute
        scalar = cast(bool | bytes | float | int | str, scalar)
        serialized = scalar_to_proto(scalar)
        actual = scalar_from_proto(serialized)

        # Assert
        assert actual == scalar


def test_status_to_proto() -> None:
    """Test status message (de-)serialization."""
    # Prepare
    code_msg = pb2.Code.OK  # pylint: disable=E1101
    status_msg = pb2.Status(code=code_msg, message="Success")  # pylint: disable=E1101

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status_msg = status_to_proto(status=status)

    # Assert
    assert actual_status_msg == status_msg


def test_status_from_proto() -> None:
    """Test status message (de-)serialization."""
    # Prepare
    code_msg = pb2.Code.OK  # pylint: disable=E1101
    status_msg = pb2.Status(code=code_msg, message="Success")  # pylint: disable=E1101

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status = status_from_proto(msg=status_msg)

    # Assert
    assert actual_status == status


def test_fab_to_proto() -> None:
    """Test Fab serialization."""
    proto_fab = ProtoFab(
        hash_str="fab_test_hash",
        content=b"fab_test_content",
        verifications={"fab_test_meta": "fab_test_meta"},
    )

    py_fab = typing.Fab(
        hash_str="fab_test_hash",
        content=b"fab_test_content",
        verifications={"fab_test_meta": "fab_test_meta"},
    )

    converted_fab = fab_to_proto(py_fab)

    # Assert
    assert converted_fab == proto_fab


def test_fab_from_proto() -> None:
    """Test Fab deserialization."""
    proto_fab = ProtoFab(
        hash_str="fab_test_hash",
        content=b"fab_test_content",
        verifications={"meta_key": "meta_value"},
    )

    py_fab = typing.Fab(
        hash_str="fab_test_hash",
        content=b"fab_test_content",
        verifications={"meta_key": "meta_value"},
    )

    converted_fab = fab_from_proto(proto_fab)

    # Assert
    assert converted_fab == py_fab


T = TypeVar("T")


class RecordMaker:
    """A record maker based on a seeded random number generator."""

    def __init__(self, state: int = 42) -> None:
        self.rng = random.Random(state)

    def randbytes(self, n: int) -> bytes:
        """Create a bytes."""
        return self.rng.getrandbits(n * 8).to_bytes(n, "little")

    def get_str(self, length: int | None = None) -> str:
        """Create a string."""
        char_pool = (
            string.ascii_letters + string.digits + " !@#$%^&*()_-+=[]|;':,./<>?{}"
        )
        if length is None:
            length = self.rng.randint(1, 10)
        return "".join(self.rng.choices(char_pool, k=length))

    def get_value(self, dtype: type[T] | str) -> T:
        """Create a value of a given type."""
        ret: Any = None
        if dtype is bool:
            ret = self.rng.random() < 0.5
        elif dtype is str:
            ret = self.get_str(self.rng.randint(10, 100))
        elif dtype is int:
            ret = self.rng.randint(-1 << 63, (1 << 63) - 1)
        elif dtype is float:
            ret = (self.rng.random() - 0.5) * (2.0 ** self.rng.randint(0, 50))
        elif dtype is bytes:
            ret = self.randbytes(self.rng.randint(10, 100))
        elif dtype == "uint":
            ret = self.rng.randint(0, (1 << 64) - 1)
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")
        return cast(T, ret)

    def get_message_type(self) -> str:
        """Create a message type."""
        # Create a legacy message type
        if self.rng.random() < 0.5:
            return self.rng.choice(["get_parameters", "get_properties", "reconnect"])

        # Create a message type
        category = self.rng.choice(["train", "evaluate", "query"])
        suffix = self.rng.choice(["", ".custom_action", ".mock_action"])
        return f"{category}{suffix}"

    def array(self) -> Array:
        """Create a Array."""
        dtypes = ("float", "int")
        stypes = ("torch", "tf", "numpy")
        max_shape_size = 100
        max_shape_dim = 10
        min_max_bytes_size = (10, 1000)

        dtype = self.rng.choice(dtypes)
        shape = tuple(
            self.rng.randint(1, max_shape_size)
            for _ in range(self.rng.randint(1, max_shape_dim))
        )
        stype = self.rng.choice(stypes)
        data = self.randbytes(self.rng.randint(*min_max_bytes_size))
        return Array(dtype=dtype, shape=shape, stype=stype, data=data)

    def array_record(self) -> ArrayRecord:
        """Create a ArrayRecord."""
        num_arrays = self.rng.randint(1, 5)
        arrays = {self.get_str(): self.array() for i in range(num_arrays)}
        return ArrayRecord(arrays, keep_input=False)

    def metric_record(self) -> MetricRecord:
        """Create a MetricRecord."""
        num_entries = self.rng.randint(1, 5)
        types = (float, int)
        return MetricRecord(
            metric_dict={
                self.get_str(): self.get_value(self.rng.choice(types))
                for _ in range(num_entries)
            },
            keep_input=False,
        )

    def config_record(self) -> ConfigRecord:
        """Create a ConfigRecord."""
        num_entries = self.rng.randint(1, 5)
        types = (str, int, float, bytes, bool)
        return ConfigRecord(
            config_dict={
                self.get_str(): self.get_value(self.rng.choice(types))
                for _ in range(num_entries)
            },
            keep_input=False,
        )

    def recorddict(
        self,
        num_array_records: int,
        num_metric_records: int,
        num_config_records: int,
    ) -> RecordDict:
        """Create a RecordDict."""
        ret = RecordDict()
        for _ in range(num_array_records):
            ret[self.get_str()] = self.array_record()
        for _ in range(num_metric_records):
            ret[self.get_str()] = self.metric_record()
        for _ in range(num_config_records):
            ret[self.get_str()] = self.config_record()
        return ret

    def metadata(self) -> Metadata:
        """Create a Metadata."""
        return Metadata(
            run_id=self.rng.randint(0, 1 << 30),
            message_id=self.get_str(64),
            group_id=self.get_str(30),
            src_node_id=self.rng.randint(0, 1 << 63),
            dst_node_id=self.rng.randint(0, 1 << 63),
            reply_to_message_id=self.get_str(64),
            created_at=now().timestamp(),
            ttl=self.rng.randint(1, 1 << 30),
            message_type=self.get_message_type(),
        )

    def user_config(self) -> UserConfig:
        """Create a UserConfig."""
        return {
            "key1": self.rng.randint(0, 1 << 30),
            "key2": self.get_str(10),
            "key3": self.rng.random(),
            "key4": bool(self.rng.getrandbits(1)),
        }


def test_array_serialization_deserialization() -> None:
    """Test serialization and deserialization of Array."""
    # Prepare
    maker = RecordMaker()
    original = maker.array()

    # Execute
    proto = array_to_proto(original)
    deserialized = array_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoArray)
    assert original == deserialized


def test_array_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ArrayRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.array_record()

    # Execute
    proto = array_record_to_proto(original)
    deserialized = array_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoArrayRecord)
    assert original == deserialized


def test_metric_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of MetricRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.metric_record()
    original["uint64"] = (1 << 63) + 321
    original["list of uint64"] = [maker.get_value("uint") for _ in range(30)]

    # Execute
    proto = metric_record_to_proto(original)
    deserialized = metric_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoMetricRecord)
    assert original == deserialized


def test_config_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ConfigRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.config_record()
    original["uint64"] = (1 << 63) + 101
    original["list of uint64"] = [maker.get_value("uint") for _ in range(100)]

    # Execute
    proto = config_record_to_proto(original)
    deserialized = config_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoConfigRecord)
    assert original == deserialized


def test_recorddict_serialization_deserialization() -> None:
    """Test serialization and deserialization of RecordDict."""
    # Prepare
    maker = RecordMaker(state=0)
    original = maker.recorddict(2, 2, 1)

    # Execute
    proto = recorddict_to_proto(original)
    deserialized = recorddict_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoRecordDict)
    assert original == deserialized


def test_recorddict_key_order_stability() -> None:
    """Test that RecordDict preserves key order."""
    # Prepare
    maker = RecordMaker(state=1)
    original = maker.recorddict(10, 5, 5)

    # Execute
    proto = recorddict_to_proto(original)
    deserialized = recorddict_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoRecordDict)
    assert original == deserialized

    # Check that keys are in the same order
    assert list(original.keys()) == list(deserialized.keys())


@pytest.mark.parametrize(
    "content_fn, error_fn",
    [
        (
            lambda maker: maker.recorddict(1, 1, 1),
            None,
        ),  # check when only content is set
        (None, lambda code: Error(code=code)),  # check when only error is set
    ],
)
def test_message_serialization_deserialization(
    content_fn: Callable[
        [
            RecordMaker,
        ],
        RecordDict,
    ],
    error_fn: Callable[[int], Error],
) -> None:
    """Test serialization and deserialization of Message."""
    # Prepare
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    metadata.dst_node_id = SUPERLINK_NODE_ID  # Assume SuperLink node ID

    original = make_message(
        metadata=metadata,
        content=None if content_fn is None else content_fn(maker),
        error=None if error_fn is None else error_fn(0),
    )

    # Execute
    proto = message_to_proto(original)
    deserialized = message_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoMessage)

    if original.has_content():
        assert original.content == deserialized.content
    if original.has_error():
        assert original.error == deserialized.error

    assert original.metadata == deserialized.metadata


def test_context_serialization_deserialization() -> None:
    """Test serialization and deserialization of Context."""
    # Prepare
    maker = RecordMaker()
    original = Context(
        run_id=0,
        node_id=1,
        node_config=maker.user_config(),
        state=maker.recorddict(1, 1, 1),
        run_config=maker.user_config(),
    )

    # Execute
    proto = context_to_proto(original)
    deserialized = context_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoContext)
    assert original == deserialized


def test_run_serialization_deserialization() -> None:
    """Test serialization and deserialization of Run."""
    # Prepare
    maker = RecordMaker()
    original = typing.Run(
        run_id=1,
        fab_id="lorem",
        fab_version="ipsum",
        fab_hash="hash",
        override_config=maker.user_config(),
        pending_at="2021-01-01T00:00:00Z",
        starting_at="2021-01-02T23:02:11Z",
        running_at="2021-01-03T12:00:50Z",
        finished_at="",
        status=typing.RunStatus(status="running", sub_status="", details="OK"),
        flwr_aid="user123",
        federation="mock-fed",
        primary_task_id=42,
        bytes_sent=2048,
        bytes_recv=1024,
        clientapp_runtime=3.14,
        run_type=RunType.SIMULATION,
    )

    # Execute
    proto = run_to_proto(original)
    deserialized = run_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoRun)
    assert original == deserialized
