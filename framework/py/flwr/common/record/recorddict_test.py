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
"""RecordDict tests."""


import json
import pickle
from collections.abc import Callable
from copy import deepcopy
from typing import cast
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.recorddict_compat import (
    arrayrecord_to_parameters,
    parameters_to_arrayrecord,
)
from flwr.common.typing import (
    ConfigRecordValues,
    MetricRecordValues,
    NDArray,
    NDArrays,
    Parameters,
)
from flwr.supercore.inflatable.inflatable_object import (
    get_object_body,
    get_object_type_from_object_content,
)

from ..serde import config_record_to_proto, metric_record_to_proto
from . import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

# pylint: disable=E0611


def get_ndarrays() -> NDArrays:
    """Return list of NumPy arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4], [5.0, 6.0]])
    arr2 = np.eye(2, 7, 3)

    return [arr1, arr2]


def ndarray_to_array(ndarray: NDArray) -> Array:
    """Represent NumPy ndarray as Array."""
    return Array(
        data=ndarray.tobytes(),
        dtype=str(ndarray.dtype),
        stype="numpy.ndarray.tobytes",
        shape=tuple(ndarray.shape),
    )


def test_ndarray_to_array() -> None:
    """Test creation of Array object from NumPy ndarray."""
    shape = (2, 7, 9)
    arr = np.eye(*shape)

    array = ndarray_to_array(arr)

    arr_ = np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)

    assert np.array_equal(arr, arr_)


def test_parameters_to_array_and_back() -> None:
    """Test conversion between legacy Parameters and Array."""
    ndarrays = get_ndarrays()

    # Array represents a single array, unlike parameters, which represent a
    # list of arrays
    ndarray = ndarrays[0]

    parameters = ndarrays_to_parameters([ndarray])

    array = Array(
        data=parameters.tensors[0], dtype="", stype=parameters.tensor_type, shape=()
    )

    parameters = Parameters(tensors=[array.data], tensor_type=array.stype)

    ndarray_ = parameters_to_ndarrays(parameters=parameters)[0]

    assert np.array_equal(ndarray, ndarray_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (False, lambda x, x_copy, y: len(x.tensors) == 0),  # check tensors were freed
        (True, lambda x, x_copy, y: x.tensors == y.tensors),  # check they are equal
    ],
)
def test_parameters_to_arrayrecord_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[[Parameters, Parameters, Parameters], bool],
) -> None:
    """Test conversion between legacy Parameters and ArrayRecords."""
    ndarrays = get_ndarrays()

    parameters = ndarrays_to_parameters(ndarrays)
    parameters_copy = deepcopy(parameters)

    arr_record = parameters_to_arrayrecord(parameters=parameters, keep_input=keep_input)

    parameters_ = arrayrecord_to_parameters(arr_record, keep_input=keep_input)

    ndarrays_ = parameters_to_ndarrays(parameters=parameters_)

    # Validate returned NDArrays match those at the beginning
    for arr, arr_ in zip(ndarrays, ndarrays_, strict=True):
        assert np.array_equal(arr, arr_), "no"

    # Validate initial Parameters object has been handled according to `keep_input`
    assert validate_freed_fn(parameters, parameters_copy, parameters_)


def test_set_parameters_while_keeping_intputs() -> None:
    """Test keep_input functionality in ArrayRecord."""
    # Adding parameters to a record that doesn't erase entries in the input `array_dict`
    array_dict = {
        str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())
    }
    arr_record = ArrayRecord(array_dict, keep_input=True)

    # Creating a second ArrayRecord passing the same `array_dict` (not erased)
    arr_record_2 = ArrayRecord(array_dict, keep_input=False)
    assert arr_record == arr_record_2

    # Now it should be empty (the second ArrayRecord wasn't flagged to keep it)
    assert len(array_dict) == 0


def test_set_parameters_with_correct_types() -> None:
    """Test adding dictionary of Arrays to ArrayRecord."""
    arr_record = ArrayRecord()
    array_dict = {
        str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())
    }
    arr_record.update(array_dict)


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: x),  # correct key, incorrect value
        (str, lambda x: x.tolist()),  # correct key, incorrect value
        (int, ndarray_to_array),  # incorrect key, correct value
        (int, lambda x: x),  # incorrect key, incorrect value
        (int, lambda x: x.tolist()),  # incorrect key, incorrect value
    ],
)
def test_set_parameters_with_incorrect_types(
    key_type: type[int | str],
    value_fn: Callable[[NDArray], NDArray | list[float]],
) -> None:
    """Test adding dictionary of unsupported types to ArrayRecord."""
    arr_record = ArrayRecord()

    array_dict = {
        key_type(i): value_fn(ndarray) for i, ndarray in enumerate(get_ndarrays())
    }

    with pytest.raises(TypeError):
        arr_record.update(array_dict)  # type: ignore


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: int(x.flatten()[0])),  # str: int
        (str, lambda x: float(x.flatten()[0])),  # str: float
        (str, lambda x: x.flatten().astype("int").tolist()),  # str: List[int]
        (str, lambda x: x.flatten().astype("float").tolist()),  # str: List[float]
        (str, lambda x: []),  # str: empty list
    ],
)
def test_set_metrics_to_metricrecord_with_correct_types(
    key_type: type[str],
    value_fn: Callable[[NDArray], MetricRecordValues],
) -> None:
    """Test adding metrics of various types to a MetricRecord."""
    m_record = MetricRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_metrics = {
        key_type(label): value_fn(arr)
        for label, arr in zip(labels, arrays, strict=True)
    }

    # Add metric
    m_record.update(my_metrics)

    # Check metrics are actually added
    assert my_metrics == m_record


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: str(x.flatten()[0])),  # str: str  (supported: unsupported)
        (str, lambda x: bool(x.flatten()[0])),  # str: bool  (supported: unsupported)
        (
            str,
            lambda x: x.flatten().astype("str").tolist(),
        ),  # str: List[str] (supported: unsupported)
        (str, lambda x: x),  # str: NDArray (supported: unsupported)
        (
            str,
            lambda x: {str(v): v for v in x.flatten()},
        ),  # str: dict[str: float] (supported: unsupported)
        (
            str,
            lambda x: [{str(v): v for v in x.flatten()}],
        ),  # str: List[dict[str: float]] (supported: unsupported)
        (
            str,
            lambda x: [1, 2.0, 3.0, 4],
        ),  # str: List[mixing valid types] (supported: unsupported)
        (
            int,
            lambda x: x.flatten().tolist(),
        ),  # int: List[str] (unsupported: supported)
        (
            float,
            lambda x: x.flatten().tolist(),
        ),  # float: List[int] (unsupported: supported)
    ],
)
def test_set_metrics_to_metricrecord_with_incorrect_types(
    key_type: type[str | int | float | bool],
    value_fn: Callable[[NDArray], NDArray | dict[str, NDArray] | list[float]],
) -> None:
    """Test adding metrics of various unsupported types to a MetricRecord."""
    m_record = MetricRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_metrics = {
        key_type(label): value_fn(arr)
        for label, arr in zip(labels, arrays, strict=True)
    }

    with pytest.raises(TypeError):
        m_record.update(my_metrics)  # type: ignore


@pytest.mark.parametrize(
    "keep_input",
    [
        (True),
        (False),
    ],
)
def test_set_metrics_to_metricrecord_with_and_without_keeping_input(
    keep_input: bool,
) -> None:
    """Test keep_input functionality for MetricRecord."""
    # constructing a valid input
    labels = [1, 2.0]
    arrays = get_ndarrays()
    my_metrics = cast(
        dict[str, MetricRecordValues],
        {
            str(label): arr.flatten().tolist()
            for label, arr in zip(labels, arrays, strict=True)
        },
    )
    my_metrics_copy = my_metrics.copy()

    # Add metric
    m_record = MetricRecord(my_metrics, keep_input=keep_input)

    # Check metrics are actually added
    # Check that input dict has been emptied when enabled such behaviour
    if keep_input:
        assert my_metrics == m_record
    else:
        assert my_metrics_copy == m_record
        assert len(my_metrics) == 0


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: str(x.flatten()[0])),  # str: str
        (str, lambda x: int(x.flatten()[0])),  # str: int
        (str, lambda x: float(x.flatten()[0])),  # str: float
        (str, lambda x: bool(x.flatten()[0])),  # str: bool
        (str, lambda x: x.flatten().tobytes()),  # str: bytes
        (str, lambda x: x.flatten().astype("str").tolist()),  # str: List[str]
        (str, lambda x: x.flatten().astype("int").tolist()),  # str: List[int]
        (str, lambda x: x.flatten().astype("float").tolist()),  # str: List[float]
        (str, lambda x: x.flatten().astype("bool").tolist()),  # str: List[bool]
        (str, lambda x: [x.flatten().tobytes()]),  # str: List[bytes]
        (str, lambda x: []),  # str: emptyt list
    ],
)
def test_set_configs_to_configrecord_with_correct_types(
    key_type: type[str],
    value_fn: Callable[[NDArray], ConfigRecordValues],
) -> None:
    """Test adding configs of various types to a ConfigRecord."""
    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_configs = {
        key_type(label): value_fn(arr)
        for label, arr in zip(labels, arrays, strict=True)
    }
    c_record = ConfigRecord(my_configs)

    # check values are actually there
    assert c_record == my_configs


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: x),  # str: NDArray (supported: unsupported)
        (
            str,
            lambda x: {str(v): v for v in x.flatten()},
        ),  # str: dict[str: float] (supported: unsupported)
        (
            str,
            lambda x: [{str(v): v for v in x.flatten()}],
        ),  # str: List[dict[str: float]] (supported: unsupported)
        (
            str,
            lambda x: [1, 2.0, 3.0, 4],
        ),  # str: List[mixing valid types] (supported: unsupported)
        (
            int,
            lambda x: x.flatten().tolist(),
        ),  # int: List[str] (unsupported: supported)
        (
            float,
            lambda x: x.flatten().tolist(),
        ),  # float: List[int] (unsupported: supported)
    ],
)
def test_set_configs_to_configrecord_with_incorrect_types(
    key_type: type[str | int | float],
    value_fn: Callable[[NDArray], NDArray | dict[str, NDArray] | list[float]],
) -> None:
    """Test adding configs of various unsupported types to a ConfigRecord."""
    c_record = ConfigRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_configs = {
        key_type(label): value_fn(arr)
        for label, arr in zip(labels, arrays, strict=True)
    }
    with pytest.raises(TypeError):
        c_record.update(my_configs)  # type: ignore


def test_count_bytes_metricrecord() -> None:
    """Test counting bytes in MetricRecord."""
    data = {"a": 1, "b": 2.0, "c": [1, 2, 3], "d": [1.0, 2.0, 3.0, 4.0, 5.0]}
    bytes_in_dict = 8 + 8 + 3 * 8 + 5 * 8
    bytes_in_dict += 4  # represnting the keys

    m_record = MetricRecord()
    m_record.update(data)  # type: ignore
    record_bytest_count = m_record.count_bytes()
    assert bytes_in_dict == record_bytest_count


def test_count_bytes_configrecord() -> None:
    """Test counting bytes in ConfigRecord."""
    data = {"a": 1, "b": 2.0, "c": [1, 2, 3], "d": [1.0, 2.0, 3.0, 4.0, 5.0]}
    bytes_in_dict = 8 + 8 + 3 * 8 + 5 * 8
    bytes_in_dict += 4  # represnting the keys

    to_add = {
        "aa": True,
        "bb": "False",
        "cc": bytes(9),
        "dd": [True, False, False],
        "ee": ["True", "False"],
        "ff": [bytes(1), bytes(13), bytes(51)],
    }
    data = {**data, **to_add}
    bytes_in_dict += 1 + 5 + 9 + 3 + (4 + 5) + (1 + 13 + 51)
    bytes_in_dict += 12  # represnting the keys

    bytes_in_dict = int(bytes_in_dict)

    c_record = ConfigRecord()
    c_record.update(data)  # type: ignore

    record_bytest_count = c_record.count_bytes()
    assert bytes_in_dict == record_bytest_count


def test_record_is_picklable() -> None:
    """Test if RecordDict and *Record are picklable."""
    # Prepare
    arr_record = ArrayRecord()
    m_record = MetricRecord({"aa": 123})
    c_record = ConfigRecord({"cc": bytes(9)})
    rs = RecordDict()
    rs.array_records["arrays"] = arr_record
    rs.metric_records["metrics"] = m_record
    rs.config_records["configs"] = c_record

    # Execute
    pickle.dumps((arr_record, m_record, c_record, rs))


def test_recorddict_repr() -> None:
    """Test the string representation of RecordDict."""
    # Prepare
    rs = RecordDict(
        {
            "arrays": ArrayRecord(),
            "metrics": MetricRecord({"aa": 123}),
            "configs": ConfigRecord({"cc": bytes(5)}),
        },
    )
    expected = """RecordDict(
  array_records={'arrays': {}},
  metric_records={'metrics': {'aa': 123}},
  config_records={'configs': {'cc': b'\\x00\\x00\\x00\\x00\\x00'}}
)"""

    # Assert
    assert str(rs) == expected


def test_recorddict_set_get_del_item() -> None:
    """Test setting, getting, and deleting items in RecordDict."""
    # Prepare
    rs = RecordDict()
    arr_record = ArrayRecord()
    m_record = MetricRecord({"aa": 123})
    c_record = ConfigRecord({"cc": bytes(5)})

    # Execute
    rs["arrays"] = arr_record
    rs["metrics"] = m_record
    rs["configs"] = c_record

    # Assert
    assert rs["arrays"] == arr_record
    assert rs["metrics"] == m_record
    assert rs["configs"] == c_record

    # Execute
    del rs["arrays"]
    del rs["metrics"]
    del rs["configs"]

    # Assert
    assert "arrays" not in rs
    assert "metrics" not in rs
    assert "configs" not in rs


def test_constructor_with_deprecated_arguments() -> None:
    """Test constructor with deprecated arguments."""
    # Prepare
    array_rec = ArrayRecord({"weights": Array("mock", (2, 3), "mock", b"123")})
    metric_rec = MetricRecord({"accuracy": 0.95})
    config_rec = ConfigRecord({"lr": 0.01})

    # Execute
    rd = RecordDict(
        parameters_records={"param": array_rec},
        metrics_records={"metric": metric_rec},
        configs_records={"config": config_rec},
    )

    # Assert
    assert rd["param"] == array_rec
    assert rd["metric"] == metric_rec
    assert rd["config"] == config_rec


def test_parameters_records_delegation_and_return() -> None:
    """Test parameters_records property delegates to array_records."""
    # Prepare
    rd = RecordDict()

    # Execute and assert
    with patch.object(
        RecordDict, "array_records", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = Mock(name="array_records_return")

        result = rd.parameters_records

        mock_property.assert_called_once()
        assert result is mock_property.return_value


def test_metrics_records_delegation_and_return() -> None:
    """Test metrics_records property delegates to metric_records."""
    # Prepare
    rd = RecordDict()

    # Execute and assert
    with patch.object(
        RecordDict, "metric_records", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = Mock(name="metric_records_return")

        result = rd.metrics_records

        mock_property.assert_called_once()
        assert result is mock_property.return_value


def test_configs_records_delegation_and_return() -> None:
    """Test configs_records property delegates to config_records."""
    # Prepare
    rd = RecordDict()

    # Execute and assert
    with patch.object(
        RecordDict, "config_records", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = Mock(name="config_records_return")

        result = rd.configs_records

        mock_property.assert_called_once()
        assert result is mock_property.return_value


@pytest.mark.parametrize(
    "record_type, record_data, proto_conversion_fn",
    [
        (
            MetricRecord,
            {"a": 123, "b": [0.123, 0.456]},
            lambda x: metric_record_to_proto(x).SerializeToString(deterministic=True),
        ),
        (
            ConfigRecord,
            {
                "a": 123,
                "b": [0.123, 0.456],
                "data": b"hello world",
            },
            lambda x: config_record_to_proto(x).SerializeToString(deterministic=True),
        ),
    ],
)
def test_metric_and_config_record_deflate_and_inflate(
    record_type: type[ConfigRecord | MetricRecord],
    record_data: dict[str, ConfigRecordValues | MetricRecordValues],
    proto_conversion_fn: Callable[[ConfigRecord | MetricRecord], bytes],
) -> None:
    """Ensure an MetricRecord and ConfigRecord can be (de)inflated correctly."""
    record = record_type(record_data)  # type: ignore[arg-type]

    # Assert
    # Record has no children
    assert record.children is None

    record_b = record.deflate()

    # Assert
    # Class name matches
    assert (
        get_object_type_from_object_content(record_b) == record.__class__.__qualname__
    )
    # Body of deflfated Array matches its direct protobuf serialization
    assert get_object_body(record_b, record_type) == proto_conversion_fn(record)

    # Inflate
    record_ = record_type.inflate(record_b)

    # Assert
    # Both objects are identical
    assert record.object_id == record_.object_id

    # Assert
    # Inflate passing children raises ValueError
    with pytest.raises(ValueError):
        record_type.inflate(record_b, children={"1234": record})


@pytest.mark.parametrize(
    "records",
    [
        (
            {
                "m": MetricRecord({"a": 123, "b": [0.123, 0.456]}),
                "c": ConfigRecord(
                    {
                        "a": 123,
                        "b": [0.123, 0.456],
                        "data": b"hello world",
                    }
                ),
                "a": ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
            }  # All types of supported records
        ),
        ({}),  # No records
        (
            {
                "a": MetricRecord({"a": 123, "b": [0.123, 0.456]}),
                "b": MetricRecord({"a": 123, "b": [0.123, 0.456]}),
            }
        ),  # Identical records under different key
    ],
)
def test_recorddict_deflate_and_inflate(
    records: dict[str, ConfigRecord | MetricRecord | ArrayRecord],
) -> None:
    """Test that a RecordDict can be (de)inflated correctly."""
    record = RecordDict(records)

    # Assert
    # Expected children
    assert record.children == {rec.object_id: rec for rec in records.values()}

    record_b = record.deflate()

    # Assert
    # Class name matches
    assert (
        get_object_type_from_object_content(record_b) == record.__class__.__qualname__
    )
    # Body of deflfated RecordDict matches its direct protobuf serialization
    record_refs = {name: rec.object_id for name, rec in record.items()}
    record_refs_enc = json.dumps(record_refs).encode("utf-8")
    assert get_object_body(record_b, RecordDict) == record_refs_enc

    # Inflate
    record_ = RecordDict.inflate(record_b, record.children)

    # Assert
    # Both objects are identical
    assert record.object_id == record_.object_id


def test_recorddict_raises_value_error_with_unsupported_children() -> None:
    """Test that inflating a RecordDict raises a ValueError with unsupported
    Children."""
    record_dict = RecordDict({"a": ConfigRecord()})
    record_dict_b = record_dict.deflate()

    # Assert
    # Inflate but passing unexpected number of children (but of correct type)
    with pytest.raises(ValueError):
        ArrayRecord.inflate(
            record_dict_b, children={"123": ConfigRecord(), "456": MetricRecord()}
        )
    # Inflate but passing no children
    with pytest.raises(ValueError):
        ArrayRecord.inflate(record_dict_b)
    # Inflate but passing unsupported children type
    with pytest.raises(ValueError):
        ArrayRecord.inflate(record_dict_b, children={"123": RecordDict()})
    # Inflate but passing expected children value but under the wrong key
    with pytest.raises(ValueError):
        ArrayRecord.inflate(record_dict_b, children={"123": ConfigRecord()})


def test_copy_recorddict() -> None:
    """Test copying a RecordDict."""
    # Prepare
    original = RecordDict(
        {
            "m": MetricRecord({"a": 123, "b": [0.123, 0.456]}),
            "c": ConfigRecord({"data": b"hello world"}),
            "a": ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
        }
    )

    # Execute
    copy = original.copy()

    # Assert
    assert isinstance(copy, RecordDict)
    assert list(original.items()) == list(copy.items())
    original.clear()
    assert len(original) == 0
    assert len(copy) == 3


@pytest.mark.parametrize(
    "original",
    [
        MetricRecord({"a": 123, "b": [0.123, 0.456]}),
        ConfigRecord(
            {
                "a": 123,
                "b": [0.123, 0.456],
                "data": b"hello world",
            }
        ),
        ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
    ],
)
def test_copy_record(original: ConfigRecord | MetricRecord | ArrayRecord) -> None:
    """Test copying a Record."""
    # Execute
    copy = original.copy()

    # Assert
    assert isinstance(copy, original.__class__)
    assert list(original.items()) == list(copy.items())
    original.clear()
    assert len(original) == 0
    assert len(copy) != 0
