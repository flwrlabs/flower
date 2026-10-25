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
"""Tests for message-based strategy utilities."""


from unittest.mock import Mock

import numpy as np
import pytest
from parameterized import parameterized

from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.exception import InconsistentMessageReplies

from .strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    config_to_str,
    validate_message_reply_consistency,
)


def create_mock_reply(arrays: ArrayRecord, num_examples: float) -> Message:
    """Create a mock reply Message with default keys."""
    message = Mock(spec=Message)
    message.content = RecordDict(
        {"arrays": arrays, "metrics": MetricRecord({"num-examples": num_examples})}
    )
    message.has_error.side_effect = lambda: False
    message.has_content.side_effect = lambda: True
    return message


def test_config_to_str() -> None:
    """Test that items of types bytes are masked out."""
    config = ConfigRecord({"a": 123, "b": [1, 2, 3], "c": b"bytes"})
    expected_str = "{'a': 123, 'b': [1, 2, 3], 'c': <bytes>}"
    assert config_to_str(config) == expected_str


def test_arrayrecords_aggregation() -> None:
    """Test aggregation of ArrayRecords."""
    num_replies = 3
    num_arrays = 4
    weights = [0.25, 0.4, 0.35]
    np_arrays = [
        [np.random.randn(7, 3) for _ in range(num_arrays)] for _ in range(num_replies)
    ]

    avg_list = [
        np.average([lst[i] for lst in np_arrays], axis=0, weights=weights)
        for i in range(num_arrays)
    ]

    # Construct RecordDicts (mimicing replies)
    records = [
        RecordDict(
            {
                "arrays": ArrayRecord(np_arrays[i]),
                "metrics": MetricRecord({"weight": weights[i]}),
            }
        )
        for i in range(num_replies)
    ]
    # Execute aggregate
    aggrd = aggregate_arrayrecords(records, weighting_metric_name="weight")

    # Assert consistency
    assert all(
        np.allclose(a, b)
        for a, b in zip(aggrd.to_numpy_ndarrays(), avg_list, strict=True)
    )
    assert aggrd.object_id == ArrayRecord(avg_list).object_id


def test_arrayrecords_aggregation_with_ndim_zero() -> None:
    """Test aggregation of ArrayRecords with 0-dim arrays."""
    num_replies = 3
    weights = [0.25, 0.4, 0.35]
    np_arrays = [np.array(np.random.randn()) for _ in range(num_replies)]

    # For 0-dimensional arrays, we just compute the weighted average directly
    avg_list = [np.average(np_arrays, axis=0, weights=weights)]

    # Construct RecordDicts (mimicing replies)
    records = [
        RecordDict(
            {
                "arrays": ArrayRecord([np_arrays[i]]),
                "metrics": MetricRecord({"weight": weights[i]}),
            }
        )
        for i in range(num_replies)
    ]
    # Execute aggregate
    aggrd = aggregate_arrayrecords(records, weighting_metric_name="weight")

    # Assert consistency
    assert np.isclose(aggrd.to_numpy_ndarrays()[0], avg_list[0])
    assert aggrd.object_id == ArrayRecord([np.array(avg_list[0])]).object_id


def test_metricrecords_aggregation() -> None:
    """Test aggregation of MetricRecords."""
    num_replies = 3
    weights = [0.25, 0.4, 0.35]
    metric_records = [
        MetricRecord({"a": 1, "b": 2.0, "c": np.random.randn(3).tolist()})
        for _ in range(num_replies)
    ]

    # Compute expected aggregated MetricRecord.
    # For ease, we convert everything into numpy arrays, then aggregate
    as_np_entries = [
        {
            k: np.array(v) if isinstance(v, (int | float | list)) else v
            for k, v in record.items()
        }
        for record in metric_records
    ]
    avg_list = [
        np.average(
            [list(entries.values())[i] for entries in as_np_entries],
            axis=0,
            weights=weights,
        ).tolist()
        for i in range(len(as_np_entries[0]))
    ]
    expected_record = MetricRecord(
        dict(zip(as_np_entries[0].keys(), avg_list, strict=True))
    )
    expected_record["a"] = float(expected_record["a"])  # type: ignore
    expected_record["b"] = float(expected_record["b"])  # type: ignore

    # Construct RecordDicts (mimicing replies)
    # Inject weighting factor
    records = [
        RecordDict(
            {
                "metrics": MetricRecord(
                    record.__dict__["_data"] | {"weight": weights[i]}
                ),
            }
        )
        for i, record in enumerate(metric_records)
    ]

    # Execute aggregate
    aggrd = aggregate_metricrecords(records, weighting_metric_name="weight")
    # Assert
    assert expected_record.object_id == aggrd.object_id


@parameterized.expand(  # type: ignore
    [
        (
            True,
            RecordDict(
                {
                    "global-model": ArrayRecord([np.random.randn(7, 3)]),
                    "metrics": MetricRecord({"weight": 0.123}),
                }
            ),
        ),  # Compliant
        (
            False,
            RecordDict(
                {
                    "global-model": ArrayRecord([np.random.randn(7, 3)]),
                    "metrics": MetricRecord({"weight": [0.123]}),
                }
            ),
        ),  # Weighting key is not a scalar (BAD)
        (
            False,
            RecordDict(
                {
                    "global-model": ArrayRecord([np.random.randn(7, 3)]),
                    "metrics": MetricRecord({"loss": 0.01}),
                }
            ),
        ),  # No weighting key in MetricRecord (BAD)
        (
            False,
            RecordDict({"global-model": ArrayRecord([np.random.randn(7, 3)])}),
        ),  # No MetricsRecord (BAD)
        (
            False,
            RecordDict(
                {
                    "global-model": ArrayRecord([np.random.randn(7, 3)]),
                    "another-model": ArrayRecord([np.random.randn(7, 3)]),
                }
            ),
        ),  # Two ArrayRecords (BAD)
        (
            False,
            RecordDict(
                {
                    "global-model": ArrayRecord([np.random.randn(7, 3)]),
                    "metrics": MetricRecord({"weight": 0.123}),
                    "more-metrics": MetricRecord({"loss": 0.321}),
                }
            ),
        ),  # Two MetricRecords (BAD)
    ]
)
def test_consistency_of_replies_with_matching_keys(
    is_valid: bool, recorddict: RecordDict
) -> None:
    """Test consistency in replies."""
    # Create dummy records
    records = [recorddict for _ in range(3)]

    if not is_valid:
        # Should raise InconsistentMessageReplies exception
        with pytest.raises(InconsistentMessageReplies):
            validate_message_reply_consistency(
                records, weighted_by_key="weight", check_arrayrecord=True
            )
    else:
        # Should not raise an exception
        validate_message_reply_consistency(
            records, weighted_by_key="weight", check_arrayrecord=True
        )


@parameterized.expand(  # type: ignore
    [
        (
            [
                RecordDict(
                    {
                        "global-model": ArrayRecord([np.random.randn(7, 3)]),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
                RecordDict(
                    {
                        "model": ArrayRecord([np.random.randn(7, 3)]),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
            ],
        ),  # top-level keys don't match for ArrayRecords
        (
            [
                RecordDict(
                    {
                        "global-model": ArrayRecord(
                            {"a": Array(np.random.randn(7, 3))}
                        ),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
                RecordDict(
                    {
                        "global-model": ArrayRecord(
                            {"b": Array(np.random.randn(7, 3))}
                        ),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
            ],
        ),  # top-level keys match for ArrayRecords but not those for Arrays
        (
            [
                RecordDict(
                    {
                        "global-model": ArrayRecord([np.random.randn(7, 3)]),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
                RecordDict(
                    {
                        "global-model": ArrayRecord([np.random.randn(7, 3)]),
                        "my-metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
            ],
        ),  # top-level keys don't match for MetricRecords
        (
            [
                RecordDict(
                    {
                        "global-model": ArrayRecord([np.random.randn(7, 3)]),
                        "metrics": MetricRecord({"weight": 0.123}),
                    }
                ),
                RecordDict(
                    {
                        "global-model": ArrayRecord([np.random.randn(7, 3)]),
                        "my-metrics": MetricRecord({"my-weights": 0.123}),
                    }
                ),
            ],
        ),  # top-level keys match for MetricRecords but not inner ones
    ]
)
def test_consistency_of_replies_with_different_keys(
    list_records: list[RecordDict],
) -> None:
    """Test consistency in replies when records don't have matching keys."""
    # All test cases expect InconsistentMessageReplies exception to be raised
    with pytest.raises(InconsistentMessageReplies):
        validate_message_reply_consistency(
            list_records, weighted_by_key="weight", check_arrayrecord=True
        )
