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
"""MetricRecord."""


from __future__ import annotations

from logging import WARN
from typing import cast, get_args

from flwr.common.typing import MetricRecordValues, MetricScalar

# pylint: disable=E0611
from flwr.proto.recorddict_pb2 import MetricRecord as ProtoMetricRecord
from flwr.proto.recorddict_pb2 import MetricRecordValue as ProtoMetricRecordValue

# pylint: enable=E0611
from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_object_body,
)

from ..logger import log
from ..serde_utils import record_value_dict_from_proto, record_value_dict_to_proto
from .typeddict import TypedDict


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: MetricRecordValues) -> None:
    def is_valid(__v: MetricScalar) -> None:
        """Check if value is of expected type."""
        if not isinstance(__v, get_args(MetricScalar)) or isinstance(__v, bool):
            raise TypeError(
                "Not all values are of valid type."
                f" Expected `{MetricRecordValues}` but `{type(__v)}` was passed."
            )

    if isinstance(value, list):
        # If your lists are large (e.g. 1M+ elements) this will be slow
        # 1s to check 10M element list on a M2 Pro
        # In such settings, you'd be better of treating such metric as
        # an array and pass it to an ArrayRecord.
        # Empty lists are valid
        if len(value) > 0:
            is_valid(value[0])
            # all elements in the list must be of the same valid type
            # this is needed for protobuf
            value_type = type(value[0])
            if not all(isinstance(v, value_type) for v in value):
                raise TypeError(
                    "All values in a list must be of the same valid type. "
                    f"One of {MetricScalar}."
                )
    else:
        is_valid(value)


class MetricRecord(TypedDict[str, MetricRecordValues], InflatableObject):
    """Metric record.

    A :code:`MetricRecord` is a Python dictionary designed to ensure that
    each key-value pair adheres to specified data types. A :code:`MetricRecord`
    is one of the types of records that a
    `flwr.common.RecordDict <flwr.common.RecordDict.html#recorddict>`_ supports and
    can therefore be used to construct :code:`common.Message` objects.

    Parameters
    ----------
    metric_dict : Optional[Dict[str, MetricRecordValues]]
        A dictionary that stores basic types (i.e. `int`, `float` as defined
        in `MetricScalar`) and list of such types (see `MetricScalarList`).
    keep_input : bool (default: True)
        A boolean indicating whether metrics should be deleted from the input
        dictionary immediately after adding them to the record. When set
        to True, the data is duplicated in memory. If memory is a concern, set
        it to False.

    Examples
    --------
    The usage of a :code:`MetricRecord` is envisioned for communicating results
    obtained when a node performs an action. A few typical examples include:
    communicating the training accuracy after a model is trained locally by a
    :code:`ClientApp`, reporting the validation loss obtained at a :code:`ClientApp`,
    or, more generally, the output of executing a query by the :code:`ClientApp`.
    Common to these examples is that the output can be typically represented by
    a single scalar (:code:`int`, :code:`float`) or list of scalars.

    Let's see some examples of how to construct a :code:`MetricRecord` from scratch::

        from flwr.common import MetricRecord

        # A `MetricRecord` is a specialized Python dictionary
        record = MetricRecord({"accuracy": 0.94})
        # You can add more content to an existing record
        record["loss"] = 0.01
        # It also supports lists
        record["loss-historic"] = [0.9, 0.5, 0.01]

    Since types are enforced, the types of the objects inserted are checked. For a
    :code:`MetricRecord`, value types allowed are those in defined in
    :code:`flwr.common.MetricRecordValues`. Similarly, only :code:`str` keys are
    allowed::

        from flwr.common import MetricRecord

        record = MetricRecord() # an empty record
        # Add unsupported value
        record["something-unsupported"] = {'a': 123} # Will throw a `TypeError`

    If you need a more versatily type of record try :code:`ConfigRecord` or
    :code:`ArrayRecord`.
    """

    def __init__(
        self,
        metric_dict: dict[str, MetricRecordValues] | None = None,
        keep_input: bool = True,
    ) -> None:
        super().__init__(_check_key, _check_value)
        if metric_dict:
            for k in list(metric_dict.keys()):
                self[k] = metric_dict[k]
                if not keep_input:
                    del metric_dict[k]

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this object."""
        num_bytes = 0

        for k, v in self.items():
            if isinstance(v, list):
                # both int and float normally take 4 bytes
                # But MetricRecords are mapped to 64bit int/float
                # during protobuffing
                num_bytes += 8 * len(v)
            else:
                num_bytes += 8
            # We also count the bytes footprint of the keys
            num_bytes += len(k)
        return num_bytes

    def deflate(self) -> bytes:
        """Deflate object."""
        protos = record_value_dict_to_proto(self, [float, int], ProtoMetricRecordValue)
        obj_body = ProtoMetricRecord(
            items=[ProtoMetricRecord.Item(key=k, value=v) for k, v in protos.items()]
        ).SerializeToString()
        return add_header_to_object_body(object_body=obj_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> MetricRecord:
        """Inflate a MetricRecord from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the MetricRecord.

        children : Optional[dict[str, InflatableObject]] (default: None)
            Must be ``None``. ``MetricRecord`` does not support child objects.
            Providing any children will raise a ``ValueError``.

        Returns
        -------
        MetricRecord
            The inflated MetricRecord.
        """
        if children:
            raise ValueError("`MetricRecord` objects do not have children.")

        obj_body = get_object_body(object_content, cls)
        metric_record_proto = ProtoMetricRecord.FromString(obj_body)
        protos = {item.key: item.value for item in metric_record_proto.items}
        return cls(
            metric_dict=cast(
                dict[str, MetricRecordValues],
                record_value_dict_from_proto(protos),
            ),
            keep_input=False,
        )


class MetricsRecord(MetricRecord):
    """Deprecated class ``MetricsRecord``, use ``MetricRecord`` instead.

    This class exists solely for backward compatibility with legacy
    code that previously used ``MetricsRecord``. It has been renamed
    to ``MetricRecord``.

    .. warning::
        ``MetricsRecord`` is deprecated and will be removed in a future release.
        Use ``MetricRecord`` instead.

    Examples
    --------
    Legacy (deprecated) usage::

        from flwr.common import MetricsRecord

        record = MetricsRecord()

    Updated usage::

        from flwr.common import MetricRecord

        record = MetricRecord()
    """

    _warning_logged = False

    def __init__(
        self,
        metric_dict: dict[str, MetricRecordValues] | None = None,
        keep_input: bool = True,
    ):
        if not MetricsRecord._warning_logged:
            MetricsRecord._warning_logged = True
            log(
                WARN,
                "The `MetricsRecord` class has been renamed to `MetricRecord`. "
                "Support for `MetricsRecord` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(metric_dict, keep_input)
