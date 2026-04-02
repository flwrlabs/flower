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
"""RecordDict."""


from __future__ import annotations

import json
from logging import WARN
from textwrap import indent
from typing import TypeVar, cast

from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_object_body,
)

from ..logger import log
from .arrayrecord import ArrayRecord
from .configrecord import ConfigRecord
from .metricrecord import MetricRecord
from .typeddict import TypedDict

RecordType = ArrayRecord | MetricRecord | ConfigRecord


class _WarningTracker:
    """A class to track warnings for deprecated properties."""

    def __init__(self) -> None:
        # These variables are used to ensure that the deprecation warnings
        # for the deprecated properties/class are logged only once.
        self.recordset_init_logged = False
        self.recorddict_init_logged = False
        self.parameters_records_logged = False
        self.metrics_records_logged = False
        self.configs_records_logged = False


_warning_tracker = _WarningTracker()

T = TypeVar("T")


def _check_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError(
            f"Expected `{str.__name__}`, but "
            f"received `{type(key).__name__}` for the key."
        )


def _check_value(value: RecordType) -> None:
    if not isinstance(value, (ArrayRecord | MetricRecord | ConfigRecord)):
        raise TypeError(
            f"Expected `{ArrayRecord.__name__}`, `{MetricRecord.__name__}`, "
            f"or `{ConfigRecord.__name__}` but received "
            f"`{type(value).__name__}` for the value."
        )


class _SyncedDict(TypedDict[str, T]):
    """A synchronized dictionary that mirrors changes to an underlying RecordDict.

    This dictionary ensures that any modifications (set or delete operations)
    are automatically reflected in the associated `RecordDict`. Only values of
    the specified `allowed_type` are permitted.
    """

    def __init__(self, ref_recorddict: RecordDict, allowed_type: type[T]) -> None:
        if not issubclass(allowed_type, (ArrayRecord | MetricRecord | ConfigRecord)):
            raise TypeError(f"{allowed_type} is not a valid type.")
        super().__init__(_check_key, self.check_value)
        self.recorddict = ref_recorddict
        self.allowed_type = allowed_type

    def __setitem__(self, key: str, value: T) -> None:
        super().__setitem__(key, value)
        self.recorddict[key] = cast(RecordType, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.recorddict[key]

    def check_value(self, value: T) -> None:
        """Check if value is of expected type."""
        if not isinstance(value, self.allowed_type):
            raise TypeError(
                f"Expected `{self.allowed_type.__name__}`, but "
                f"received `{type(value).__name__}` for the value."
            )


class RecordDict(TypedDict[str, RecordType], InflatableObject):
    """RecordDict stores groups of arrays, metrics and configs.

    A :class:`RecordDict` is the unified mechanism by which arrays,
    metrics and configs can be either stored as part of a :class:`Context`
    in your apps or communicated as part of a :class:`Message` between
    your apps.

    Parameters
    ----------
    records : Optional[dict[str, RecordType]]
        A dictionary mapping string keys to record instances, where each value
        is either a :class:`ParametersRecord`, :class:`MetricsRecord`,
        or :class:`ConfigsRecord`.

    Examples
    --------
    A :class:`RecordDict` can hold three types of records, each designed
    with an specific purpose. What is common to all of them is that they
    are Python dictionaries designed to ensure that each key-value pair
    adheres to specified data types.

    Let's see an example::

        from flwr.common import RecordDict
        from flwr.common import ArrayRecord, ConfigRecord, MetricRecord

        # Let's begin with an empty record
        my_records = RecordDict()

        # We can create a ConfigRecord
        c_record = ConfigRecord({"lr": 0.1, "batch-size": 128})
        # Adding it to the RecordDict would look like this
        my_records["my_config"] = c_record

        # We can create a MetricRecord following a similar process
        m_record = MetricRecord({"accuracy": 0.93, "losses": [0.23, 0.1]})
        # Adding it to the RecordDict would look like this
        my_records["my_metrics"] = m_record

    Adding an :code:`ArrayRecord` follows the same steps as above but first,
    the array needs to be serialized and represented as a :code:`flwr.common.Array`.
    For example::

        from flwr.common import Array
        # Creating an ArrayRecord would look like this
        arr_np = np.random.randn(3, 3)

        # You can use the built-in tool to serialize the array
        arr = Array(arr_np)

        # Finally, create the record
        arr_record = ArrayRecord({"my_array": arr})

        # Adding it to the RecordDict would look like this
        my_records["my_parameters"] = arr_record

    For additional examples on how to construct each of the records types shown
    above, please refer to the documentation for :code:`ConfigRecord`,
    :code:`MetricRecord` and :code:`ArrayRecord`.
    """

    def __init__(
        self,
        records: dict[str, RecordType] | None = None,
        *,
        parameters_records: dict[str, ArrayRecord] | None = None,
        metrics_records: dict[str, MetricRecord] | None = None,
        configs_records: dict[str, ConfigRecord] | None = None,
    ) -> None:
        super().__init__(_check_key, _check_value)

        # Warning for deprecated usage
        if (
            parameters_records is not None
            or metrics_records is not None
            or configs_records is not None
        ):
            log(
                WARN,
                "The arguments `parameters_records`, `metrics_records`, and "
                "`configs_records` of `RecordDict` are deprecated and will "
                "be removed in a future release. "
                "Please pass all records using the `records` argument instead.",
            )
            records = records or {}
            records.update(parameters_records or {})
            records.update(metrics_records or {})
            records.update(configs_records or {})

        if records is not None:
            for key, record in records.items():
                self[key] = record

    @property
    def array_records(self) -> TypedDict[str, ArrayRecord]:
        """Dictionary holding only ArrayRecord instances."""
        synced_dict = _SyncedDict[ArrayRecord](self, ArrayRecord)
        for key, record in self.items():
            if isinstance(record, ArrayRecord):
                synced_dict[key] = record
        return synced_dict

    @property
    def metric_records(self) -> TypedDict[str, MetricRecord]:
        """Dictionary holding only MetricRecord instances."""
        synced_dict = _SyncedDict[MetricRecord](self, MetricRecord)
        for key, record in self.items():
            if isinstance(record, MetricRecord):
                synced_dict[key] = record
        return synced_dict

    @property
    def config_records(self) -> TypedDict[str, ConfigRecord]:
        """Dictionary holding only ConfigRecord instances."""
        synced_dict = _SyncedDict[ConfigRecord](self, ConfigRecord)
        for key, record in self.items():
            if isinstance(record, ConfigRecord):
                synced_dict[key] = record
        return synced_dict

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        flds = ("array_records", "metric_records", "config_records")
        fld_views = [f"{fld}={dict(getattr(self, fld))!r}" for fld in flds]
        view = indent(",\n".join(fld_views), "  ")
        return f"{self.__class__.__qualname__}(\n{view}\n)"

    def __setitem__(self, key: str, value: RecordType) -> None:
        """Set the given key to the given value after type checking."""
        original_value = self.get(key, None)
        super().__setitem__(key, value)
        if original_value is not None and not isinstance(value, type(original_value)):
            log(
                WARN,
                "Key '%s' was overwritten: record of type `%s` replaced with type `%s`",
                key,
                type(original_value).__name__,
                type(value).__name__,
            )

    @property
    def parameters_records(self) -> TypedDict[str, ArrayRecord]:
        """Deprecated property.

        Use ``array_records`` instead.
        """
        if _warning_tracker.parameters_records_logged:
            _warning_tracker.parameters_records_logged = True
            log(
                WARN,
                "The `parameters_records` property of `RecordDict` "
                "(formerly `RecordSet`) is deprecated and will be removed in a "
                "future release. Please use the `array_records` property instead.",
            )
        return self.array_records

    @property
    def metrics_records(self) -> TypedDict[str, MetricRecord]:
        """Deprecated property.

        Use ``metric_records`` instead.
        """
        if not _warning_tracker.metrics_records_logged:
            _warning_tracker.metrics_records_logged = True
            log(
                WARN,
                "The `metrics_records` property of `RecordDict` "
                "(formerly `RecordSet`) is deprecated and will be removed in a "
                "future release. Please use the `metric_records` property instead.",
            )
        return self.metric_records

    @property
    def configs_records(self) -> TypedDict[str, ConfigRecord]:
        """Deprecated property.

        Use ``config_records`` instead.
        """
        if not _warning_tracker.configs_records_logged:
            _warning_tracker.configs_records_logged = True
            log(
                WARN,
                "The `configs_records` property of `RecordDict` "
                "(formerly `RecordSet`) is deprecated and will be removed in a "
                "future release. Please use the `config_records` property instead.",
            )
        return self.config_records

    @property
    def children(self) -> dict[str, InflatableObject]:
        """Return a dictionary of records with their Object IDs as keys."""
        return {record.object_id: record for record in self.values()}

    def deflate(self) -> bytes:
        """Deflate the RecordDict."""
        # record_name: record_object_id mapping
        record_refs: dict[str, str] = {}

        for record_name, record in self.items():
            record_refs[record_name] = record.object_id

        # Serialize references dict
        object_body = json.dumps(record_refs).encode("utf-8")
        return add_header_to_object_body(object_body=object_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> RecordDict:
        """Inflate an RecordDict from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the RecordDict.
        children : Optional[dict[str, InflatableObject]]  (default: None)
            Dictionary of children InflatableObjects mapped to their Object IDs.
            These children enable the full inflation of the RecordDict. Default is None.

        Returns
        -------
        RecordDict
            The inflated RecordDict.
        """
        if children is None:
            children = {}

        # Inflate mapping of record_names (keys in the RecordDict) to Record' object IDs
        obj_body = get_object_body(object_content, cls)
        record_refs: dict[str, str] = json.loads(obj_body.decode(encoding="utf-8"))

        unique_records = set(record_refs.values())
        children_obj_ids = set(children.keys())
        if unique_records != children_obj_ids:
            raise ValueError(
                "Unexpected set of `children`. "
                f"Expected {unique_records} but got {children_obj_ids}."
            )

        # Ensure children are one of the *Record objects exepecte in a RecordDict
        if not all(
            isinstance(ch, (ArrayRecord | ConfigRecord | MetricRecord))
            for ch in children.values()
        ):
            raise ValueError(
                "`Children` are expected to be of type `ArrayRecord`, "
                "`ConfigRecord` or `MetricRecord`."
            )

        # Instantiate new RecordDict
        return RecordDict(
            {name: children[object_id] for name, object_id in record_refs.items()}  # type: ignore
        )


class RecordSet(RecordDict):
    """Deprecated class ``RecordSet``, use ``RecordDict`` instead.

    This class exists solely for backward compatibility with legacy
    code that previously used ``RecordSet``. It has been renamed
    to ``RecordDict`` and will be removed in a future release.

    .. warning::
        ``RecordSet`` is deprecated and will be removed in a future release.
        Use ``RecordDict`` instead.

    Examples
    --------
    Legacy (deprecated) usage::

        from flwr.common import RecordSet

        my_content = RecordSet()

    Updated usage::

        from flwr.common import RecordDict

        my_content = RecordDict()
    """

    def __init__(
        self,
        records: dict[str, RecordType] | None = None,
        *,
        parameters_records: dict[str, ArrayRecord] | None = None,
        metrics_records: dict[str, MetricRecord] | None = None,
        configs_records: dict[str, ConfigRecord] | None = None,
    ) -> None:
        if not _warning_tracker.recordset_init_logged:
            _warning_tracker.recordset_init_logged = True
            log(
                WARN,
                "The `RecordSet` class has been renamed to `RecordDict`. "
                "Support for `RecordSet` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(
            records,
            parameters_records=parameters_records,
            metrics_records=metrics_records,
            configs_records=configs_records,
        )
