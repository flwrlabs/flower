"""Shared helpers for SSFL Flower app records and metrics."""

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict


def array_record(records: RecordDict, key: str = "arrays") -> ArrayRecord:
    """Return an ArrayRecord stored under ``key``."""
    record = records[key]
    if not isinstance(record, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord under {key!r}")
    return record


def config_record(records: RecordDict, key: str = "config") -> ConfigRecord:
    """Return a ConfigRecord stored under ``key``."""
    record = records[key]
    if not isinstance(record, ConfigRecord):
        raise TypeError(f"Expected ConfigRecord under {key!r}")
    return record


def metric_record(records: RecordDict, key: str = "metrics") -> MetricRecord:
    """Return a MetricRecord stored under ``key``."""
    record = records[key]
    if not isinstance(record, MetricRecord):
        raise TypeError(f"Expected MetricRecord under {key!r}")
    return record


def metric_float(value: object) -> float:
    """Return a scalar numeric metric as a float."""
    if not isinstance(value, (int, float)):
        raise TypeError("Expected a scalar numeric metric")
    return float(value)
