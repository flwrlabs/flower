# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Helpers for compressing Message API ArrayRecords."""

from __future__ import annotations

from dataclasses import dataclass
import math

from flwr.common.constant import SType
from flwr.common.record import Array, ArrayRecord, RecordDict

from .envelope import CompressionEnvelope, encode_envelope
from .pipeline import TransformationPipeline


@dataclass(frozen=True)
class CompressionStats:
    """Compression byte counters."""

    raw_bytes: int
    compressed_bytes: int
    arrays: int
    squared_error: float = 0.0
    squared_norm: float = 0.0
    reconstructed_squared_norm: float = 0.0
    dot_product: float = 0.0

    @property
    def ratio(self) -> float:
        """Return raw/compressed ratio."""
        return self.raw_bytes / max(1, self.compressed_bytes)

    @property
    def relative_rmse(self) -> float:
        """Return aggregate L2 reconstruction error relative to input norm."""
        return math.sqrt(self.squared_error / max(self.squared_norm, 1e-30))

    @property
    def cosine_similarity(self) -> float:
        """Return aggregate cosine similarity of original and reconstructed arrays."""
        denominator = math.sqrt(
            max(self.squared_norm, 1e-30)
            * max(self.reconstructed_squared_norm, 1e-30)
        )
        return self.dot_product / denominator


def compress_array(
    array: Array, pipeline: TransformationPipeline
) -> tuple[Array, CompressionStats]:
    """Compress one Array into a compressed Array envelope."""
    if array.stype == SType.COMPRESSED_PIPELINE:
        return array, CompressionStats(len(array.data), len(array.data), 0)
    ndarray = array.numpy()
    payload, metadata = pipeline.forward(ndarray)
    envelope = CompressionEnvelope(
        pipeline_id=pipeline.pipeline_id,
        pipeline_params=pipeline.params,
        metadata=metadata,
        payload=payload,
    )
    encoded = encode_envelope(envelope)
    compressed = Array(
        dtype=array.dtype,
        shape=tuple(array.shape),
        stype=SType.COMPRESSED_PIPELINE,
        data=encoded,
    )
    transformer_metadata = metadata.get("transformers", [{}])[0]
    return compressed, CompressionStats(
        len(array.data),
        len(encoded),
        1,
        squared_error=float(transformer_metadata.get("squared_error", 0.0)),
        squared_norm=float(transformer_metadata.get("squared_norm", 0.0)),
        reconstructed_squared_norm=float(
            transformer_metadata.get("reconstructed_squared_norm", 0.0)
        ),
        dot_product=float(transformer_metadata.get("dot_product", 0.0)),
    )


def compress_arrayrecord(
    record: ArrayRecord, pipeline: TransformationPipeline
) -> tuple[ArrayRecord, CompressionStats]:
    """Compress all Arrays in an ArrayRecord."""
    out = ArrayRecord()
    raw = 0
    compressed = 0
    arrays = 0
    squared_error = 0.0
    squared_norm = 0.0
    reconstructed_squared_norm = 0.0
    dot_product = 0.0
    for key, array in record.items():
        out_array, stats = compress_array(array, pipeline)
        out[key] = out_array
        raw += stats.raw_bytes
        compressed += stats.compressed_bytes
        arrays += stats.arrays
        squared_error += stats.squared_error
        squared_norm += stats.squared_norm
        reconstructed_squared_norm += stats.reconstructed_squared_norm
        dot_product += stats.dot_product
    return out, CompressionStats(
        raw,
        compressed,
        arrays,
        squared_error,
        squared_norm,
        reconstructed_squared_norm,
        dot_product,
    )


def compress_recorddict_arrayrecords(
    records: RecordDict, pipeline: TransformationPipeline
) -> tuple[int, int]:
    """Compress ArrayRecords in-place inside a RecordDict."""
    raw = 0
    compressed = 0
    for key, record in list(records.array_records.items()):
        compressed_record, stats = compress_arrayrecord(record, pipeline)
        records[key] = compressed_record
        raw += stats.raw_bytes
        compressed += stats.compressed_bytes
    return raw, compressed
