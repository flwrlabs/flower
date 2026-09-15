# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Flower compression pipelines."""

from .delta import ArrayRecordDelta, DeltaState
from .cuda import (
    dequantize_mse_cuda,
    describe_cuda_path,
    is_cuda_available,
    pack_3bit,
    quantize_mse_cuda,
    unpack_3bit,
)
from .envelope import CompressionEnvelope, decode_envelope, encode_envelope
from .message import (
    CompressionStats,
    compress_array,
    compress_arrayrecord,
    compress_recorddict_arrayrecords,
)
from .pipeline import NoCompressionPipeline, TransformationPipeline, Transformer
from .registry import create_pipeline
from .turboquant import (
    EdenUnbiasedPipeline,
    TurboQuantMSEPipeline,
    TurboQuantMSETransformer,
)

__all__ = [
    "ArrayRecordDelta",
    "CompressionEnvelope",
    "CompressionStats",
    "DeltaState",
    "EdenUnbiasedPipeline",
    "NoCompressionPipeline",
    "TransformationPipeline",
    "Transformer",
    "TurboQuantMSEPipeline",
    "TurboQuantMSETransformer",
    "compress_array",
    "compress_arrayrecord",
    "compress_recorddict_arrayrecords",
    "create_pipeline",
    "decode_envelope",
    "dequantize_mse_cuda",
    "describe_cuda_path",
    "encode_envelope",
    "is_cuda_available",
    "pack_3bit",
    "quantize_mse_cuda",
    "unpack_3bit",
]
