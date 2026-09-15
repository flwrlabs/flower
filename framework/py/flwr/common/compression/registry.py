# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Compression pipeline registry."""

from __future__ import annotations

from typing import Any

from .pipeline import NoCompressionPipeline, TransformationPipeline
from .turboquant import EdenUnbiasedPipeline, TurboQuantMSEPipeline


def _as_bool(value: Any) -> bool:
    """Parse booleans from run config values."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def create_pipeline(name: str, **params: Any) -> TransformationPipeline:
    """Create a compression pipeline by name."""
    normalized = name.strip().lower()
    if normalized in {"none", "no_compression", "no-compression"}:
        return NoCompressionPipeline()
    if normalized in {"turboquant_mse", "turboquant-mse", "tq_mse"}:
        return TurboQuantMSEPipeline(
            n_bits=int(params.get("n_bits", params.get("bits", 3))),
            block_size=int(params.get("block_size", 262_144)),
            use_cuda=_as_bool(params.get("use_cuda", params.get("cuda", False))),
            rotation=_as_bool(params.get("rotation", True)),
            rotation_seed=int(params.get("rotation_seed", 2025)),
        )
    if normalized in {"eden", "eden_unbiased", "eden-unbiased"}:
        return EdenUnbiasedPipeline(
            n_bits=int(params.get("n_bits", params.get("bits", 3))),
            block_size=int(params.get("block_size", 262_144)),
            use_cuda=_as_bool(params.get("use_cuda", params.get("cuda", False))),
            rotation=_as_bool(params.get("rotation", True)),
            rotation_seed=int(params.get("rotation_seed", 2025)),
        )
    raise ValueError(f"Unknown compression pipeline: {name}")
