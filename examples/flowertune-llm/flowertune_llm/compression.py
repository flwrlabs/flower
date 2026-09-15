# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Compression helpers for the layer-wise FlowerTune LLM example."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.common.compression import CompressionStats, compress_arrayrecord, create_pipeline


def _as_bool(value: Any) -> bool:
    """Parse booleans from TOML/run-config values."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get(config: Mapping[str, Any] | ConfigRecord, key: str, default: Any) -> Any:
    """Read a value from a ConfigRecord-like mapping."""
    return config[key] if key in config else default


def compression_enabled(config: Mapping[str, Any] | ConfigRecord) -> bool:
    """Return whether layer-wise payload compression is enabled."""
    enabled = _as_bool(_get(config, "compression.enabled", False))
    pipeline_name = str(_get(config, "compression.pipeline", "none"))
    return enabled and pipeline_name.lower() not in {"none", "no_compression"}


def compression_config(config: Mapping[str, Any] | ConfigRecord) -> dict[str, Any]:
    """Extract compression settings to forward to ClientApp messages."""
    return {
        "compression.enabled": _as_bool(_get(config, "compression.enabled", False)),
        "compression.pipeline": str(_get(config, "compression.pipeline", "none")),
        "compression.n-bits": int(_get(config, "compression.n-bits", 3)),
        "compression.block-size": int(_get(config, "compression.block-size", 262_144)),
        "compression.cuda-enabled": _as_bool(
            _get(config, "compression.cuda-enabled", False)
        ),
        "compression.rotation-enabled": _as_bool(
            _get(config, "compression.rotation-enabled", True)
        ),
        "compression.rotation-seed": int(
            _get(config, "compression.rotation-seed", 2025)
        ),
    }


def compress_if_enabled(
    arrays: ArrayRecord,
    config: Mapping[str, Any] | ConfigRecord,
) -> tuple[ArrayRecord, CompressionStats | None, float]:
    """Compress an ArrayRecord when enabled and return elapsed milliseconds."""
    if not compression_enabled(config) or len(arrays) == 0:
        return arrays, None, 0.0

    pipeline = create_pipeline(
        str(_get(config, "compression.pipeline", "turboquant_mse")),
        n_bits=int(_get(config, "compression.n-bits", 3)),
        block_size=int(_get(config, "compression.block-size", 262_144)),
        use_cuda=_as_bool(_get(config, "compression.cuda-enabled", False)),
        rotation=_as_bool(_get(config, "compression.rotation-enabled", True)),
        rotation_seed=int(_get(config, "compression.rotation-seed", 2025)),
    )
    started_at = perf_counter()
    compressed, stats = compress_arrayrecord(arrays, pipeline)
    return compressed, stats, (perf_counter() - started_at) * 1000.0


def add_compression_metrics(
    metrics: MetricRecord,
    *,
    prefix: str,
    stats: CompressionStats | None,
    elapsed_ms: float,
) -> None:
    """Attach compression counters to a MetricRecord."""
    if stats is None:
        return
    metrics[f"{prefix}.raw_bytes"] = stats.raw_bytes
    metrics[f"{prefix}.compressed_bytes"] = stats.compressed_bytes
    metrics[f"{prefix}.ratio"] = stats.ratio
    metrics[f"{prefix}.arrays"] = stats.arrays
    metrics[f"{prefix}.ms"] = elapsed_ms
    metrics[f"{prefix}.relative_rmse"] = stats.relative_rmse
    metrics[f"{prefix}.cosine_similarity"] = stats.cosine_similarity
