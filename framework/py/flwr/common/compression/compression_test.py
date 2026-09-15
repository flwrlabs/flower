# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Tests for compression pipelines."""

from __future__ import annotations

import numpy as np

from flwr.common.compression import (
    DeltaState,
    EdenUnbiasedPipeline,
    TurboQuantMSEPipeline,
    compress_arrayrecord,
    create_pipeline,
)
from flwr.common.constant import SType
from flwr.common.record import Array, ArrayRecord


def test_turboquant_bit_packing_roundtrip_for_sweep_widths() -> None:
    """Four- and six-bit vectorized packing should preserve every index."""
    from flwr.common.compression.turboquant import _pack_bits, _unpack_bits

    rng = np.random.default_rng(4)
    for bits in (4, 6):
        values = rng.integers(0, 1 << bits, size=1027, dtype=np.uint8)
        restored = _unpack_bits(_pack_bits(values, bits), bits, values.size)
        np.testing.assert_array_equal(restored, values)


def test_hadamard_rotation_improves_spiky_delta_distortion() -> None:
    """Distribution mixing should protect sparse, heavy-tailed update blocks."""
    array = np.zeros(256, dtype=np.float32)
    array[[0, 17, 93]] = np.array([12.0, -5.0, 2.0], dtype=np.float32)
    plain = TurboQuantMSEPipeline(n_bits=3, block_size=256, rotation=False)
    rotated = TurboQuantMSEPipeline(n_bits=3, block_size=256, rotation=True)

    plain_restored = plain.backward(*plain.forward(array))
    rotated_restored = rotated.backward(*rotated.forward(array))
    plain_error = np.mean((plain_restored - array) ** 2)
    rotated_error = np.mean((rotated_restored - array) ** 2)

    assert rotated_error < plain_error


def test_turboquant_mse_roundtrip_shape_dtype() -> None:
    """TurboQuant should preserve shape and dtype on decode."""
    rng = np.random.default_rng(7)
    array = rng.normal(size=(256, 32)).astype(np.float32)
    pipeline = TurboQuantMSEPipeline(n_bits=4, block_size=256)

    payload, metadata = pipeline.forward(array)
    decoded = pipeline.backward(payload, metadata)

    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype
    assert np.mean((decoded - array) ** 2) < np.mean(array**2)


def test_arrayrecord_compression_decodes_via_array_numpy() -> None:
    """Compressed Array stype should decode through Array.numpy."""
    rng = np.random.default_rng(8)
    array = rng.normal(size=(1024,)).astype(np.float32)
    record = ArrayRecord({"x": Array(array)})
    pipeline = TurboQuantMSEPipeline(n_bits=4, block_size=128)

    compressed, stats = compress_arrayrecord(record, pipeline)
    decoded = compressed["x"].numpy()

    assert compressed["x"].stype == SType.COMPRESSED_PIPELINE
    assert stats.raw_bytes > stats.compressed_bytes
    expected_relative_rmse = float(
        np.linalg.norm(decoded.astype(np.float64) - array.astype(np.float64))
        / np.linalg.norm(array.astype(np.float64))
    )
    expected_cosine = float(
        np.dot(decoded.astype(np.float64), array.astype(np.float64))
        / (
            np.linalg.norm(decoded.astype(np.float64))
            * np.linalg.norm(array.astype(np.float64))
        )
    )
    assert abs(stats.relative_rmse - expected_relative_rmse) < 1e-5
    assert abs(stats.cosine_similarity - expected_cosine) < 1e-5
    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype


def test_turboquant_mse_cuda_flag_roundtrip() -> None:
    """CUDA flag should be optional and preserve CPU-compatible payloads."""
    rng = np.random.default_rng(9)
    array = rng.normal(size=(128,)).astype(np.float32)
    pipeline = create_pipeline(
        "turboquant_mse", n_bits=3, block_size=32, use_cuda=True
    )

    payload, metadata = pipeline.forward(array)
    decoded = pipeline.backward(payload, metadata)

    assert metadata["transformers"][0]["cuda_requested"] is True
    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype


def test_eden_unbiased_scale_preserves_input_projection() -> None:
    """EDEN's scale should make the reconstruction projection unbiased."""
    rng = np.random.default_rng(19)
    array = rng.normal(size=256).astype(np.float32)
    pipeline = EdenUnbiasedPipeline(n_bits=4, block_size=128)

    decoded = pipeline.backward(*pipeline.forward(array))

    expected = float(np.dot(array, array))
    observed = float(np.dot(array, decoded))
    assert abs(observed - expected) / expected < 2e-3


def test_delta_state_extract_and_apply() -> None:
    """DeltaState should compute and apply ArrayRecord deltas."""
    base = ArrayRecord({"w": Array(np.array([1.0, 2.0], dtype=np.float32))})
    updated = ArrayRecord({"w": Array(np.array([1.5, 1.75], dtype=np.float32))})
    state = DeltaState.from_arrayrecord(base)

    delta = state.extract_delta(updated)
    restored = state.apply_delta(delta)

    np.testing.assert_allclose(
        delta["w"].numpy(), np.array([0.5, -0.25], dtype=np.float32)
    )
    np.testing.assert_allclose(restored["w"].numpy(), updated["w"].numpy())
