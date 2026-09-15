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
"""TurboQuant MSE compression pipeline."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from flwr.common.typing import NDArray

from .pipeline import TransformationPipeline, Transformer


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    values_u = values.astype(np.uint8, copy=False).reshape(-1)
    if bits == 8:
        return values_u.tobytes()
    if bits == 3:
        pad = (-values_u.size) % 8
        if pad:
            values_u = np.pad(values_u, (0, pad))
        grouped = values_u.reshape(-1, 8).astype(np.uint16, copy=False)
        out = np.empty((grouped.shape[0], 3), dtype=np.uint8)
        out[:, 0] = grouped[:, 0] | (grouped[:, 1] << 3) | (grouped[:, 2] << 6)
        out[:, 1] = (
            (grouped[:, 2] >> 2)
            | (grouped[:, 3] << 1)
            | (grouped[:, 4] << 4)
            | (grouped[:, 5] << 7)
        )
        out[:, 2] = (grouped[:, 5] >> 1) | (grouped[:, 6] << 2) | (
            grouped[:, 7] << 5
        )
        return out.reshape(-1).tobytes()
    if bits == 4:
        pad = (-values_u.size) % 2
        if pad:
            values_u = np.pad(values_u, (0, pad))
        grouped = values_u.reshape(-1, 2).astype(np.uint16, copy=False)
        return (grouped[:, 0] | (grouped[:, 1] << 4)).astype(np.uint8).tobytes()
    if bits == 6:
        pad = (-values_u.size) % 4
        if pad:
            values_u = np.pad(values_u, (0, pad))
        grouped = values_u.reshape(-1, 4).astype(np.uint16, copy=False)
        out = np.empty((grouped.shape[0], 3), dtype=np.uint8)
        out[:, 0] = grouped[:, 0] | (grouped[:, 1] << 6)
        out[:, 1] = (grouped[:, 1] >> 2) | (grouped[:, 2] << 4)
        out[:, 2] = (grouped[:, 2] >> 4) | (grouped[:, 3] << 2)
        return out.reshape(-1).tobytes()
    out = bytearray(math.ceil(values_u.size * bits / 8))
    bit_pos = 0
    mask = (1 << bits) - 1
    for value in values_u:
        v = int(value) & mask
        byte_idx = bit_pos >> 3
        offset = bit_pos & 7
        out[byte_idx] |= (v << offset) & 0xFF
        if offset + bits > 8:
            out[byte_idx + 1] |= v >> (8 - offset)
        bit_pos += bits
    return bytes(out)


def _unpack_bits(payload: bytes, bits: int, count: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(payload, dtype=np.uint8, count=count).copy()
    if bits == 3:
        raw = np.frombuffer(payload, dtype=np.uint8)
        pad = (-raw.size) % 3
        if pad:
            raw = np.pad(raw, (0, pad))
        grouped = raw.reshape(-1, 3).astype(np.uint16, copy=False)
        out = np.empty((grouped.shape[0], 8), dtype=np.uint8)
        b0 = grouped[:, 0]
        b1 = grouped[:, 1]
        b2 = grouped[:, 2]
        out[:, 0] = b0 & 0x07
        out[:, 1] = (b0 >> 3) & 0x07
        out[:, 2] = ((b0 >> 6) | (b1 << 2)) & 0x07
        out[:, 3] = (b1 >> 1) & 0x07
        out[:, 4] = (b1 >> 4) & 0x07
        out[:, 5] = ((b1 >> 7) | (b2 << 1)) & 0x07
        out[:, 6] = (b2 >> 2) & 0x07
        out[:, 7] = (b2 >> 5) & 0x07
        return out.reshape(-1)[:count].copy()
    if bits == 4:
        raw = np.frombuffer(payload, dtype=np.uint8)
        out = np.empty((raw.size, 2), dtype=np.uint8)
        out[:, 0] = raw & 0x0F
        out[:, 1] = raw >> 4
        return out.reshape(-1)[:count].copy()
    if bits == 6:
        raw = np.frombuffer(payload, dtype=np.uint8)
        pad = (-raw.size) % 3
        if pad:
            raw = np.pad(raw, (0, pad))
        grouped = raw.reshape(-1, 3).astype(np.uint16, copy=False)
        out = np.empty((grouped.shape[0], 4), dtype=np.uint8)
        b0 = grouped[:, 0]
        b1 = grouped[:, 1]
        b2 = grouped[:, 2]
        out[:, 0] = b0 & 0x3F
        out[:, 1] = ((b0 >> 6) | (b1 << 2)) & 0x3F
        out[:, 2] = ((b1 >> 4) | (b2 << 4)) & 0x3F
        out[:, 3] = (b2 >> 2) & 0x3F
        return out.reshape(-1)[:count].copy()
    raw = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint8)
    bit_pos = 0
    mask = (1 << bits) - 1
    for idx in range(count):
        byte_idx = bit_pos >> 3
        offset = bit_pos & 7
        value = int(raw[byte_idx]) >> offset
        if offset + bits > 8:
            value |= int(raw[byte_idx + 1]) << (8 - offset)
        out[idx] = value & mask
        bit_pos += bits
    return out


@lru_cache(maxsize=16)
def _normal_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    levels = 1 << bits
    rng = np.random.default_rng(1000 + bits)
    sample = np.sort(rng.standard_normal(250_000).astype(np.float32))
    centroids = np.quantile(
        sample, np.linspace(0.0, 1.0, levels + 2, dtype=np.float32)[1:-1]
    ).astype(np.float32)
    for _ in range(30):
        boundaries = ((centroids[:-1] + centroids[1:]) * 0.5).astype(np.float32)
        indices = np.searchsorted(boundaries, sample, side="right")
        updated = centroids.copy()
        for level in range(levels):
            values = sample[indices == level]
            if values.size:
                updated[level] = values.mean(dtype=np.float64)
        if np.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    boundaries = ((centroids[:-1] + centroids[1:]) * 0.5).astype(np.float32)
    return centroids.astype(np.float32), boundaries


def _cuda_enabled_for(bits: int, use_cuda: bool) -> bool:
    """Return whether the CUDA implementation should be used."""
    if not use_cuda or bits not in {3, 4}:
        return False
    try:
        from .cuda import is_cuda_available
    except ImportError:
        return False
    return is_cuda_available()


def _is_cuda_oom(exc: Exception) -> bool:
    """Return whether an exception is a CUDA OOM from PyTorch."""
    return exc.__class__.__name__ == "OutOfMemoryError" and "CUDA out of memory" in str(
        exc
    )


@lru_cache(maxsize=32)
def _rotation_signs(block_size: int, seed: int) -> np.ndarray:
    """Return deterministic Rademacher signs for a randomized Hadamard rotation."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), block_size)


def _hadamard_rotate(blocks: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply an orthonormal randomized Walsh-Hadamard transform to rows."""
    width = blocks.shape[1]
    rotated = blocks * signs[None, :]
    stride = 1
    while stride < width:
        grouped = rotated.reshape(-1, width // (2 * stride), 2, stride)
        left = grouped[:, :, 0, :].copy()
        right = grouped[:, :, 1, :].copy()
        grouped[:, :, 0, :] = left + right
        grouped[:, :, 1, :] = left - right
        stride *= 2
    rotated *= np.float32(1.0 / math.sqrt(width))
    return rotated


def _hadamard_unrotate(blocks: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Invert the orthonormal randomized Walsh-Hadamard transform."""
    return _hadamard_rotate(blocks, np.ones_like(signs)) * signs[None, :]


class TurboQuantMSETransformer(Transformer):
    """Block-normalized scalar MSE TurboQuant transformer."""

    def __init__(
        self,
        *,
        n_bits: int = 3,
        block_size: int = 262_144,
        use_cuda: bool = False,
        rotation: bool = True,
        rotation_seed: int = 2025,
        scale_mode: str = "rms",
    ) -> None:
        if not 1 <= n_bits <= 8:
            raise ValueError("n_bits must be in [1, 8]")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if rotation and block_size & (block_size - 1):
            raise ValueError("Hadamard rotation requires a power-of-two block_size")
        self.n_bits = n_bits
        self.block_size = block_size
        self.use_cuda = use_cuda
        self.rotation = rotation
        self.rotation_seed = rotation_seed
        if scale_mode not in {"rms", "unbiased"}:
            raise ValueError("scale_mode must be 'rms' or 'unbiased'")
        self.scale_mode = scale_mode

    def forward(self, array: NDArray) -> tuple[bytes, dict[str, Any]]:
        """Compress an array."""
        original = np.asarray(array)
        dtype = str(original.dtype)
        shape = tuple(int(dim) for dim in original.shape)
        if _cuda_enabled_for(self.n_bits, self.use_cuda):
            try:
                return self._forward_cuda(original, dtype, shape)
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
        original_flat = original.astype(np.float32, copy=False).reshape(-1)
        flat = original_flat
        numel = int(flat.size)
        pad = (-numel) % self.block_size
        if pad:
            flat = np.pad(flat, (0, pad))
        blocks = flat.reshape(-1, self.block_size)
        if self.rotation:
            blocks = _hadamard_rotate(
                blocks, _rotation_signs(self.block_size, self.rotation_seed)
            )
        normalization_scales = np.sqrt(np.mean(blocks * blocks, axis=1))
        safe_normalization_scales = np.where(
            normalization_scales > 0, normalization_scales, 1.0
        )
        standardized = blocks / safe_normalization_scales[:, None]
        centroids, boundaries = _normal_codebook(self.n_bits)
        indices = np.searchsorted(boundaries, standardized, side="right").astype(
            np.uint8
        )
        unscaled_reconstruction = centroids[indices.astype(np.int64)]
        if self.scale_mode == "unbiased":
            numerator = np.sum(blocks * blocks, axis=1, dtype=np.float64)
            denominator = np.sum(
                unscaled_reconstruction * blocks, axis=1, dtype=np.float64
            )
            scales_float = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=np.abs(denominator) > 1e-30,
            ).astype(np.float32)
        else:
            scales_float = normalization_scales
        scales = scales_float.astype(np.float16)
        packed_indices = _pack_bits(indices.reshape(-1), self.n_bits)
        scales_payload = scales.tobytes()
        reconstructed_blocks = unscaled_reconstruction * scales.astype(np.float32)[:, None]
        if self.rotation:
            reconstructed_blocks = _hadamard_unrotate(
                reconstructed_blocks,
                _rotation_signs(self.block_size, self.rotation_seed),
            )
        reconstructed = reconstructed_blocks.reshape(-1)[:numel]
        error = reconstructed - original_flat
        metadata = {
            "dtype": dtype,
            "shape": shape,
            "numel": numel,
            "n_bits": self.n_bits,
            "block_size": self.block_size,
            "padded_numel": int(indices.size),
            "q_nbytes": len(packed_indices),
            "scale_nbytes": len(scales_payload),
            "cuda_requested": self.use_cuda,
            "cuda_used": False,
            "rotation": self.rotation,
            "rotation_seed": self.rotation_seed,
            "scale_mode": self.scale_mode,
            "squared_error": float(np.sum(error * error, dtype=np.float64)),
            "squared_norm": float(
                np.sum(original_flat * original_flat, dtype=np.float64)
            ),
            "reconstructed_squared_norm": float(
                np.sum(reconstructed * reconstructed, dtype=np.float64)
            ),
            "dot_product": float(
                np.sum(original_flat * reconstructed, dtype=np.float64)
            ),
        }
        return packed_indices + scales_payload, metadata

    def backward(self, payload: bytes, metadata: dict[str, Any]) -> NDArray:
        """Decompress an array."""
        n_bits = int(metadata["n_bits"])
        block_size = int(metadata["block_size"])
        numel = int(metadata["numel"])
        padded_numel = int(metadata["padded_numel"])
        q_nbytes = int(metadata["q_nbytes"])
        scale_nbytes = int(metadata["scale_nbytes"])
        if bool(metadata.get("cuda_used", False)) and _cuda_enabled_for(
            n_bits, self.use_cuda
        ):
            try:
                return self._backward_cuda(
                    payload,
                    metadata,
                    q_nbytes=q_nbytes,
                    scale_nbytes=scale_nbytes,
                    padded_numel=padded_numel,
                    numel=numel,
                )
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
        indices = _unpack_bits(payload[:q_nbytes], n_bits, padded_numel)
        scales = np.frombuffer(
            payload[q_nbytes : q_nbytes + scale_nbytes], dtype=np.float16
        ).astype(np.float32)
        centroids, _ = _normal_codebook(n_bits)
        values = centroids[indices.astype(np.int64)].reshape(-1, block_size)
        restored_blocks = values * scales[:, None]
        if bool(metadata.get("rotation", False)):
            restored_blocks = _hadamard_unrotate(
                restored_blocks,
                _rotation_signs(block_size, int(metadata.get("rotation_seed", 2025))),
            )
        restored = restored_blocks.reshape(-1)[:numel]
        return restored.reshape(tuple(metadata["shape"])).astype(str(metadata["dtype"]))

    def _forward_cuda(
        self, original: NDArray, dtype: str, shape: tuple[int, ...]
    ) -> tuple[bytes, dict[str, Any]]:
        """Compress with CUDA tensor ops and transfer only the packed payload to CPU."""
        import torch

        from .cuda import quantize_mse_cuda

        try:
            tensor = torch.from_numpy(original.astype(np.float32, copy=False)).to(
                "cuda"
            )
            packed, scales, _, padded_numel, numel, distortion = quantize_mse_cuda(
                tensor,
                bits=self.n_bits,
                block_size=self.block_size,
                rotation=self.rotation,
                rotation_seed=self.rotation_seed,
                scale_mode=self.scale_mode,
            )
            torch.cuda.synchronize()
            packed_payload = packed.cpu().numpy().tobytes()
            scales_payload = scales.cpu().numpy().tobytes()
            metadata = {
                "dtype": dtype,
                "shape": shape,
                "numel": int(numel),
                "n_bits": self.n_bits,
                "block_size": self.block_size,
                "padded_numel": int(padded_numel),
                "q_nbytes": len(packed_payload),
                "scale_nbytes": len(scales_payload),
                "cuda_requested": True,
                "cuda_used": True,
                "rotation": self.rotation,
                "rotation_seed": self.rotation_seed,
                "scale_mode": self.scale_mode,
                **distortion,
            }
            return packed_payload + scales_payload, metadata
        finally:
            torch.cuda.empty_cache()

    def _backward_cuda(
        self,
        payload: bytes,
        metadata: dict[str, Any],
        *,
        q_nbytes: int,
        scale_nbytes: int,
        padded_numel: int,
        numel: int,
    ) -> NDArray:
        """Decompress with CUDA tensor ops and return a NumPy array."""
        import torch

        from .cuda import dequantize_mse_cuda

        try:
            packed_np = np.frombuffer(payload[:q_nbytes], dtype=np.uint8).copy()
            scale_np = np.frombuffer(
                payload[q_nbytes : q_nbytes + scale_nbytes], dtype=np.float16
            ).copy()
            packed = torch.from_numpy(packed_np).to("cuda")
            scales = torch.from_numpy(scale_np).to("cuda")
            output_dtype = getattr(torch, str(metadata["dtype"]), torch.float32)
            restored = dequantize_mse_cuda(
                packed,
                scales,
                None,
                padded_numel,
                numel,
                bits=int(metadata["n_bits"]),
                output_dtype=output_dtype,
                rotation=bool(metadata.get("rotation", False)),
                rotation_seed=int(metadata.get("rotation_seed", 2025)),
            )
            torch.cuda.synchronize()
            return restored.cpu().numpy().reshape(tuple(metadata["shape"]))
        finally:
            torch.cuda.empty_cache()


class TurboQuantMSEPipeline(TransformationPipeline):
    """TurboQuant MSE pipeline."""

    def __init__(
        self,
        *,
        n_bits: int = 3,
        block_size: int = 262_144,
        use_cuda: bool = False,
        rotation: bool = True,
        rotation_seed: int = 2025,
    ) -> None:
        super().__init__(
            "turboquant_mse",
            [
                TurboQuantMSETransformer(
                    n_bits=n_bits,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    rotation=rotation,
                    rotation_seed=rotation_seed,
                    scale_mode="rms",
                )
            ],
            {
                "n_bits": n_bits,
                "block_size": block_size,
                "use_cuda": use_cuda,
                "rotation": rotation,
                "rotation_seed": rotation_seed,
                "scale_mode": "rms",
            },
        )


class EdenUnbiasedPipeline(TransformationPipeline):
    """Blockwise EDEN-style unbiased distributed-mean pipeline."""

    def __init__(
        self,
        *,
        n_bits: int = 3,
        block_size: int = 262_144,
        use_cuda: bool = False,
        rotation: bool = True,
        rotation_seed: int = 2025,
    ) -> None:
        super().__init__(
            "eden_unbiased",
            [
                TurboQuantMSETransformer(
                    n_bits=n_bits,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    rotation=rotation,
                    rotation_seed=rotation_seed,
                    scale_mode="unbiased",
                )
            ],
            {
                "n_bits": n_bits,
                "block_size": block_size,
                "use_cuda": use_cuda,
                "rotation": rotation,
                "rotation_seed": rotation_seed,
                "scale_mode": "unbiased",
            },
        )
