# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Optional CUDA helpers for TurboQuant.

Flower does not require PyTorch or CUDA to use compression. These helpers are
import-safe in non-PyTorch environments and import PyTorch lazily only when a CUDA
function is called.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def is_cuda_available() -> bool:
    """Return True if PyTorch CUDA is available."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def describe_cuda_path() -> dict[str, Any]:
    """Describe the optional CUDA acceleration path."""
    if not is_cuda_available():
        return {"available": False}
    import torch

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "summary": (
            "CUDA compression keeps deltas on device, performs block RMS "
            "normalization, centroid lookup, 3-bit packing, payload transfer, "
            "unpacking, and dequantization without materializing the full model."
        ),
    }


@lru_cache(maxsize=16)
def _normal_codebook_cpu(bits: int):  # type: ignore[no-untyped-def]
    import torch

    levels = 1 << bits
    gen = torch.Generator(device="cpu").manual_seed(1000 + bits)
    sample = torch.randn(250_000, generator=gen)
    centroids = torch.quantile(sample, torch.linspace(0, 1, levels + 2)[1:-1])
    sample, _ = torch.sort(sample)
    for _ in range(30):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        idx = torch.bucketize(sample, boundaries)
        updated = centroids.clone()
        for level in range(levels):
            values = sample[idx == level]
            if values.numel() > 0:
                updated[level] = values.mean()
        if torch.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return centroids.contiguous(), ((centroids[:-1] + centroids[1:]) * 0.5).contiguous()


def _normal_codebook(bits: int, device):  # type: ignore[no-untyped-def]
    centroids, boundaries = _normal_codebook_cpu(bits)
    return centroids.to(device=device), boundaries.to(device=device)


def _hadamard_rotate(blocks, signs):  # type: ignore[no-untyped-def]
    """Apply an orthonormal randomized Walsh-Hadamard transform to rows."""
    import math

    width = blocks.shape[1]
    rotated = blocks * signs.view(1, -1)
    stride = 1
    while stride < width:
        grouped = rotated.view(-1, width // (2 * stride), 2, stride)
        left = grouped[:, :, 0, :].clone()
        right = grouped[:, :, 1, :].clone()
        grouped[:, :, 0, :] = left + right
        grouped[:, :, 1, :] = left - right
        stride *= 2
    return rotated * (1.0 / math.sqrt(width))


def _rotation_signs(block_size: int, seed: int, device):  # type: ignore[no-untyped-def]
    """Build the same deterministic signs used by the NumPy implementation."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), block_size)
    return torch.from_numpy(signs).to(device=device)


def pack_3bit(values):  # type: ignore[no-untyped-def]
    """Pack uint8 values in [0, 7] into a CUDA/CPU 3-bit byte tensor."""
    import torch
    import torch.nn.functional as F

    flat = values.flatten().to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad))
    vals = flat.view(-1, 8).to(torch.int16)
    out = torch.empty((vals.shape[0], 3), device=flat.device, dtype=torch.uint8)
    out[:, 0] = (vals[:, 0] | (vals[:, 1] << 3) | (vals[:, 2] << 6)).to(torch.uint8)
    out[:, 1] = (
        (vals[:, 2] >> 2) | (vals[:, 3] << 1) | (vals[:, 4] << 4) | (vals[:, 5] << 7)
    ).to(torch.uint8)
    out[:, 2] = ((vals[:, 5] >> 1) | (vals[:, 6] << 2) | (vals[:, 7] << 5)).to(
        torch.uint8
    )
    return out.flatten()


def unpack_3bit(packed, count: int):  # type: ignore[no-untyped-def]
    """Unpack a 3-bit byte tensor into uint8 values in [0, 7]."""
    import torch

    bytes3 = packed.view(-1, 3).to(torch.int16)
    out = torch.empty((bytes3.shape[0], 8), device=packed.device, dtype=torch.uint8)
    b0 = bytes3[:, 0]
    b1 = bytes3[:, 1]
    b2 = bytes3[:, 2]
    out[:, 0] = (b0 & 0x07).to(torch.uint8)
    out[:, 1] = ((b0 >> 3) & 0x07).to(torch.uint8)
    out[:, 2] = (((b0 >> 6) | (b1 << 2)) & 0x07).to(torch.uint8)
    out[:, 3] = ((b1 >> 1) & 0x07).to(torch.uint8)
    out[:, 4] = ((b1 >> 4) & 0x07).to(torch.uint8)
    out[:, 5] = (((b1 >> 7) | (b2 << 1)) & 0x07).to(torch.uint8)
    out[:, 6] = ((b2 >> 2) & 0x07).to(torch.uint8)
    out[:, 7] = ((b2 >> 5) & 0x07).to(torch.uint8)
    return out.flatten()[:count]


def pack_4bit(values):  # type: ignore[no-untyped-def]
    """Pack uint8 values in [0, 15] into a CUDA/CPU 4-bit byte tensor."""
    import torch
    import torch.nn.functional as F

    flat = values.flatten().to(torch.uint8)
    if flat.numel() % 2:
        flat = F.pad(flat, (0, 1))
    vals = flat.view(-1, 2).to(torch.int16)
    return (vals[:, 0] | (vals[:, 1] << 4)).to(torch.uint8)


def unpack_4bit(packed, count: int):  # type: ignore[no-untyped-def]
    """Unpack a 4-bit byte tensor into uint8 values in [0, 15]."""
    import torch

    raw = packed.flatten().to(torch.int16)
    out = torch.empty((raw.numel(), 2), device=packed.device, dtype=torch.uint8)
    out[:, 0] = (raw & 0x0F).to(torch.uint8)
    out[:, 1] = (raw >> 4).to(torch.uint8)
    return out.flatten()[:count]


def quantize_mse_cuda(  # type: ignore[no-untyped-def]
    delta,
    bits: int = 3,
    block_size: int = 262_144,
    rotation: bool = True,
    rotation_seed: int = 2025,
    scale_mode: str = "rms",
):
    """Quantize a CUDA tensor with TurboQuant MSE primitives."""
    import torch
    import torch.nn.functional as F

    if not delta.is_cuda:
        raise ValueError("quantize_mse_cuda expects a CUDA tensor")
    numel = delta.numel()
    flat = delta.float().flatten()
    pad = (-numel) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    signs = None
    if rotation:
        signs = _rotation_signs(block_size, rotation_seed, blocks.device)
        blocks = _hadamard_rotate(blocks, signs)
    normalization_scale = blocks.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
    standardized = blocks / normalization_scale[:, None]
    centroids, boundaries = _normal_codebook(bits, standardized.device)
    indices = torch.bucketize(standardized, boundaries).to(torch.uint8)
    if bits == 3:
        packed = pack_3bit(indices)
    elif bits == 4:
        packed = pack_4bit(indices)
    else:
        packed = indices.flatten()
    unscaled_reconstruction = centroids[indices.long()]
    if scale_mode == "unbiased":
        numerator = torch.sum(blocks * blocks, dim=1)
        denominator = torch.sum(unscaled_reconstruction * blocks, dim=1)
        scale = torch.where(
            denominator.abs() > 1e-30,
            numerator / denominator,
            torch.zeros_like(numerator),
        )
    elif scale_mode == "rms":
        scale = normalization_scale
    else:
        raise ValueError("scale_mode must be 'rms' or 'unbiased'")
    reconstructed = unscaled_reconstruction * scale[:, None]
    if rotation:
        assert signs is not None
        reconstructed = _hadamard_rotate(
            reconstructed, reconstructed.new_ones(block_size)
        ) * signs
    reconstructed = reconstructed.flatten()[:numel]
    original = delta.float().flatten()
    error = reconstructed - original
    distortion = {
        "squared_error": float(torch.sum(error * error).item()),
        "squared_norm": float(torch.sum(original * original).item()),
        "reconstructed_squared_norm": float(
            torch.sum(reconstructed * reconstructed).item()
        ),
        "dot_product": float(torch.sum(original * reconstructed).item()),
    }
    return (
        packed,
        scale.to(torch.float16),
        centroids,
        indices.numel(),
        numel,
        distortion,
    )


def dequantize_mse_cuda(
    packed,
    scale,
    centroids,
    padded_numel: int,
    numel: int,
    bits: int = 3,
    output_dtype=None,  # type: ignore[no-untyped-def]
    rotation: bool = True,
    rotation_seed: int = 2025,
):  # type: ignore[no-untyped-def]
    """Dequantize CUDA TurboQuant MSE payloads."""
    import torch

    if centroids is None:
        centroids, _ = _normal_codebook(bits, packed.device)
    if bits == 3:
        indices = unpack_3bit(packed, padded_numel)
    elif bits == 4:
        indices = unpack_4bit(packed, padded_numel)
    else:
        indices = packed.flatten()[:padded_numel]
    block_size = padded_numel // scale.numel()
    dtype = output_dtype or torch.float32
    values = centroids[indices.long()].to(dtype=dtype).view(-1, block_size)
    restored = values * scale.to(dtype=dtype)[:, None]
    if rotation:
        signs = _rotation_signs(block_size, rotation_seed, restored.device)
        restored = _hadamard_rotate(restored, signs.new_ones(block_size)) * signs
    return restored.flatten()[:numel]
