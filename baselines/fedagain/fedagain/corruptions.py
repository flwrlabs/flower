"""fedagain: Common image corruptions used to simulate degraded clients.

This module re-implements, in plain PyTorch, a subset of the common corruptions
of Hendrycks & Dietterich (ICLR 2019) using the severity constants of their
CIFAR-10-C generation script. The paper (Sec. 5.1.4 and Table 3) uses this
family of corruptions to simulate hospitals with degraded acquisition devices.

All functions take an image tensor of shape ``(C, H, W)`` with values in
``[0, 1]`` (i.e. *before* normalization) and return a tensor of the same shape
and range. They work for grayscale (MNIST) and RGB (CIFAR-10) inputs.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

# Severity constants (severity 1..5) from the CIFAR-10-C generation script.
_GAUSSIAN_NOISE = [0.04, 0.06, 0.08, 0.09, 0.10]
_SHOT_NOISE = [500.0, 250.0, 100.0, 75.0, 50.0]
_IMPULSE_NOISE = [0.01, 0.02, 0.03, 0.05, 0.07]
_SPECKLE_NOISE = [0.06, 0.10, 0.12, 0.16, 0.20]
_GAUSSIAN_BLUR = [0.4, 0.6, 0.7, 0.8, 1.0]
_CONTRAST = [0.75, 0.5, 0.4, 0.3, 0.15]
_BRIGHTNESS = [0.05, 0.1, 0.15, 0.2, 0.3]
_PIXELATE = [0.95, 0.9, 0.85, 0.75, 0.65]


def gaussian_noise(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Additive white Gaussian noise (sensor thermal noise)."""
    sigma = _GAUSSIAN_NOISE[severity - 1]
    return x + torch.randn(x.shape, generator=gen) * sigma


def shot_noise(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Poisson photon noise (low-light acquisition)."""
    lam = _SHOT_NOISE[severity - 1]
    return torch.poisson(x * lam, generator=gen) / lam


def impulse_noise(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Salt-and-pepper noise (transmission or sensor read-out errors)."""
    amount = _IMPULSE_NOISE[severity - 1]
    u = torch.rand(x.shape, generator=gen)
    out = x.clone()
    out[u < amount / 2] = 0.0
    out[u > 1 - amount / 2] = 1.0
    return out


def speckle_noise(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Multiplicative noise (coherent imaging interference)."""
    sigma = _SPECKLE_NOISE[severity - 1]
    return x + x * torch.randn(x.shape, generator=gen) * sigma


def gaussian_blur(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Isotropic Gaussian blur (out-of-focus optics)."""
    del gen  # deterministic corruption
    sigma = _GAUSSIAN_BLUR[severity - 1]
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = x.shape[0]
    img = x.unsqueeze(0)
    k_h = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    k_v = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    img = F.pad(img, (radius, radius, 0, 0), mode="reflect")
    img = F.conv2d(img, k_h, groups=channels)  # pylint: disable=not-callable
    img = F.pad(img, (0, 0, radius, radius), mode="reflect")
    img = F.conv2d(img, k_v, groups=channels)  # pylint: disable=not-callable
    return img.squeeze(0)


def contrast(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Contrast reduction (illumination drift, haze)."""
    del gen
    c = _CONTRAST[severity - 1]
    means = x.mean(dim=(-2, -1), keepdim=True)
    return (x - means) * c + means


def brightness(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Global brightness shift (over-exposure)."""
    del gen
    return x + _BRIGHTNESS[severity - 1]


def pixelate(x: torch.Tensor, severity: int, gen: torch.Generator):
    """Down- and up-sampling (low-resolution sensors, compression)."""
    del gen
    c = _PIXELATE[severity - 1]
    height, width = x.shape[-2:]
    small = (max(1, int(height * c)), max(1, int(width * c)))
    img = F.interpolate(x.unsqueeze(0), size=small, mode="area")
    img = F.interpolate(img, size=(height, width), mode="nearest")
    return img.squeeze(0)


CORRUPTIONS: dict[str, Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]]
CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur,
    "contrast": contrast,
    "brightness": brightness,
    "pixelate": pixelate,
}


def parse_corruption_types(spec: str) -> list[str]:
    """Parse the ``corruption-types`` run-config value into a list of names."""
    spec = spec.strip()
    if spec in ("", "all"):
        return list(CORRUPTIONS)
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in CORRUPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown corruption(s) {unknown}. Available: {list(CORRUPTIONS)}"
        )
    return names


def apply_corruption(
    x: torch.Tensor, name: str, severity: int, gen: torch.Generator
) -> torch.Tensor:
    """Apply corruption ``name`` to an image in ``[0, 1]`` and clip the result."""
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in [1, 5], got {severity}")
    return CORRUPTIONS[name](x, severity, gen).clamp(0.0, 1.0)
