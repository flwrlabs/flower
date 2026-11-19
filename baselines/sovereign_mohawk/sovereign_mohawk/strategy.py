"""Verification helpers for the sovereign_mohawk baseline strategy."""

from dataclasses import asdict, dataclass

import torch
from flwr.app import ArrayRecord


@dataclass
class VerificationReport:
    """Summary of the lightweight verification pass over model tensors."""

    enabled: bool
    status: str
    total_tensors: int
    non_finite_tensors: int
    max_abs_value: float
    l2_norm: float


def verify_arrayrecord(arrays: ArrayRecord, enabled: bool) -> VerificationReport:
    """Run lightweight numerical checks over aggregated model tensors.

    This hook is intentionally simple: it checks for finite values and computes
    aggregate statistics that can be logged or persisted as baseline evidence.
    """
    if not enabled:
        return VerificationReport(
            enabled=False,
            status="skipped",
            total_tensors=0,
            non_finite_tensors=0,
            max_abs_value=0.0,
            l2_norm=0.0,
        )

    state_dict = arrays.to_torch_state_dict()

    total_tensors = 0
    non_finite_tensors = 0
    max_abs_value = 0.0
    sum_squared_l2 = 0.0

    for value in state_dict.values():
        tensor = value.detach().float()
        total_tensors += 1

        finite_mask = torch.isfinite(tensor)
        if not bool(finite_mask.all()):
            non_finite_tensors += 1

        if tensor.numel() == 0:
            continue

        finite_tensor = tensor[finite_mask]
        if finite_tensor.numel() > 0:
            local_max_abs = float(finite_tensor.abs().max().item())
            if local_max_abs > max_abs_value:
                max_abs_value = local_max_abs
            sum_squared_l2 += float(torch.sum(finite_tensor * finite_tensor).item())

    status = "passed" if non_finite_tensors == 0 else "failed"
    return VerificationReport(
        enabled=True,
        status=status,
        total_tensors=total_tensors,
        non_finite_tensors=non_finite_tensors,
        max_abs_value=max_abs_value,
        l2_norm=sum_squared_l2**0.5,
    )


def verification_report_to_dict(report: VerificationReport) -> dict[str, object]:
    """Serialize verification report for logs and artifacts."""
    return asdict(report)
