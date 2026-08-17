"""Communication accounting helpers for discovery, mask install, and training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommStats:
    """Byte/param counters. Payload bytes are ArrayRecord payloads, not wire bytes."""

    discovery_downlink_payload_bytes: int = 0
    discovery_uplink_payload_bytes: int = 0
    mask_downlink_payload_bytes: int = 0
    train_uplink_payload_bytes: int = 0
    train_comm_params: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "discovery_downlink_payload_bytes": float(
                self.discovery_downlink_payload_bytes
            ),
            "discovery_uplink_payload_bytes": float(self.discovery_uplink_payload_bytes),
            "mask_downlink_payload_bytes": float(self.mask_downlink_payload_bytes),
            "train_uplink_payload_bytes": float(self.train_uplink_payload_bytes),
            "train_comm_params": float(self.train_comm_params),
            "one_time_discovery_and_mask_payload_bytes": float(
                self.discovery_downlink_payload_bytes
                + self.discovery_uplink_payload_bytes
                + self.mask_downlink_payload_bytes
            ),
        }

    def summary_lines(self) -> list[str]:
        d = self.as_dict()
        lines = [
            "Communication summary (ArrayRecord payload bytes; excludes Message framing):",
            f"  discovery downlink: {int(d['discovery_downlink_payload_bytes']):,}",
            f"  discovery uplink:   {int(d['discovery_uplink_payload_bytes']):,}",
            f"  mask downlink:      {int(d['mask_downlink_payload_bytes']):,}",
            f"  train uplink:       {int(d['train_uplink_payload_bytes']):,}",
            f"  one-time total:     {int(d['one_time_discovery_and_mask_payload_bytes']):,}",
            f"  train nonzero params (sum over replies): {int(d['train_comm_params']):,}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines
