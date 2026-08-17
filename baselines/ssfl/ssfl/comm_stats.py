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
        """Return counters in a metrics-friendly mapping."""
        return {
            "discovery_downlink_payload_bytes": float(
                self.discovery_downlink_payload_bytes
            ),
            "discovery_uplink_payload_bytes": float(
                self.discovery_uplink_payload_bytes
            ),
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
        """Format the communication counters for logging."""
        d = self.as_dict()
        discovery_downlink = int(d["discovery_downlink_payload_bytes"])
        discovery_uplink = int(d["discovery_uplink_payload_bytes"])
        mask_downlink = int(d["mask_downlink_payload_bytes"])
        train_uplink = int(d["train_uplink_payload_bytes"])
        one_time_total = int(d["one_time_discovery_and_mask_payload_bytes"])
        train_params = int(d["train_comm_params"])
        lines = [
            "Communication summary (ArrayRecord payload bytes; "
            "excludes Message framing):",
            f"  discovery downlink: {discovery_downlink:,}",
            f"  discovery uplink:   {discovery_uplink:,}",
            f"  mask downlink:      {mask_downlink:,}",
            f"  train uplink:       {train_uplink:,}",
            f"  one-time total:     {one_time_total:,}",
            f"  train nonzero params (sum over replies): {train_params:,}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines
