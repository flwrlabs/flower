"""Tests for communication accounting helpers."""

from __future__ import annotations

from ssfl.comm_stats import CommStats
from ssfl.wandb_utils import WandbSession


def test_comm_stats_totals():
    stats = CommStats(
        discovery_downlink_payload_bytes=100,
        discovery_uplink_payload_bytes=200,
        mask_downlink_payload_bytes=50,
        train_uplink_payload_bytes=1000,
        train_comm_params=42,
    )
    d = stats.as_dict()
    assert d["one_time_discovery_and_mask_payload_bytes"] == 350.0
    lines = stats.summary_lines()
    assert any("Communication summary" in line for line in lines)


def test_wandb_disabled_is_noop():
    session = WandbSession()
    session.start({"wandb-mode": "disabled"})
    assert session.enabled is False
    session.log({"a": 1.0}, step=1)
    session.finish()
