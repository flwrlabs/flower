"""Tests for communication accounting helpers."""

from __future__ import annotations

import json

from ssfl.comm_stats import CommStats
from ssfl.server_app import _append_jsonl, _reset_jsonl
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


def test_metrics_jsonl_is_replaced_on_reset(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _append_jsonl(path, {"event": "old"})
    _append_jsonl(path, {"event": "also-old"})
    _reset_jsonl(path)
    _append_jsonl(path, {"event": "new"})
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "new"
