"""Tests for communication accounting helpers."""

from __future__ import annotations

import json
import sys

import torch

from flwr.app import ArrayRecord, ConfigRecord
from ssfl.comm_stats import CommStats
from ssfl.server_app import _append_jsonl, _reset_jsonl
from ssfl.strategy import SSFLStrategy
from ssfl.wandb_utils import WandbSession


def test_comm_stats_totals():
    stats = CommStats(
        discovery_downlink_payload_bytes=100,
        discovery_uplink_payload_bytes=200,
        mask_downlink_payload_bytes=50,
        train_downlink_payload_bytes=4000,
        train_uplink_payload_bytes=1000,
        train_comm_params=42,
    )
    d = stats.as_dict()
    assert d["one_time_discovery_and_mask_payload_bytes"] == 350.0
    assert d["train_downlink_payload_bytes"] == 4000.0
    assert d["train_payload_bytes"] == 5000.0
    assert d["total_payload_bytes"] == 5350.0
    lines = stats.summary_lines()
    assert any("Communication summary" in line for line in lines)
    assert any("train downlink:" in line for line in lines)


class _FakeGrid:
    def get_node_ids(self) -> list[int]:
        return [10, 20, 30]


def test_configure_train_counts_downlink_once_per_destination():
    arrays = ArrayRecord({"w": torch.ones(4)})
    strategy = SSFLStrategy(
        node_to_client_id={10: 0, 20: 1, 30: 2},
        sample_seed=0,
        transport="dense",
        fraction_train=1.0,
        min_train_nodes=1,
    )
    messages = list(
        strategy.configure_train(1, arrays, ConfigRecord({}), _FakeGrid())  # type: ignore[arg-type]
    )
    expected = int(arrays.count_bytes()) * 3
    assert len(messages) == 3
    assert strategy.train_downlink_payload_bytes == expected
    assert strategy.train_downlink_by_round[1] == expected


def test_wandb_disabled_is_noop():
    session = WandbSession()
    session.start({"wandb-mode": "disabled"})
    assert session.enabled is False
    session.log({"a": 1.0}, step=1, commit=False)
    session.finish()


def test_wandb_log_forwards_commit(monkeypatch):
    calls: list[tuple[dict, dict]] = []

    class FakeWandb:
        @staticmethod
        def log(metrics, **kwargs):
            calls.append((metrics, kwargs))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    session = WandbSession()
    session.enabled = True
    session._run = object()
    session.log({"a": 1.0}, step=0, commit=False)
    assert calls == [({"a": 1.0}, {"step": 0, "commit": False})]


def test_metrics_jsonl_is_replaced_on_reset(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _append_jsonl(path, {"event": "old"})
    _append_jsonl(path, {"event": "also-old"})
    _reset_jsonl(path)
    _append_jsonl(path, {"event": "new"})
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "new"
