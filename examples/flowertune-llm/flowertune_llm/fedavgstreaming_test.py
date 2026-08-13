"""Tests for layerwise aggregation batching helpers."""

import sys
import types
from unittest.mock import MagicMock

import torch

from flwr.common.profiling import (
    ProfileRecorder,
    clear_active_profiler,
    set_active_profiler,
)

task_stub = types.ModuleType("flowertune_llm.task")
task_stub.state_dict_fingerprint = lambda state_dict: 0.0
sys.modules.setdefault("flowertune_llm.task", task_stub)

from flowertune_llm.fedavgstreaming import (
    FedAvgStreaming,
    _batch_entries_by_size,
    _build_layer_chunk_entries,
    _record_network_profile,
)


def test_batch_entries_by_size_groups_multiple_layers() -> None:
    """Small layers should share a message until the byte budget is full."""
    state_dict = {
        "a": torch.zeros(2, dtype=torch.float32),
        "b": torch.zeros(2, dtype=torch.float32),
        "c": torch.zeros(3, dtype=torch.float32),
    }
    entries = _build_layer_chunk_entries(list(state_dict), state_dict, 64)

    batches = _batch_entries_by_size(
        entries,
        max_batch_bytes=16,
        max_chunks_per_message=0,
    )

    assert [[entry["layer_name"] for entry in batch] for batch in batches] == [
        ["a", "b"],
        ["c"],
    ]


def test_batch_entries_by_size_honors_explicit_chunk_cap() -> None:
    """The chunk cap is opt-in and can force one chunk per message."""
    state_dict = {
        "a": torch.zeros(2, dtype=torch.float32),
        "b": torch.zeros(2, dtype=torch.float32),
        "c": torch.zeros(2, dtype=torch.float32),
    }
    entries = _build_layer_chunk_entries(list(state_dict), state_dict, 64)

    batches = _batch_entries_by_size(
        entries,
        max_batch_bytes=64,
        max_chunks_per_message=1,
    )

    assert len(batches) == len(entries)
    assert all(len(batch) == 1 for batch in batches)


def test_record_network_profile_preserves_endpoint_and_bytes() -> None:
    """Custom layerwise transfers should use the standard network profile schema."""
    profiler = ProfileRecorder(run_id=1)
    set_active_profiler(profiler)
    try:
        _record_network_profile(
            "upstream",
            12.5,
            node_id=7,
            sender_node_id=7,
            receiver_node_id="server",
            network_bytes=3 * 1024 * 1024,
        )
    finally:
        clear_active_profiler()

    entries = profiler.summarize()["entries"]
    assert len(entries) == 1
    assert entries[0]["scope"] == "network"
    assert entries[0]["task"] == "upstream"
    assert entries[0]["total_network_mb"] == 3.0
    assert entries[0]["sender_node_id"] == 7
    assert entries[0]["receiver_node_id"] == "server"


def test_streamed_upload_accepts_downstream_profile_arguments() -> None:
    """The streamed upload helper accepts the metadata supplied by ``start``."""
    strategy = FedAvgStreaming(initial_state_dict={})
    strategy._aggregate_streamed_upload_replies(  # pylint: disable=protected-access
        grid=MagicMock(),
        msg_ids=[],
        batch_idx=0,
        batch_count=1,
        batch_entries=[],
        timeout=None,
        process=MagicMock(),
        state_dict={},
        aggregated_layers={},
        offload_enabled=False,
        offload_dir="",
        chunk_count_by_layer={},
        layer_names=[],
        downstream_bytes_by_id={},
        downstream_duration_ms=0.0,
    )
