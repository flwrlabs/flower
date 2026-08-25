"""Tests for layerwise aggregation batching helpers."""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from flwr.common import ConfigRecord, MetricRecord, RecordDict
from flwr.common.message import Message
from flwr.common.profiling import (
    ProfileRecorder,
    clear_active_profiler,
    set_active_profiler,
)

task_stub = types.ModuleType("flowertune_llm.task")
task_stub.state_dict_fingerprint = lambda state_dict: 0.0
sys.modules.setdefault("flowertune_llm.task", task_stub)

from flowertune_llm.fedavgstreaming import (  # noqa: E402
    FedAvgStreaming,
    _batch_entries_by_size,
    _build_layer_chunk_entries,
    _downstream_duration_from_message,
    _record_network_profile,
    _record_profile_replies,
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
            client_name="client-a",
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
    assert entries[0]["node_name"] == "client-a"
    assert entries[0]["sender_node_name"] == "client-a"
    assert entries[0]["receiver_node_name"] == "server"


def test_layerwise_download_uses_standard_network_definitions() -> None:
    """Layerwise replies must report downstream, upstream, and their sum."""
    instruction = Message(RecordDict(), dst_node_id=7, message_type="train")
    reply = Message(
        RecordDict(
            {
                "metrics": MetricRecord({"profile.client.train.ms": 1.0}),
                "_flwr_profile": ConfigRecord({"client_name": "client-a"}),
            }
        ),
        reply_to=instruction,
    )
    reply.metadata.created_at = 10.0
    reply.metadata.__dict__["_network_delivered_at_ms"] = 10_500.0
    reply.metadata.__dict__["_network_upstream_bytes"] = 2 * 1024
    reply.metadata.__dict__["_network_downstream_bytes"] = 1024
    reply.metadata.__dict__["_network_downstream_ms"] = 200.0

    profiler = ProfileRecorder(run_id=1)
    set_active_profiler(profiler)
    try:
        _record_profile_replies([reply])
    finally:
        clear_active_profiler()

    entries = profiler.summarize()["entries"]
    upstream = next(entry for entry in entries if entry["task"] == "upstream")
    downstream = next(entry for entry in entries if entry["task"] == "downstream")
    combined = next(entry for entry in entries if entry["task"] == "combined")
    assert upstream["avg_ms"] == 500.0
    assert downstream["avg_ms"] == 200.0
    assert combined["avg_ms"] == 700.0
    assert combined["avg_ms"] == upstream["avg_ms"] + downstream["avg_ms"]
    assert downstream["total_network_mb"] == 1024.0 / (1024.0**2)
    assert upstream["total_network_mb"] == 2 * 1024.0 / (1024.0**2)
    assert combined["total_network_mb"] == 3 * 1024.0 / (1024.0**2)


def test_streamed_reply_reads_downstream_sidecar_from_inline_content() -> None:
    """Raw streamed replies must read delivery timing from their protobuf content."""
    reply = Message(
        RecordDict(
            {
                "_flwr_network_delivery": MetricRecord(
                    {"downstream_ms": 321.0}
                )
            }
        ),
        dst_node_id=7,
        message_type="train",
    )

    assert _downstream_duration_from_message(reply) == 321.0


def test_streamed_upload_accepts_downstream_profile_bytes() -> None:
    """The streamed upload helper accepts the payload metadata supplied by ``start``."""
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
        client_names_by_node_id={},
    )


def test_streamed_upload_timeout_rejects_partial_aggregation() -> None:
    """A missing streamed reply must abort before applying any update."""
    strategy = FedAvgStreaming(initial_state_dict={})
    grid = MagicMock()
    grid._run.run_id = 1  # pylint: disable=protected-access

    with pytest.raises(TimeoutError, match="No partial client update was applied"):
        strategy._aggregate_streamed_upload_replies(  # pylint: disable=protected-access
            grid=grid,
            msg_ids=["missing-reply"],
            batch_idx=0,
            batch_count=1,
            batch_entries=[],
            timeout=0.0,
            process=MagicMock(),
            state_dict={},
            aggregated_layers={},
            offload_enabled=False,
            offload_dir="",
            chunk_count_by_layer={},
            layer_names=[],
            downstream_bytes_by_id={},
            client_names_by_node_id={},
        )


def test_apply_aggregated_upload_chunk_has_no_network_arguments() -> None:
    """Applying an aggregate remains independent of transport profiling."""
    strategy = FedAvgStreaming(initial_state_dict={"layer": torch.zeros(2)})
    state_dict = {"layer": torch.zeros(2)}
    strategy._apply_aggregated_upload_chunk(  # pylint: disable=protected-access
        entry={
            "layer_idx": 0,
            "layer_name": "layer",
            "start": 0,
            "end": 2,
            "is_last_chunk": True,
        },
        chunk_tensor=torch.ones(2),
        state_dict=state_dict,
        aggregated_layers={},
        offload_enabled=False,
        offload_dir="",
        chunk_count_by_layer={"layer": 1},
        layer_names=["layer"],
    )

    assert torch.equal(state_dict["layer"], torch.ones(2))


def test_layerwise_download_failure_aborts_before_training() -> None:
    """A missing download acknowledgement must fail instead of being ignored."""
    strategy = FedAvgStreaming(initial_state_dict={"layer": torch.ones(2)})
    strategy._layer_names = ["layer"]  # pylint: disable=protected-access
    strategy._download_pipeline_depth = 1  # pylint: disable=protected-access
    grid = MagicMock()
    grid.send_and_receive.return_value = []

    with pytest.raises(RuntimeError, match="No training messages were sent"):
        strategy._download_layers_to_clients(  # pylint: disable=protected-access
            grid=grid,
            node_ids=[7],
            state_dict={"layer": torch.ones(2)},
            timeout=30.0,
        )
