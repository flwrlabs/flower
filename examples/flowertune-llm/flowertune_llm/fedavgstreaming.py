# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower message-based FedAvg strategy with layer-wise aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import gc
import io
import json
from logging import INFO, WARNING
import os
import re
from time import perf_counter
import time
from typing import Any

import torch
import psutil

from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    GRPC_MAX_MESSAGE_LENGTH,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.common.inflatable import (
    get_object_body,
    get_object_children_ids_from_object_content,
    iterate_object_tree,
)
from flwr.common.inflatable_protobuf_utils import make_pull_object_fn_protobuf
from flwr.common.inflatable_utils import pull_objects
from flwr.common.profiling import (
    client_name_from_message,
    get_active_profiler,
    get_current_round,
    publish_profile_summary,
    record_network_delivery_metrics_from_messages,
    record_profile_metrics_from_messages,
    set_current_round,
)
from flwr.common.record.arraychunk import ArrayChunk
from flwr.common.serde import message_from_proto
from flwr.proto.appio_pb2 import PullAppMessagesRequest
from flwr.proto.message_pb2 import ConfirmMessageReceivedRequest
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    log_strategy_start_info,
    sample_nodes,
)

from flowertune_llm.task import state_dict_fingerprint

SAFE_GRPC_BYTES = int(GRPC_MAX_MESSAGE_LENGTH * 0.75)
DEFAULT_LAYERWISE_CHUNKS_PER_MESSAGE = 0
DEFAULT_LAYERWISE_DOWNLOAD_PIPELINE_DEPTH = 4


def _chunk_slices(tensor: torch.Tensor, max_bytes: int) -> list[tuple[int, int]]:
    """Return (start, end) slices along dim0 to fit under max_bytes."""
    if tensor.ndim == 0:
        return [(0, 1)]
    total_bytes = tensor.numel() * tensor.element_size()
    if total_bytes <= max_bytes:
        return [(0, tensor.shape[0])]
    bytes_per_row = tensor[0:1].numel() * tensor.element_size()
    rows_per_chunk = max(1, int(max_bytes // max(bytes_per_row, 1)))
    slices: list[tuple[int, int]] = []
    start = 0
    while start < tensor.shape[0]:
        end = min(start + rows_per_chunk, tensor.shape[0])
        slices.append((start, end))
        start = end
    return slices


def _chunk_key(layer_name: str, start: int, end: int) -> str:
    """Build a deterministic key for a layer chunk."""
    return f"{layer_name}::chunk_{start}_{end}"


def _shape_to_text(shape: list[int]) -> str:
    if not shape:
        return ""
    return ",".join(str(dim) for dim in shape)


def _split_replies(replies: Iterable[Message]) -> tuple[list[Message], list[Message]]:
    """Split replies into valid and error lists without strategy-level logging."""
    valid: list[Message] = []
    errors: list[Message] = []
    for msg in replies:
        if msg.has_error():
            errors.append(msg)
        else:
            valid.append(msg)
    return valid, errors


def _has_streaming_object_pull(grid: Grid) -> bool:
    """Return whether this Grid exposes the private object pull RPCs."""
    return all(
        hasattr(grid, attr)
        for attr in ("_stub", "node", "_run")  # pylint: disable=protected-access
    )


def _json_body(object_content: bytes, cls: type[Any]) -> dict[str, Any]:
    """Parse a JSON body from a deflated Flower object."""
    return json.loads(get_object_body(object_content, cls).decode("utf-8"))


@dataclass
class _StreamedReply:
    """Metadata needed to pull one client's layer chunks incrementally."""

    object_id: str
    reply_to_message_id: str
    node_id: int
    owner_key: str
    object_ids: set[str]
    message_bytes: int
    created_at_ms: float
    delivered_at_ms: float
    downstream_ms: float | None
    array_refs: dict[str, str]
    metrics: MetricRecord
    client_name: str | None = None
    network_bytes: int = 0


class _StreamObjectReader:
    """Pull deflated reply objects and track the time spent doing it."""

    def __init__(self, grid: Grid, run_id: int) -> None:
        self._stub = getattr(grid, "_stub")
        self._node = getattr(grid, "node")
        self._run_id = run_id
        self._pull_object = make_pull_object_fn_protobuf(
            self._stub.PullObject, self._node, run_id
        )
        self.pull_ms = 0.0
        self._pull_ms_by_owner: dict[str, float] = {}
        self._object_sizes: dict[str, int] = {}

    def pull(self, object_id: str, *, owner_key: str | None = None) -> bytes:
        """Pull one deflated object through the Grid's private object RPC."""
        t0 = perf_counter()
        content = pull_objects([object_id], self._pull_object)[object_id]
        elapsed_ms = (perf_counter() - t0) * 1000.0
        self.pull_ms += elapsed_ms
        self._object_sizes[object_id] = len(content)
        if owner_key is not None:
            self._pull_ms_by_owner[owner_key] = (
                self._pull_ms_by_owner.get(owner_key, 0.0) + elapsed_ms
            )
        return content

    def pull_ms_for(self, owner_key: str) -> float:
        """Return the object-pull time attributed to one reply."""
        return self._pull_ms_by_owner.get(owner_key, 0.0)

    def bytes_for(self, object_ids: set[str]) -> int:
        """Return the bytes pulled for the selected object IDs."""
        return sum(self._object_sizes.get(object_id, 0) for object_id in object_ids)

    def confirm(self, object_id: str) -> None:
        """Confirm a streamed message so SuperLink can release objects."""
        self._stub.ConfirmMessageReceived(
            ConfirmMessageReceivedRequest(
                node=self._node,
                run_id=self._run_id,
                message_object_id=object_id,
            )
        )

    def array_refs(
        self, message_object_id: str, *, owner_key: str
    ) -> tuple[dict[str, str], MetricRecord, float | None]:
        """Pull a reply's small metadata objects and return array-name refs."""
        message_content = self.pull(message_object_id, owner_key=owner_key)
        record_dict_ids = get_object_children_ids_from_object_content(message_content)
        if not record_dict_ids:
            return {}, MetricRecord(), None

        record_refs = _json_body(
            self.pull(record_dict_ids[0], owner_key=owner_key), RecordDict
        )

        metrics = MetricRecord()
        if metrics_id := record_refs.get("metrics"):
            metrics = MetricRecord.inflate(self.pull(metrics_id, owner_key=owner_key))

        downstream_ms = None
        if delivery_id := record_refs.get("_flwr_network_delivery"):
            delivery_metrics = MetricRecord.inflate(
                self.pull(delivery_id, owner_key=owner_key)
            )
            value = delivery_metrics.get("downstream_ms")
            if isinstance(value, (int, float)):
                downstream_ms = max(float(value), 0.0)

        array_refs: dict[str, str] = {}
        if arrays_id := record_refs.get("arrays"):
            array_refs = {
                str(array_name): str(array_id)
                for array_name, array_id in _json_body(
                    self.pull(arrays_id, owner_key=owner_key), ArrayRecord
                ).items()
            }

        return array_refs, metrics, downstream_ms

    def array(self, array_object_id: str, *, owner_key: str) -> Array:
        """Pull and inflate one Array from a streamed reply object tree."""
        array_content = self.pull(array_object_id, owner_key=owner_key)
        children = {
            chunk_id: ArrayChunk.inflate(self.pull(chunk_id, owner_key=owner_key))
            for chunk_id in get_object_children_ids_from_object_content(array_content)
        }
        try:
            return Array.inflate(array_content, children=children)
        finally:
            children.clear()


def _build_layer_chunk_entries(
    layer_names: list[str],
    state_dict: dict[str, Any],
    chunk_max_bytes: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for layer_idx, layer_name in enumerate(layer_names):
        tensor = state_dict[layer_name]
        cpu_tensor = tensor.detach().cpu() if hasattr(tensor, "cpu") else tensor
        chunk_slices = _chunk_slices(cpu_tensor, chunk_max_bytes)
        for chunk_idx, (start, end) in enumerate(chunk_slices):
            chunk_tensor = cpu_tensor if cpu_tensor.ndim == 0 else cpu_tensor[start:end]
            entries.append(
                {
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "layer_shape": list(cpu_tensor.shape),
                    "start": start,
                    "end": end,
                    "chunk_idx": chunk_idx,
                    "chunk_count": len(chunk_slices),
                    "is_last_chunk": chunk_idx == (len(chunk_slices) - 1),
                    "nbytes": chunk_tensor.numel() * chunk_tensor.element_size(),
                    "tensor": cpu_tensor,
                }
            )
    return entries


def _batch_entries_by_size(
    entries: list[dict[str, Any]],
    max_batch_bytes: int,
    max_chunks_per_message: int,
) -> list[list[dict[str, Any]]]:
    """Batch chunks greedily by byte budget to minimize message round trips."""
    if not entries:
        return []

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_nbytes = 0
    max_batch_bytes = max(1, int(max_batch_bytes))
    max_chunks_per_message = max(0, int(max_chunks_per_message))

    for entry in entries:
        entry_nbytes = int(entry.get("nbytes", 0))
        would_exceed_bytes = (
            current_batch and (current_nbytes + entry_nbytes) > max_batch_bytes
        )
        would_exceed_chunks = (
            max_chunks_per_message > 0
            and len(current_batch) >= max_chunks_per_message
        )
        if would_exceed_bytes or would_exceed_chunks:
            batches.append(current_batch)
            current_batch = []
            current_nbytes = 0

        current_batch.append(entry)
        current_nbytes += entry_nbytes
    if current_batch:
        batches.append(current_batch)
    return batches


def _record_profile_replies(replies: list[Message]) -> None:
    """Record reply metrics using the standard Flower network definitions."""
    profiler = get_active_profiler()
    if profiler is None or not replies:
        return
    record_network_delivery_metrics_from_messages(
        replies,
        delivered_at_ms=time.time() * 1000.0,
    )
    record_profile_metrics_from_messages(replies)
    publish_profile_summary()


def _downstream_duration_from_message(message: Message) -> float | None:
    """Read the downstream timing sidecar from a lightweight reply message."""
    if not message.has_content():
        return None
    delivery_record = message.content.metric_records.get("_flwr_network_delivery")
    if delivery_record is None:
        return None
    duration_ms = delivery_record.get("downstream_ms")
    if not isinstance(duration_ms, (int, float)):
        return None
    return max(float(duration_ms), 0.0)


def _record_server_profile(task: str, duration_ms: float, metadata: dict[str, Any]) -> None:
    profiler = get_active_profiler()
    if profiler is None:
        return
    profiler.record(
        scope="server",
        task=task,
        round=get_current_round(),
        node_id=None,
        duration_ms=duration_ms,
        metadata=metadata,
    )
    publish_profile_summary()


def _record_network_profile(
    task: str,
    duration_ms: float,
    *,
    node_id: int,
    sender_node_id: int | str | None,
    receiver_node_id: int | str | None,
    network_bytes: int,
    client_name: str | None = None,
) -> None:
    """Record a per-peer network event for custom layerwise transfers."""
    profiler = get_active_profiler()
    if profiler is None:
        return
    metadata: dict[str, Any] = {
        "sender_node_id": sender_node_id,
        "receiver_node_id": receiver_node_id,
        "network_bytes": network_bytes,
    }
    if client_name:
        metadata["node_name"] = client_name
        if sender_node_id == node_id:
            metadata["sender_node_name"] = client_name
        if receiver_node_id == node_id:
            metadata["receiver_node_name"] = client_name
    if sender_node_id == "server":
        metadata["sender_node_name"] = "server"
    if receiver_node_id == "server":
        metadata["receiver_node_name"] = "server"
    profiler.record(
        scope="network",
        task=task,
        round=get_current_round(),
        node_id=node_id,
        duration_ms=duration_ms,
        metadata=metadata,
    )


def _sanitize_layer_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _rehydrate_state_dict(layer_names: list[str], offload_dir: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(layer_names):
        file_name = f"{idx:04d}_{_sanitize_layer_name(name)}.pt"
        file_path = os.path.join(offload_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing offloaded layer: {file_path}")
        state[name] = torch.load(file_path, map_location="cpu")
    return state


def _has_meta_tensors(state_dict: dict[str, Any]) -> bool:
    for value in state_dict.values():
        if hasattr(value, "is_meta") and value.is_meta:
            return True
    return False


# pylint: disable=too-many-instance-attributes
class FedAvgStreaming(FedAvg):
    """Federated Averaging strategy with layer-wise aggregation."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        initial_state_dict: dict[str, Any] | None = None,
        max_chunk_bytes: int = SAFE_GRPC_BYTES,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self._state_dict = initial_state_dict
        self._layer_names: list[str] | None = None
        self._upload_max_chunk_bytes = max_chunk_bytes
        self._download_max_chunk_bytes = max_chunk_bytes
        self._chunks_per_message = DEFAULT_LAYERWISE_CHUNKS_PER_MESSAGE
        self._download_pipeline_depth = DEFAULT_LAYERWISE_DOWNLOAD_PIPELINE_DEPTH

    def _download_layers_to_clients(
        self,
        *,
        grid: Grid,
        node_ids: list[int],
        state_dict: dict[str, Any],
        timeout: float,
    ) -> None:
        """Stream current model layers/chunks from server to selected clients."""
        if not node_ids:
            return

        layer_names = self._layer_names or []
        max_bytes_per_layer_chunk = max(1, int(self._download_max_chunk_bytes))
        entries = _build_layer_chunk_entries(
            layer_names,
            state_dict,
            max_bytes_per_layer_chunk,
        )
        batches = _batch_entries_by_size(
            entries,
            self._download_max_chunk_bytes,
            self._chunks_per_message,
        )
        if not batches:
            return

        log(
            INFO,
            (
                "[Layer download] sending %s batches (%s chunks across %s layers, "
                "%.2f MB/message max, %.2f MB/chunk max, pipeline depth %s) "
                "to %s clients"
            ),
            len(batches),
            len(entries),
            len(layer_names),
            self._download_max_chunk_bytes / (1024 * 1024),
            max_bytes_per_layer_chunk / (1024 * 1024),
            self._download_pipeline_depth,
            len(node_ids),
        )
        progress_every = max(1, len(batches) // 10)

        def build_download_record(
            batch_idx: int, batch_entries: list[dict[str, Any]]
        ) -> RecordDict:
            arrays_dict: dict[str, torch.Tensor] = {}
            payload_layer_idxs: list[int] = []
            payload_layer_names: list[str] = []
            payload_layer_shapes: list[str] = []
            payload_chunk_starts: list[int] = []
            payload_chunk_ends: list[int] = []
            payload_is_last_chunk: list[bool] = []
            for entry in batch_entries:
                start = int(entry["start"])
                end = int(entry["end"])
                layer_idx = int(entry["layer_idx"])
                layer_name = str(entry["layer_name"])
                tensor = entry["tensor"]
                chunk_tensor = tensor if tensor.ndim == 0 else tensor[start:end]
                arrays_dict[_chunk_key(layer_name, start, end)] = chunk_tensor
                payload_layer_idxs.append(layer_idx)
                payload_layer_names.append(layer_name)
                payload_layer_shapes.append(_shape_to_text(entry["layer_shape"]))
                payload_chunk_starts.append(start)
                payload_chunk_ends.append(end)
                payload_is_last_chunk.append(bool(entry["is_last_chunk"]))

            config = ConfigRecord(
                {
                    "download_layer_idxs": payload_layer_idxs,
                    "download_layer_names": payload_layer_names,
                    "download_layer_shapes": payload_layer_shapes,
                    "download_chunk_starts": payload_chunk_starts,
                    "download_chunk_ends": payload_chunk_ends,
                    "download_is_last_chunk": payload_is_last_chunk,
                    "download_batch_idx": batch_idx,
                    "download_batch_count": len(batches),
                    "download_chunks_in_message": len(batch_entries),
                }
            )
            record = RecordDict(
                {
                    self.arrayrecord_key: ArrayRecord(arrays_dict),
                    self.configrecord_key: config,
                }
            )
            return record

        def log_download_progress(batch_idx: int) -> None:
            if (
                (batch_idx + 1) % progress_every == 0
                or batch_idx == 0
                or (batch_idx + 1) == len(batches)
            ):
                log(
                    INFO,
                    "[Layer download] progress %s/%s batches (%.1f%%)",
                    batch_idx + 1,
                    len(batches),
                    (100.0 * (batch_idx + 1)) / len(batches),
                )

        if self._download_pipeline_depth <= 1:
            for batch_idx, batch_entries in enumerate(batches):
                record = build_download_record(batch_idx, batch_entries)
                replies = grid.send_and_receive(
                    messages=self._construct_messages(
                        record,
                        node_ids,
                        message_type="train.layer_wise_download",
                    ),
                    timeout=timeout,
                )
                valid_replies, error_replies = _split_replies(replies)
                for msg in error_replies:
                    log(
                        WARNING,
                        "[Layer download] batch %s/%s error from node %s: %s",
                        batch_idx + 1,
                        len(batches),
                        msg.metadata.src_node_id,
                        msg.error.reason,
                    )
                if len(valid_replies) < len(node_ids):
                    log(
                        WARNING,
                        "Layer download ack mismatch for batch %s/%s: %s/%s",
                        batch_idx + 1,
                        len(batches),
                        len(valid_replies),
                        len(node_ids),
                    )
                log_download_progress(batch_idx)
            return

        pending: dict[str, int] = {}
        batch_pending_counts: dict[int, int] = {}
        batch_valid_counts: dict[int, int] = {}
        batch_errors: dict[int, list[Message]] = {}
        pending_downstream_bytes: dict[str, int] = {}
        next_batch_idx = 0
        completed_batches = 0
        pull_interval = float(getattr(grid, "pull_interval", 0.1))
        end_time = None if timeout is None else time.time() + timeout

        def push_more_batches() -> None:
            nonlocal next_batch_idx
            while (
                next_batch_idx < len(batches)
                and len(batch_pending_counts) < self._download_pipeline_depth
            ):
                record = build_download_record(
                    next_batch_idx, batches[next_batch_idx]
                )
                messages = list(
                    self._construct_messages(
                        record,
                        node_ids,
                        message_type="train.layer_wise_download",
                    )
                )
                push_start = perf_counter()
                msg_ids = [
                    msg_id for msg_id in grid.push_messages(messages) if msg_id
                ]
                downstream_bytes = grid.pop_pushed_message_sizes(msg_ids)
                _record_server_profile(
                    "network_downstream",
                    (perf_counter() - push_start) * 1000.0,
                    {
                        "expected_replies": len(msg_ids),
                        "batch_idx": next_batch_idx,
                        "pipeline_depth": self._download_pipeline_depth,
                        "network_bytes": sum(downstream_bytes.values()),
                    },
                )
                batch_pending_counts[next_batch_idx] = len(msg_ids)
                batch_valid_counts[next_batch_idx] = 0
                batch_errors[next_batch_idx] = []
                for msg_id in msg_ids:
                    pending[msg_id] = next_batch_idx
                    pending_downstream_bytes[msg_id] = downstream_bytes.get(msg_id, 0)
                next_batch_idx += 1

        push_more_batches()
        while completed_batches < len(batches):
            if end_time is not None and time.time() >= end_time:
                log(WARNING, "Timed out waiting for pipelined layer downloads")
                break
            if not pending:
                push_more_batches()
                if not pending:
                    break
            pull_start = perf_counter()
            replies = list(grid.pull_messages(list(pending)))
            pull_ms = (perf_counter() - pull_start) * 1000.0
            if replies:
                upstream_bytes = 0
                for msg in replies:
                    reply_to = msg.metadata.reply_to_message_id
                    downstream_bytes = pending_downstream_bytes.pop(reply_to, 0)
                    if downstream_bytes:
                        msg.metadata.__dict__["_network_downstream_bytes"] = (
                            downstream_bytes
                        )
                    upstream_bytes += int(
                        msg.metadata.__dict__.get("_network_upstream_bytes", 0)
                    )
                _record_server_profile(
                    "network_upstream",
                    pull_ms,
                    {
                        "received": len(replies),
                        "pending": len(pending),
                        "pipeline_depth": self._download_pipeline_depth,
                        "network_bytes": upstream_bytes,
                    },
                )
                _record_profile_replies(replies)
            else:
                time.sleep(pull_interval)
                continue

            newly_complete: list[int] = []
            for msg in replies:
                batch_idx = pending.pop(msg.metadata.reply_to_message_id, None)
                if batch_idx is None:
                    continue
                batch_pending_counts[batch_idx] -= 1
                if msg.has_error():
                    batch_errors[batch_idx].append(msg)
                else:
                    batch_valid_counts[batch_idx] += 1
                if batch_pending_counts[batch_idx] <= 0:
                    newly_complete.append(batch_idx)

            for batch_idx in sorted(set(newly_complete)):
                error_replies = batch_errors.pop(batch_idx, [])
                valid_count = batch_valid_counts.pop(batch_idx, 0)
                batch_pending_counts.pop(batch_idx, None)
                for msg in error_replies:
                    log(
                        WARNING,
                        "[Layer download] batch %s/%s error from node %s: %s",
                        batch_idx + 1,
                        len(batches),
                        msg.metadata.src_node_id,
                        msg.error.reason,
                    )
                if valid_count < len(node_ids):
                    log(
                        WARNING,
                        "Layer download ack mismatch for batch %s/%s: %s/%s",
                        batch_idx + 1,
                        len(batches),
                        valid_count,
                        len(node_ids),
                    )
                completed_batches += 1
                log_download_progress(batch_idx)

            push_more_batches()

    def _apply_aggregated_upload_chunk(
        self,
        *,
        entry: dict[str, Any],
        chunk_tensor: torch.Tensor,
        state_dict: dict[str, Any],
        aggregated_layers: dict[str, torch.Tensor],
        offload_enabled: bool,
        offload_dir: str,
        chunk_count_by_layer: dict[str, int],
        layer_names: list[str],
    ) -> None:
        """Apply one aggregated upload chunk to the server state."""
        layer_idx = int(entry["layer_idx"])
        layer_name = str(entry["layer_name"])
        start = int(entry["start"])
        end = int(entry["end"])
        is_last_chunk = bool(entry["is_last_chunk"])

        tensor = state_dict[layer_name]
        full_layer_chunk = tensor.ndim == 0 or (
            start == 0 and getattr(tensor, "ndim", 0) > 0 and end >= tensor.shape[0]
        )
        if full_layer_chunk:
            aggregated_layers[layer_name] = chunk_tensor
        else:
            agg_tensor = aggregated_layers.get(layer_name)
            if agg_tensor is None:
                base_tensor = tensor.detach().cpu() if hasattr(tensor, "cpu") else tensor
                agg_tensor = base_tensor.clone()
            agg_tensor[start:end] = chunk_tensor
            aggregated_layers[layer_name] = agg_tensor

        if is_last_chunk:
            final_tensor = aggregated_layers.pop(layer_name)
            if offload_enabled:
                file_name = f"{layer_idx:04d}_{_sanitize_layer_name(layer_name)}.pt"
                file_path = os.path.join(offload_dir, file_name)
                torch.save(final_tensor, file_path)
                del state_dict[layer_name]
            else:
                state_dict[layer_name] = final_tensor
            layer_count = len(layer_names)
            progress_every = max(1, layer_count // 10)
            if (
                layer_idx == 0
                or (layer_idx + 1) % progress_every == 0
                or (layer_idx + 1) == layer_count
            ):
                log(
                    INFO,
                    "[Layer %s/%s] done (%s chunks)",
                    layer_idx + 1,
                    layer_count,
                    chunk_count_by_layer.get(layer_name, 0),
                )

    def _aggregate_streamed_upload_replies(
        self,
        *,
        grid: Grid,
        msg_ids: list[str],
        batch_idx: int,
        batch_count: int,
        batch_entries: list[dict[str, Any]],
        timeout: float | None,
        process: psutil.Process,
        state_dict: dict[str, Any],
        aggregated_layers: dict[str, torch.Tensor],
        offload_enabled: bool,
        offload_dir: str,
        chunk_count_by_layer: dict[str, int],
        layer_names: list[str],
        downstream_bytes_by_id: dict[str, int],
        client_names_by_node_id: dict[int, str],
    ) -> None:
        """Pull reply objects one array at a time and aggregate incrementally."""
        if not msg_ids:
            return
        if not _has_streaming_object_pull(grid):
            raise RuntimeError("Grid does not expose private streaming object pull RPCs")

        run_id = getattr(getattr(grid, "_run"), "run_id")
        stub = getattr(grid, "_stub")
        object_reader = _StreamObjectReader(grid, run_id)
        pending = set(msg_ids)
        streamed_replies: list[_StreamedReply] = []
        error_replies: list[Message] = []
        pull_interval = float(getattr(grid, "pull_interval", 0.1))
        end_time = None if timeout is None else time.time() + timeout
        upstream_pull_ms = 0.0

        while pending:
            if end_time is not None and time.time() >= end_time:
                log(
                    WARNING,
                    "Timed out waiting for streamed layer upload replies: %s pending",
                    len(pending),
                )
                break

            pull_start = perf_counter()
            res = stub.PullMessages(
                PullAppMessagesRequest(
                    message_ids=list(pending),
                    run_id=run_id,
                )
            )
            upstream_pull_ms += (perf_counter() - pull_start) * 1000.0
            if not res.messages_list:
                time.sleep(pull_interval)
                continue

            # This timestamp is taken before pulling the reply's child objects.
            # Add those object-pull durations below exactly once.
            replies_received_at_ms = time.time() * 1000.0

            for msg_proto, msg_tree in zip(
                res.messages_list, res.message_object_trees, strict=True
            ):
                light_msg = message_from_proto(msg_proto)
                pending.discard(light_msg.metadata.reply_to_message_id)
                if light_msg.has_error():
                    error_replies.append(light_msg)
                    object_reader.confirm(msg_tree.object_id)
                    continue

                owner_key = str(msg_tree.object_id)
                object_ids = {
                    str(tree.object_id) for tree in iterate_object_tree(msg_tree)
                }
                array_refs, metrics, downstream_ms = object_reader.array_refs(
                    msg_tree.object_id, owner_key=owner_key
                )
                # Streamed replies are pulled from the raw ServerAppIo response, so
                # the delivery sidecar is inline in ``msg_proto`` rather than part of
                # the object tree. Keep the object-tree fallback for compatibility,
                # but prefer the lightweight reply metadata when present.
                sidecar_downstream_ms = _downstream_duration_from_message(light_msg)
                if sidecar_downstream_ms is not None:
                    downstream_ms = sidecar_downstream_ms
                streamed_replies.append(
                    _StreamedReply(
                        object_id=msg_tree.object_id,
                        reply_to_message_id=light_msg.metadata.reply_to_message_id,
                        node_id=int(light_msg.metadata.src_node_id),
                        owner_key=owner_key,
                        object_ids=object_ids,
                        message_bytes=(
                            len(msg_proto.SerializeToString())
                            + len(msg_tree.SerializeToString())
                        ),
                        created_at_ms=float(light_msg.metadata.created_at) * 1000.0,
                        delivered_at_ms=replies_received_at_ms,
                        downstream_ms=downstream_ms,
                        array_refs=array_refs,
                        metrics=metrics,
                        client_name=client_names_by_node_id.get(
                            int(light_msg.metadata.src_node_id)
                        ),
                    )
                )

        for msg in error_replies:
            log(
                WARNING,
                "[Layer upload] streamed batch %s/%s error from node %s: %s",
                batch_idx + 1,
                batch_count,
                msg.metadata.src_node_id,
                msg.error.reason,
            )
        if not streamed_replies:
            return

        weights = [
            float(reply.metrics.get(self.weighted_by_key, 1.0))
            for reply in streamed_replies
        ]

        before_mb = process.memory_info().rss / (1024**2)
        log(INFO, "Aggregation memory before (layerwise streamed): %.2f MB", before_mb)

        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 for _ in streamed_replies]
            total_weight = float(len(streamed_replies))
        weight_factors = [weight / total_weight for weight in weights]

        aggregate_ms = 0.0
        for entry in batch_entries:
            layer_name = str(entry["layer_name"])
            start = int(entry["start"])
            end = int(entry["end"])
            key = _chunk_key(layer_name, start, end)
            aggregated_np = None

            for reply, weight in zip(streamed_replies, weight_factors, strict=True):
                array_id = reply.array_refs.get(key)
                if array_id is None:
                    continue

                array = object_reader.array(array_id, owner_key=reply.owner_key)
                aggregate_start = perf_counter()
                value_np = array.numpy()
                if value_np.dtype.kind in {"f", "c"} and weight != 1.0:
                    value_np *= weight
                if aggregated_np is None:
                    aggregated_np = (
                        value_np
                        if value_np.dtype.kind in {"f", "c"}
                        else value_np * weight
                    )
                else:
                    if value_np.dtype.kind in {"f", "c"}:
                        aggregated_np += value_np
                    else:
                        aggregated_np += value_np * weight
                aggregate_ms += (perf_counter() - aggregate_start) * 1000.0
                if aggregated_np is not value_np:
                    del value_np
                del array

            if aggregated_np is None:
                continue

            apply_start = perf_counter()
            self._apply_aggregated_upload_chunk(
                entry=entry,
                chunk_tensor=torch.from_numpy(aggregated_np),
                state_dict=state_dict,
                aggregated_layers=aggregated_layers,
                offload_enabled=offload_enabled,
                offload_dir=offload_dir,
                chunk_count_by_layer=chunk_count_by_layer,
                layer_names=layer_names,
            )
            aggregate_ms += (perf_counter() - apply_start) * 1000.0
            del aggregated_np

        for reply in streamed_replies:
            object_reader.confirm(reply.object_id)

        for reply in streamed_replies:
            reply.network_bytes = (
                reply.message_bytes
                + object_reader.bytes_for(reply.object_ids)
            )
            upstream_duration_ms = max(
                reply.delivered_at_ms - reply.created_at_ms,
                0.0,
            ) + object_reader.pull_ms_for(reply.owner_key)
            downstream_bytes = downstream_bytes_by_id.get(
                reply.reply_to_message_id, 0
            )
            _record_network_profile(
                "upstream",
                upstream_duration_ms,
                node_id=reply.node_id,
                sender_node_id=reply.node_id,
                receiver_node_id="server",
                network_bytes=reply.network_bytes,
                client_name=reply.client_name,
            )
            if reply.downstream_ms is not None:
                _record_network_profile(
                    "downstream",
                    reply.downstream_ms,
                    node_id=reply.node_id,
                    sender_node_id="server",
                    receiver_node_id=reply.node_id,
                    network_bytes=downstream_bytes,
                    client_name=reply.client_name,
                )
                _record_network_profile(
                    "combined",
                    upstream_duration_ms + reply.downstream_ms,
                    node_id=reply.node_id,
                    sender_node_id=None,
                    receiver_node_id=None,
                    network_bytes=reply.network_bytes + downstream_bytes,
                    client_name=reply.client_name,
                )

        _record_server_profile(
            "network_upstream",
            upstream_pull_ms + object_reader.pull_ms,
            {
                "mode": "layerwise_streamed_objects",
                "batch_idx": batch_idx,
                "replies": len(streamed_replies),
                "chunks_in_message": len(batch_entries),
                "network_bytes": sum(
                    reply.network_bytes for reply in streamed_replies
                ),
            },
        )
        _record_server_profile(
            "aggregate",
            aggregate_ms,
            {
                "mode": "layerwise_streamed_objects",
                "batch_idx": batch_idx,
                "chunks_in_message": len(batch_entries),
            },
        )

        after_mb = process.memory_info().rss / (1024**2)
        log(INFO, "Aggregation memory after (layerwise streamed): %.2f MB", after_mb)
        publish_profile_summary()
        gc.collect()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training (no arrays sent)."""
        if self.fraction_train == 0.0:
            return []
        aggregation_mode = config.get("aggregation.mode", "layerwise")
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        config["server-round"] = server_round
        config["model_preloaded"] = aggregation_mode != "all_at_once"
        if self._layer_names is not None:
            config["layer_names"] = self._layer_names
            config["num_layers"] = len(self._layer_names)

        if aggregation_mode == "all_at_once":
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: config}
            )
        else:
            record = RecordDict({self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate MetricRecords only (arrays are streamed separately)."""
        valid_replies, _ = self._check_and_log_replies(
            replies, is_train=True, validate=False
        )

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return None, metrics

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy with layer-wise aggregation."""
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        process = psutil.Process()
        upload_target_message_size_mb = train_config.get(
            "aggregation.upload-target-message-size",
            train_config.get(
                "aggregation.target-message-size-mb",
                train_config.get("aggregation.batch-size-mb", 2048),
            ),
        )
        upload_target_message_size_mb = float(upload_target_message_size_mb)
        if upload_target_message_size_mb <= 0:
            raise ValueError("aggregation.upload-target-message-size must be > 0")
        upload_target_message_bytes = max(
            1, int(upload_target_message_size_mb * 1024 * 1024)
        )
        self._upload_max_chunk_bytes = min(upload_target_message_bytes, SAFE_GRPC_BYTES)
        download_target_message_size_mb = float(
            train_config.get(
                "aggregation.download-target-message-size",
                train_config.get(
                    "aggregation.download-target-message-size-mb",
                    upload_target_message_size_mb,
                ),
            )
        )
        if download_target_message_size_mb <= 0:
            raise ValueError(
                "aggregation.download-target-message-size must be > 0"
            )
        download_target_message_bytes = max(
            1, int(download_target_message_size_mb * 1024 * 1024)
        )
        self._download_max_chunk_bytes = min(
            download_target_message_bytes, SAFE_GRPC_BYTES
        )
        self._chunks_per_message = DEFAULT_LAYERWISE_CHUNKS_PER_MESSAGE
        self._download_pipeline_depth = DEFAULT_LAYERWISE_DOWNLOAD_PIPELINE_DEPTH
        log(
            INFO,
            (
                "Layerwise target message size (upload/download): %.2f MB / %.2f MB, "
                "chunks/message cap: %s, download pipeline depth: %s"
            ),
            self._upload_max_chunk_bytes / (1024 * 1024),
            self._download_max_chunk_bytes / (1024 * 1024),
            self._chunks_per_message,
            self._download_pipeline_depth,
        )
        offload_enabled = bool(train_config.get("aggregation.offload", False))
        offload_dir = str(train_config.get("aggregation.offload-dir", ""))
        if offload_enabled:
            if not offload_dir:
                offload_dir = os.path.join(os.getcwd(), "results", "aggregated_layers")
            os.makedirs(offload_dir, exist_ok=True)
            log(INFO, "Layer offload enabled. Writing layers to %s", offload_dir)

        if self._state_dict is None:
            if len(initial_arrays) == 0:
                raise ValueError("initial_state_dict required when initial_arrays empty")
            self._state_dict = initial_arrays.to_torch_state_dict()
        state_dict = self._state_dict
        self._layer_names = list(state_dict.keys())

        t_start = time.time()
        if evaluate_fn:
            if offload_enabled:
                state_dict = _rehydrate_state_dict(self._layer_names, offload_dir)
            if _has_meta_tensors(state_dict):
                log(
                    WARNING,
                    "Skipping initial evaluation: model contains meta tensors.",
                )
            else:
                res = evaluate_fn(0, ArrayRecord(state_dict))
                log(INFO, "Initial global evaluation results: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            set_current_round(current_round)
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            aggregation_mode = train_config.get("aggregation.mode", "layerwise")
            server_input_fingerprint = state_dict_fingerprint(state_dict)
            log(
                INFO,
                "[Model fingerprint] round %s server before train: %.12g",
                current_round,
                server_input_fingerprint,
            )

            # In all_at_once mode, send the current global model each round.
            # Using `initial_arrays` here would resend stale/empty arrays.
            round_arrays = (
                ArrayRecord(state_dict)
                if aggregation_mode == "all_at_once"
                else initial_arrays
            )

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------
            train_messages = list(
                self.configure_train(
                    current_round,
                    round_arrays,
                    train_config,
                    grid,
                )
            )
            if not train_messages:
                log(WARNING, "No train messages configured for round %s", current_round)
                continue

            selected_node_ids = [msg.metadata.dst_node_id for msg in train_messages]
            if aggregation_mode != "all_at_once":
                self._download_layers_to_clients(
                    grid=grid,
                    node_ids=selected_node_ids,
                    state_dict=state_dict,
                    timeout=timeout,
                )

            train_replies = grid.send_and_receive(
                messages=train_messages,
                timeout=timeout,
            )

            # Aggregate metrics only
            _, agg_train_metrics = self.aggregate_train(current_round, train_replies)
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            node_ids = [
                msg.metadata.src_node_id for msg in train_replies if not msg.has_error()
            ]
            client_names_by_node_id: dict[int, str] = {}
            for msg in train_replies:
                client_name = client_name_from_message(msg)
                if client_name:
                    client_names_by_node_id[int(msg.metadata.src_node_id)] = client_name
            if not node_ids:
                log(WARNING, "No valid training replies, skipping round %s", current_round)
                continue

            # -----------------------------------------------------------------
            # --- LAYER-WISE COMMUNICATION -----------------------------------
            # -----------------------------------------------------------------
            if aggregation_mode == "all_at_once":
                before_mb = process.memory_info().rss / (1024**2)
                log(
                    INFO,
                    "Aggregation memory before (all_at_once): %.2f MB",
                    before_mb,
                )
                valid_replies, _ = self._check_and_log_replies(
                    train_replies, is_train=True
                )
                if not valid_replies:
                    log(
                        WARNING,
                        "No valid training replies for all_at_once aggregation, "
                        "skipping round %s",
                        current_round,
                    )
                    continue
                reply_contents = [msg.content for msg in valid_replies]
                profiler = get_active_profiler()
                start_time = perf_counter() if profiler is not None else None
                agg_record = aggregate_arrayrecords(
                    reply_contents,
                    self.weighted_by_key,
                )
                if profiler is not None and start_time is not None:
                    duration_ms = (perf_counter() - start_time) * 1000.0
                    profiler.record(
                        scope="server",
                        task="aggregate",
                        round=current_round,
                        node_id=None,
                        duration_ms=duration_ms,
                        metadata={"mode": "all_at_once"},
                    )
                    publish_profile_summary()
                after_mb = process.memory_info().rss / (1024**2)
                log(
                    INFO,
                    "Aggregation memory after (all_at_once): %.2f MB",
                    after_mb,
                )
                state_dict = agg_record.to_torch_state_dict()
            else:
                layer_names = self._layer_names or []
                upload_entries = _build_layer_chunk_entries(
                    layer_names,
                    state_dict,
                    max(1, int(self._upload_max_chunk_bytes)),
                )
                upload_batches = [upload_entries] if upload_entries else []
                if not upload_batches:
                    log(
                        WARNING,
                        "No upload batches generated, skipping round %s",
                        current_round,
                    )
                    continue

                chunk_count_by_layer = {
                    layer_name: 0 for layer_name in layer_names
                }
                for entry in upload_entries:
                    layer_name = str(entry["layer_name"])
                    chunk_count_by_layer[layer_name] = (
                        chunk_count_by_layer.get(layer_name, 0) + 1
                    )

                log(
                    INFO,
                    (
                        "[Layer upload] sending %s batches (%s chunks across %s "
                        "layers, %.2f MB/message max, %.2f MB/chunk max, "
                        "streamed object aggregation)"
                    ),
                    len(upload_batches),
                    len(upload_entries),
                    len(layer_names),
                    self._upload_max_chunk_bytes / (1024 * 1024),
                    self._upload_max_chunk_bytes / (1024 * 1024),
                )
                upload_progress_every = max(1, len(upload_batches) // 10)

                def build_upload_record(
                    batch_idx: int, batch_entries: list[dict[str, Any]]
                ) -> RecordDict:
                    upload_layer_idxs: list[int] = []
                    upload_layer_names: list[str] = []
                    upload_chunk_starts: list[int] = []
                    upload_chunk_ends: list[int] = []
                    upload_is_last_chunk: list[bool] = []
                    for entry in batch_entries:
                        upload_layer_idxs.append(int(entry["layer_idx"]))
                        upload_layer_names.append(str(entry["layer_name"]))
                        upload_chunk_starts.append(int(entry["start"]))
                        upload_chunk_ends.append(int(entry["end"]))
                        upload_is_last_chunk.append(bool(entry["is_last_chunk"]))

                    config = ConfigRecord(
                        {
                            "upload_layer_idxs": upload_layer_idxs,
                            "upload_layer_names": upload_layer_names,
                            "upload_chunk_starts": upload_chunk_starts,
                            "upload_chunk_ends": upload_chunk_ends,
                            "upload_is_last_chunk": upload_is_last_chunk,
                            "upload_batch_idx": batch_idx,
                            "upload_batch_count": len(upload_batches),
                            "upload_chunks_in_message": len(batch_entries),
                        }
                    )
                    return RecordDict({self.configrecord_key: config})

                def log_upload_progress(batch_idx: int) -> None:
                    if (
                        (batch_idx + 1) % upload_progress_every == 0
                        or batch_idx == 0
                        or (batch_idx + 1) == len(upload_batches)
                    ):
                        log(
                            INFO,
                            "[Layer upload] progress %s/%s batches (%.1f%%)",
                            batch_idx + 1,
                            len(upload_batches),
                            (100.0 * (batch_idx + 1)) / len(upload_batches),
                        )

                aggregated_layers: dict[str, torch.Tensor] = {}
                for batch_idx, batch_entries in enumerate(upload_batches):
                    record = build_upload_record(batch_idx, batch_entries)
                    messages = list(
                        self._construct_messages(
                            record,
                            node_ids,
                            message_type="train.layer_wise_communication",
                        )
                    )
                    push_start = perf_counter()
                    msg_ids = [
                        msg_id for msg_id in grid.push_messages(messages) if msg_id
                    ]
                    downstream_bytes_by_id = grid.pop_pushed_message_sizes(msg_ids)
                    downstream_enqueue_ms = (perf_counter() - push_start) * 1000.0
                    _record_server_profile(
                        "network_downstream",
                        downstream_enqueue_ms,
                        {
                            "expected_replies": len(msg_ids),
                            "batch_idx": batch_idx,
                            "mode": "layerwise_streamed_objects",
                            "network_bytes": sum(downstream_bytes_by_id.values()),
                        },
                    )
                    self._aggregate_streamed_upload_replies(
                        grid=grid,
                        msg_ids=msg_ids,
                        batch_idx=batch_idx,
                        batch_count=len(upload_batches),
                        batch_entries=batch_entries,
                        timeout=timeout,
                        process=process,
                        state_dict=state_dict,
                        aggregated_layers=aggregated_layers,
                        offload_enabled=offload_enabled,
                        offload_dir=offload_dir,
                        chunk_count_by_layer=chunk_count_by_layer,
                        layer_names=layer_names,
                        downstream_bytes_by_id=downstream_bytes_by_id,
                        client_names_by_node_id=client_names_by_node_id,
                    )
                    log_upload_progress(batch_idx)

                if aggregated_layers:
                    log(
                        WARNING,
                        "Round %s finished with %s incomplete layer aggregates",
                        current_round,
                        len(aggregated_layers),
                    )
                    for layer_name, agg_tensor in aggregated_layers.items():
                        state_dict[layer_name] = agg_tensor
                    aggregated_layers.clear()

            server_output_fingerprint = state_dict_fingerprint(state_dict)
            log(
                INFO,
                "[Model fingerprint] round %s server after aggregation: %.12g (delta %.12g)",
                current_round,
                server_output_fingerprint,
                server_output_fingerprint - server_input_fingerprint,
            )

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------
            if self.fraction_evaluate > 0.0:
                if offload_enabled:
                    state_dict = _rehydrate_state_dict(self._layer_names, offload_dir)
                arrays = ArrayRecord(state_dict)
                evaluate_replies = grid.send_and_receive(
                    messages=self.configure_evaluate(
                        current_round,
                        arrays,
                        evaluate_config,
                        grid,
                    ),
                    timeout=timeout,
                )
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round,
                    evaluate_replies,
                )
                if agg_evaluate_metrics is not None:
                    log(
                        INFO,
                        "\t└──> Aggregated MetricRecord: %s",
                        agg_evaluate_metrics,
                    )
                    result.evaluate_metrics_clientapp[current_round] = (
                        agg_evaluate_metrics
                    )

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------
            if evaluate_fn:
                if offload_enabled:
                    state_dict = _rehydrate_state_dict(self._layer_names, offload_dir)
                if _has_meta_tensors(state_dict):
                    log(
                        WARNING,
                        "Skipping global evaluation: model contains meta tensors.",
                    )
                else:
                    arrays = ArrayRecord(state_dict)
                    log(INFO, "Global evaluation")
                    res = evaluate_fn(current_round, arrays)
                    log(INFO, "\t└──> MetricRecord: %s", res)
                    if res is not None:
                        result.evaluate_metrics_serverapp[current_round] = res

            publish_profile_summary()

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        if offload_enabled:
            state_dict = _rehydrate_state_dict(self._layer_names, offload_dir)
            log(INFO, "Final model layers offloaded to %s", offload_dir)
        elif _has_meta_tensors(state_dict):
            log(
                WARNING,
                "Skipping final ArrayRecord: model contains meta tensors.",
            )
        else:
            final_arrays = ArrayRecord(state_dict)
            result.arrays = final_arrays
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")
        publish_profile_summary()

        return result
