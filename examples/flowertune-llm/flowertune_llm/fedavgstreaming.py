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
import os
import re
import gc
import io
from logging import INFO, WARNING
from time import perf_counter
import time
from typing import Any

import torch
import psutil

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    GRPC_MAX_MESSAGE_LENGTH,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.common.profiling import (
    get_active_profiler,
    publish_profile_summary,
    set_current_round,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    log_strategy_start_info,
    sample_nodes,
)

SAFE_GRPC_BYTES = int(GRPC_MAX_MESSAGE_LENGTH * 0.75)


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
                    "tensor": cpu_tensor,
                }
            )
    return entries


def _batch_entries_by_layer(
    entries: list[dict[str, Any]], layers_per_message: int
) -> list[list[dict[str, Any]]]:
    """Batch chunks by layer groups to keep protocol behavior easy to reason about."""
    if not entries or layers_per_message <= 0:
        return []

    by_layer: dict[int, list[dict[str, Any]]] = {}
    layer_order: list[int] = []
    for entry in entries:
        layer_idx = int(entry["layer_idx"])
        if layer_idx not in by_layer:
            by_layer[layer_idx] = []
            layer_order.append(layer_idx)
        by_layer[layer_idx].append(entry)

    batches: list[list[dict[str, Any]]] = []
    for group_start in range(0, len(layer_order), layers_per_message):
        group = layer_order[group_start : group_start + layers_per_message]
        max_chunks = max(len(by_layer[layer_idx]) for layer_idx in group)
        for chunk_pos in range(max_chunks):
            batch_entries: list[dict[str, Any]] = []
            for layer_idx in group:
                layer_entries = by_layer[layer_idx]
                if chunk_pos < len(layer_entries):
                    batch_entries.append(layer_entries[chunk_pos])
            if batch_entries:
                batches.append(batch_entries)
    return batches

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
        self._layers_per_message = 1

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
        max_bytes_per_layer_chunk = max(
            1, int(self._download_max_chunk_bytes // self._layers_per_message)
        )
        entries = _build_layer_chunk_entries(
            layer_names,
            state_dict,
            max_bytes_per_layer_chunk,
        )
        batches = _batch_entries_by_layer(entries, self._layers_per_message)
        if not batches:
            return

        log(
            INFO,
            "[Layer download] sending %s batches (%s chunks across %s layers, %s layers/message, %.2f MB/layer max) to %s clients",
            len(batches),
            len(entries),
            len(layer_names),
            self._layers_per_message,
            max_bytes_per_layer_chunk / (1024 * 1024),
            len(node_ids),
        )
        progress_every = max(1, len(batches) // 10)

        for batch_idx, batch_entries in enumerate(batches):
            arrays_dict: dict[str, torch.Tensor] = {}
            payload_layer_names: list[str] = []
            payload_layer_shapes: list[str] = []
            payload_chunk_starts: list[int] = []
            payload_chunk_ends: list[int] = []
            payload_is_last_chunk: list[bool] = []
            for entry in batch_entries:
                start = int(entry["start"])
                end = int(entry["end"])
                layer_name = str(entry["layer_name"])
                tensor = entry["tensor"]
                chunk_tensor = tensor if tensor.ndim == 0 else tensor[start:end]
                arrays_dict[_chunk_key(layer_name, start, end)] = chunk_tensor
                payload_layer_names.append(layer_name)
                payload_layer_shapes.append(_shape_to_text(entry["layer_shape"]))
                payload_chunk_starts.append(start)
                payload_chunk_ends.append(end)
                payload_is_last_chunk.append(bool(entry["is_last_chunk"]))

            arrays = ArrayRecord(arrays_dict)
            config = ConfigRecord(
                {
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
                {self.arrayrecord_key: arrays, self.configrecord_key: config}
            )
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
        layers_per_message = int(train_config.get("aggregation.layers-per-message", 1))
        if layers_per_message <= 0:
            raise ValueError("aggregation.layers-per-message must be > 0")
        self._layers_per_message = layers_per_message
        log(
            INFO,
            "Layerwise target message size (upload/download): %.2f MB / %.2f MB, layers/message: %s",
            self._upload_max_chunk_bytes / (1024 * 1024),
            self._download_max_chunk_bytes / (1024 * 1024),
            self._layers_per_message,
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
                    max(
                        1,
                        int(self._upload_max_chunk_bytes // self._layers_per_message),
                    ),
                )
                upload_batches = _batch_entries_by_layer(
                    upload_entries, self._layers_per_message
                )
                if not upload_batches:
                    log(WARNING, "No upload batches generated, skipping round %s", current_round)
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
                    "[Layer upload] sending %s batches (%s chunks across %s layers, %s layers/message, %.2f MB/layer max)",
                    len(upload_batches),
                    len(upload_entries),
                    len(layer_names),
                    self._layers_per_message,
                    (
                        self._upload_max_chunk_bytes
                        / max(1, self._layers_per_message)
                        / (1024 * 1024)
                    ),
                )
                upload_progress_every = max(1, len(upload_batches) // 10)

                aggregated_layers: dict[str, torch.Tensor] = {}
                for batch_idx, batch_entries in enumerate(upload_batches):
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
                    record = RecordDict({self.configrecord_key: config})
                    replies = grid.send_and_receive(
                        messages=self._construct_messages(
                            record,
                            node_ids,
                            message_type="train.layer_wise_communication",
                        ),
                        timeout=timeout,
                    )
                    valid_replies, error_replies = _split_replies(replies)
                    for msg in error_replies:
                        log(
                            WARNING,
                            "[Layer upload] batch %s/%s error from node %s: %s",
                            batch_idx + 1,
                            len(upload_batches),
                            msg.metadata.src_node_id,
                            msg.error.reason,
                        )
                    if not valid_replies:
                        continue

                    before_mb = process.memory_info().rss / (1024**2)
                    log(
                        INFO,
                        "Aggregation memory before (layerwise): %.2f MB",
                        before_mb,
                    )

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
                            metadata={
                                "mode": "layerwise",
                                "batch_idx": batch_idx,
                                "chunks_in_message": len(batch_entries),
                            },
                        )
                        publish_profile_summary()

                    after_mb = process.memory_info().rss / (1024**2)
                    log(
                        INFO,
                        "Aggregation memory after (layerwise): %.2f MB",
                        after_mb,
                    )

                    for entry in batch_entries:
                        layer_idx = int(entry["layer_idx"])
                        layer_name = str(entry["layer_name"])
                        start = int(entry["start"])
                        end = int(entry["end"])
                        is_last_chunk = bool(entry["is_last_chunk"])
                        chunk_key = _chunk_key(layer_name, start, end)
                        if chunk_key not in agg_record:
                            continue

                        tensor = state_dict[layer_name]
                        agg_tensor = aggregated_layers.get(layer_name)
                        if agg_tensor is None:
                            base_tensor = (
                                tensor.detach().cpu() if hasattr(tensor, "cpu") else tensor
                            )
                            agg_tensor = base_tensor.clone()
                        chunk_np = agg_record[chunk_key].numpy()
                        chunk_tensor = torch.from_numpy(chunk_np)

                        if tensor.ndim == 0:
                            agg_tensor = chunk_tensor
                        else:
                            agg_tensor[start:end] = chunk_tensor
                        aggregated_layers[layer_name] = agg_tensor
                        del chunk_np
                        del chunk_tensor

                        if is_last_chunk:
                            final_tensor = aggregated_layers.pop(layer_name, agg_tensor)
                            if offload_enabled:
                                file_name = (
                                    f"{layer_idx:04d}_{_sanitize_layer_name(layer_name)}.pt"
                                )
                                file_path = os.path.join(offload_dir, file_name)
                                torch.save(final_tensor, file_path)
                                del state_dict[layer_name]
                            else:
                                state_dict[layer_name] = final_tensor
                            del final_tensor
                            gc.collect()
                            log(
                                INFO,
                                "[Layer %s/%s] done (%s chunks)",
                                layer_idx + 1,
                                len(layer_names),
                                chunk_count_by_layer.get(layer_name, 0),
                            )

                    # Release per-batch memory aggressively
                    del agg_record
                    del reply_contents
                    del valid_replies
                    gc.collect()

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
