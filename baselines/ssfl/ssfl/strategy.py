"""SSFL training strategy: deterministic sampling + custom metric aggregation."""

from __future__ import annotations

import random
from collections.abc import Iterable
from logging import INFO
from typing import Optional

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from ssfl.mask import apply_mask_to_state_dict
from ssfl.metrics import aggregate_train_metrics
from ssfl.sparse_codec import pack_state_dict, unpack_state_dict


class SSFLStrategy(FedAvg):
    """FedAvg with stable client-ID sampling and SSFL metric aggregation."""

    def __init__(
        self,
        *,
        node_to_client_id: dict[int, int],
        masks: dict[str, torch.Tensor] | None = None,
        mask_version: str = "",
        sample_seed: int = 550,
        transport: str = "dense",
        **kwargs,
    ) -> None:
        kwargs.setdefault("train_metrics_aggr_fn", aggregate_train_metrics)
        super().__init__(**kwargs)
        self.node_to_client_id = dict(node_to_client_id)
        self.client_id_to_node = {v: k for k, v in self.node_to_client_id.items()}
        self.masks = masks
        self.mask_version = mask_version
        self.sample_seed = sample_seed
        if transport not in {"dense", "sparse"}:
            raise ValueError(f"Unsupported transport: {transport}")
        self.transport = transport

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure a training round with deterministic client sampling."""
        if self.fraction_train == 0.0:
            return []

        available = set(grid.get_node_ids())
        eligible_client_ids = sorted(
            cid for nid, cid in self.node_to_client_id.items() if nid in available
        )
        if not eligible_client_ids:
            raise RuntimeError("No discovery-mapped clients are available for training")

        sample_size = max(
            int(len(eligible_client_ids) * self.fraction_train),
            self.min_train_nodes,
        )
        sample_size = min(sample_size, len(eligible_client_ids))

        # Deterministic sampling by stable client ID.
        rng = random.Random(self.sample_seed + server_round)
        selected_client_ids = sorted(rng.sample(eligible_client_ids, sample_size))
        node_ids = [self.client_id_to_node[cid] for cid in selected_client_ids]

        log(
            INFO,
            "configure_train: Sampled %s clients by stable ID (out of %s mapped) "
            "transport=%s",
            len(node_ids),
            len(eligible_client_ids),
            self.transport,
        )

        # Copy so round-specific keys do not permanently mutate the shared config.
        round_config = ConfigRecord(dict(config))
        round_config["server-round"] = server_round
        round_config["mask-version"] = self.mask_version
        round_config["transport"] = self.transport

        arrays_to_send = arrays
        if self.transport == "sparse":
            if self.masks is None:
                raise RuntimeError("Sparse transport requires an installed global mask")
            dense = arrays.to_torch_state_dict()
            packed = pack_state_dict(dense, self.masks)
            arrays_to_send = ArrayRecord(packed)

        record = RecordDict(
            {
                self.arrayrecord_key: arrays_to_send,
                self.configrecord_key: round_config,
            }
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate client updates and reapply the global mask."""
        reply_list = list(replies)
        if self.transport == "sparse":
            if self.masks is None:
                raise RuntimeError("Sparse transport requires an installed global mask")
            for reply in reply_list:
                if reply.has_error():
                    continue
                array_record = reply.content[self.arrayrecord_key]
                if not isinstance(array_record, ArrayRecord):
                    raise TypeError(
                        f"Expected ArrayRecord under {self.arrayrecord_key!r}"
                    )
                packed = array_record.to_torch_state_dict()
                dense = unpack_state_dict(packed, self.masks)
                reply.content[self.arrayrecord_key] = ArrayRecord(dense)

        arrays, metrics = super().aggregate_train(server_round, reply_list)
        if arrays is None or self.masks is None:
            return arrays, metrics

        # Re-enforce the static mask after aggregation.
        state = arrays.to_torch_state_dict()
        masked = apply_mask_to_state_dict(state, self.masks)
        return ArrayRecord(masked), metrics
