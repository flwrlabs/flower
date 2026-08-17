"""SSFL Flower ClientApp: saliency discovery, mask install, sparse-aware train."""

from __future__ import annotations

from collections.abc import Sized
from typing import cast

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from ssfl.data import first_batches, load_partition_dataloaders
from ssfl.mask import mask_digest, masks_from_uint8, masks_to_cpu_uint8
from ssfl.model import create_model, num_classes_for_dataset
from ssfl.reproducibility import client_round_seed, seed_everything
from ssfl.saliency import average_saliency_over_batches
from ssfl.sparse_codec import pack_state_dict, unpack_state_dict
from ssfl.training import count_nonzero_params, sparsity_from_state_dict, train_local

app = ClientApp()


def _array_record(records: RecordDict, key: str = "arrays") -> ArrayRecord:
    record = records[key]
    if not isinstance(record, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord under {key!r}")
    return record


def _config_record(records: RecordDict, key: str = "config") -> ConfigRecord:
    record = records[key]
    if not isinstance(record, ConfigRecord):
        raise TypeError(f"Expected ConfigRecord under {key!r}")
    return record


def _run_config(context: Context) -> dict:
    return dict(context.run_config)


def _stable_client_id(context: Context) -> int:
    node_cfg = context.node_config
    if "partition-id" in node_cfg:
        return int(node_cfg["partition-id"])
    if "client-id" in node_cfg:
        return int(node_cfg["client-id"])
    raise KeyError(
        "Client identity missing: expected 'partition-id' (simulation) "
        "or 'client-id' (deployment) in context.node_config"
    )


def _num_partitions(context: Context, cfg: dict) -> int:
    if "num-partitions" in context.node_config:
        return int(context.node_config["num-partitions"])
    return int(cfg["num-clients"])


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_masks_from_state(context: Context) -> dict[str, torch.Tensor] | None:
    if "ssfl-mask" not in context.state:
        return None
    mask_record = _array_record(context.state, "ssfl-mask")
    return masks_from_uint8(mask_record.to_torch_state_dict())


@app.query("saliency")
def saliency(msg: Message, context: Context) -> Message:
    """Compute local SSFL saliency scores for mask discovery."""
    cfg = _run_config(context)
    client_id = _stable_client_id(context)
    seed_everything(client_round_seed(int(cfg["seed"]), client_id, round_idx=0))

    dataset_name = str(cfg["dataset"])
    model_name = str(cfg["model"])
    batch_size = int(cfg["batch-size"])
    num_partitions = _num_partitions(context, cfg)
    device = _device()

    model = create_model(model_name, num_classes_for_dataset(dataset_name))
    model.load_state_dict(_array_record(msg.content).to_torch_state_dict())
    model.to(device)

    trainloader, _ = load_partition_dataloaders(
        dataset_name=dataset_name,
        partition_id=client_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        partition_alpha=float(cfg["partition-alpha"]),
        seed=int(cfg["seed"]),
        data_path=str(cfg.get("data-path", "")),
        max_partition_samples=int(cfg.get("max-partition-samples", 0)),
    )
    n_batches = int(cfg["saliency-batches"])
    batches = first_batches(trainloader, n_batches)
    if not batches:
        raise RuntimeError(f"Client {client_id} has no saliency batches")

    scores = average_saliency_over_batches(model, batches, device=device)
    # ArrayRecord expects tensors; keep CPU float32 scores.
    score_record = ArrayRecord({k: v.contiguous() for k, v in scores.items()})
    metrics = MetricRecord(
        {
            "client-id": float(client_id),
            "num-examples": float(len(cast(Sized, trainloader.dataset))),
            "num-score-tensors": float(len(scores)),
        }
    )
    content = RecordDict({"arrays": score_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.query("install_mask")
def install_mask(msg: Message, context: Context) -> Message:
    """Persist the global static mask in ClientApp context state."""
    masks_uint8 = _array_record(msg.content).to_torch_state_dict()
    masks = masks_from_uint8(masks_uint8)
    digest = mask_digest(masks)
    expected = str(_config_record(msg.content).get("mask-version", ""))
    if expected and digest != expected:
        raise ValueError(f"Mask digest mismatch: got {digest}, expected {expected}")

    context.state["ssfl-mask"] = ArrayRecord(masks_to_cpu_uint8(masks))
    context.state["ssfl-mask-meta"] = ConfigRecord(
        {
            "mask-version": digest,
            "num-mask-tensors": len(masks),
        }
    )
    metrics = MetricRecord({"mask_ack": 1.0, "num-mask-tensors": float(len(masks))})
    content = RecordDict(
        {
            "metrics": metrics,
            "config": ConfigRecord({"mask-version": digest}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Local masked SGD training for one federated round."""
    cfg = _run_config(context)
    client_id = _stable_client_id(context)
    train_config = _config_record(msg.content)
    server_round = int(cast(int | float | str, train_config["server-round"]))
    seed_everything(client_round_seed(int(cfg["seed"]), client_id, server_round))

    dataset_name = str(cfg["dataset"])
    model_name = str(cfg["model"])
    device = _device()

    requested_mask = str(train_config.get("mask-version", ""))
    masks = _load_masks_from_state(context)
    if masks is None:
        raise RuntimeError(
            f"Client {client_id} has no installed mask; run query.install_mask first"
        )
    local_digest = mask_digest(masks)
    if requested_mask and local_digest != requested_mask:
        raise RuntimeError(
            f"Client {client_id} mask version mismatch: "
            f"local={local_digest}, requested={requested_mask}"
        )

    transport = str(train_config.get("transport", cfg.get("transport", "dense")))
    incoming = _array_record(msg.content).to_torch_state_dict()
    if transport == "sparse":
        state_in = unpack_state_dict(incoming, masks)
    elif transport == "dense":
        state_in = incoming
    else:
        raise ValueError(f"Unsupported transport: {transport}")

    model = create_model(model_name, num_classes_for_dataset(dataset_name))
    model.load_state_dict(state_in)

    trainloader, _ = load_partition_dataloaders(
        dataset_name=dataset_name,
        partition_id=client_id,
        num_partitions=_num_partitions(context, cfg),
        batch_size=int(cfg["batch-size"]),
        partition_alpha=float(cfg["partition-alpha"]),
        seed=int(cfg["seed"]),
        data_path=str(cfg.get("data-path", "")),
        max_partition_samples=int(cfg.get("max-partition-samples", 0)),
    )

    train_loss, final_lr = train_local(
        model,
        trainloader,
        epochs=int(cfg["local-epochs"]),
        lr=float(
            cast(
                int | float | str,
                train_config.get("lr", cfg["learning-rate"]),
            )
        ),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        max_grad_norm=float(cfg["max-grad-norm"]),
        round_idx=server_round,
        lr_scheduler_name=str(cfg["lr-scheduler"]),
        lr_decay=float(cfg["lr-decay"]),
        scheduler_cycle_length=int(cfg["scheduler-cycle-length"]),
        device=device,
        masks=masks,
    )

    state = model.state_dict()
    if transport == "sparse":
        outbound = pack_state_dict(state, masks)
    else:
        outbound = state
    model_record = ArrayRecord(outbound)
    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "num-examples": float(len(cast(Sized, trainloader.dataset))),
            "learning_rate": float(final_lr),
            "comm_params": float(count_nonzero_params(state)),
            "arrayrecord_payload_bytes": float(model_record.count_bytes()),
            "sparsity_percent": float(sparsity_from_state_dict(state, model)),
            "client-id": float(client_id),
        }
    )
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)
