"""SSFL Flower ServerApp: mask discovery then masked FedAvg."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from logging import INFO
from pathlib import Path
from typing import Any

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from ssfl.comm_stats import CommStats
from ssfl.data import load_centralized_testloader
from ssfl.mask import (
    apply_mask_to_state_dict,
    create_mask_from_scores,
    get_mean_saliency_scores,
    get_weighted_mean_saliency_scores,
    mask_digest,
    masks_to_cpu_uint8,
)
from ssfl.model import create_model, num_classes_for_dataset
from ssfl.reproducibility import seed_everything
from ssfl.strategy import SSFLStrategy
from ssfl.training import evaluate_model
from ssfl.wandb_utils import WandbSession

app = ServerApp()


def _array_record(records: RecordDict, key: str = "arrays") -> ArrayRecord:
    record = records[key]
    if not isinstance(record, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord under {key!r}")
    return record


def _metric_record(records: RecordDict, key: str = "metrics") -> MetricRecord:
    record = records[key]
    if not isinstance(record, MetricRecord):
        raise TypeError(f"Expected MetricRecord under {key!r}")
    return record


def _metric_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("Expected a scalar numeric metric")
    return float(value)


def _device(prefer_cpu: bool = True) -> torch.device:
    if prefer_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def _cfg_bool(cfg: dict, key: str, default: bool = False) -> bool:
    if key not in cfg:
        return default
    value = cfg[key]
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _wait_for_nodes(grid: Grid, expected: int, timeout_s: float = 120.0) -> list[int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        node_ids = list(grid.get_node_ids())
        if len(node_ids) >= expected:
            return sorted(node_ids)[:expected]
        time.sleep(0.5)
    node_ids = list(grid.get_node_ids())
    raise RuntimeError(
        f"Timed out waiting for {expected} nodes; only {len(node_ids)} available"
    )


def _run_saliency_discovery(
    grid: Grid,
    *,
    arrays: ArrayRecord,
    node_ids: list[int],
    timeout: float,
) -> tuple[dict[int, int], list[dict[str, torch.Tensor]], list[float], int]:
    """Query all clients for saliency scores.

    Returns node->client map, scores, weights, and uplink payload bytes.
    """
    config = ConfigRecord({"phase": "saliency"})
    messages = [
        Message(
            content=RecordDict({"arrays": arrays, "config": config}),
            message_type="query.saliency",
            dst_node_id=node_id,
            group_id="discovery",
        )
        for node_id in node_ids
    ]
    replies = list(grid.send_and_receive(messages, timeout=timeout))

    node_to_client: dict[int, int] = {}
    score_dicts: list[dict[str, torch.Tensor]] = []
    weights: list[float] = []
    uplink_bytes = 0
    errors = []

    for reply in replies:
        if reply.has_error():
            errors.append((reply.metadata.src_node_id, reply.error.reason))
            continue
        src = int(reply.metadata.src_node_id)
        metrics = _metric_record(reply.content)
        client_id = int(_metric_float(metrics["client-id"]))
        if client_id in node_to_client.values():
            raise RuntimeError(f"Duplicate client-id {client_id} in saliency replies")
        node_to_client[src] = client_id
        score_record = _array_record(reply.content)
        uplink_bytes += int(score_record.count_bytes())
        score_dicts.append(score_record.to_torch_state_dict())
        weights.append(_metric_float(metrics["num-examples"]))

    if errors:
        raise RuntimeError(f"Saliency discovery failed for nodes: {errors}")
    if len(score_dicts) != len(node_ids):
        missing = set(node_ids) - set(node_to_client)
        raise RuntimeError(
            f"Incomplete saliency discovery: got {len(score_dicts)}/{len(node_ids)}; "
            f"missing nodes={sorted(missing)}"
        )
    return node_to_client, score_dicts, weights, uplink_bytes


def _install_masks(
    grid: Grid,
    *,
    masks: dict[str, torch.Tensor],
    digest: str,
    node_ids: list[int],
    timeout: float,
) -> int:
    """Install masks and return total client downlink payload bytes."""
    mask_record = ArrayRecord(masks_to_cpu_uint8(masks))
    per_client_bytes = int(mask_record.count_bytes())
    config = ConfigRecord({"mask-version": digest, "phase": "install_mask"})
    messages = [
        Message(
            content=RecordDict({"arrays": mask_record, "config": config}),
            message_type="query.install_mask",
            dst_node_id=node_id,
            group_id="install-mask",
        )
        for node_id in node_ids
    ]
    replies = list(grid.send_and_receive(messages, timeout=timeout))
    acks = 0
    for reply in replies:
        if reply.has_error():
            raise RuntimeError(
                f"Mask install failed on node {reply.metadata.src_node_id}: "
                f"{reply.error.reason}"
            )
        got = str(reply.content["config"]["mask-version"])
        if got != digest:
            raise RuntimeError(f"Mask ack digest mismatch: {got} != {digest}")
        acks += 1
    if acks != len(node_ids):
        raise RuntimeError(f"Mask install incomplete: {acks}/{len(node_ids)} acks")
    return per_client_bytes * len(node_ids)


def _global_mask_density(masks: dict[str, torch.Tensor]) -> float:
    total = sum(int(m.numel()) for m in masks.values())
    if total == 0:
        return 0.0
    active = sum(int(m.sum().item()) for m in masks.values())
    return active / total


def _save_checkpoint(
    directory: Path,
    *,
    arrays: ArrayRecord,
    masks: dict[str, torch.Tensor],
    digest: str,
    server_round: int | str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"round_{server_round}" if isinstance(server_round, int) else str(server_round)
    )
    torch.save(arrays.to_torch_state_dict(), directory / f"{stem}_model.pt")
    torch.save(masks_to_cpu_uint8(masks), directory / f"{stem}_mask.pt")
    (directory / f"{stem}_mask_version.txt").write_text(digest + "\n")


def _reset_jsonl(path: Path) -> None:
    """Replace an existing JSONL log so reruns do not mix events."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run SSFL mask discovery and federated training."""
    # This function intentionally keeps the complete server lifecycle together.
    # pylint: disable=too-many-locals,too-many-statements
    cfg = context.run_config
    seed = int(cfg["seed"])
    seed_everything(seed)

    dataset_name = str(cfg["dataset"])
    model_name = str(cfg["model"])
    num_clients = int(cfg["num-clients"])
    num_rounds = int(cfg["num-server-rounds"])
    dense_ratio = float(cfg["dense-ratio"])
    timeout = 3600.0
    checkpoint_dir = Path(str(cfg.get("checkpoint-dir", "outputs")))
    save_model = _cfg_bool(cfg, "save-model", False)
    save_checkpoints = _cfg_bool(cfg, "save-checkpoints", False)
    save_metrics = _cfg_bool(cfg, "save-metrics", True)
    metrics_history: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    wandb_session = WandbSession()
    wandb_session.start(dict(cfg))

    comm = CommStats()
    comm.notes.append(
        "Values are ArrayRecord payload bytes, not full serialized "
        "Message / wire bytes."
    )

    log(
        INFO,
        "SSFL ServerApp starting (dataset=%s, model=%s, clients=%s)",
        dataset_name,
        model_name,
        num_clients,
    )
    if save_metrics:
        _reset_jsonl(checkpoint_dir / "metrics.jsonl")
        log(INFO, "Metrics will be written under %s", checkpoint_dir)

    # Wait for the simulation/deployment cohort.
    node_ids = _wait_for_nodes(grid, expected=num_clients)
    log(INFO, "Connected nodes: %s", len(node_ids))

    # Initialize global model once on the server.
    model = create_model(model_name, num_classes_for_dataset(dataset_name))
    arrays = ArrayRecord(model.state_dict())

    # Distributed mask discovery.
    log(INFO, "Starting saliency discovery on all clients...")
    discovery_downlink = int(arrays.count_bytes()) * len(node_ids)
    node_to_client, score_dicts, sample_weights, discovery_uplink = (
        _run_saliency_discovery(grid, arrays=arrays, node_ids=node_ids, timeout=timeout)
    )
    comm.discovery_downlink_payload_bytes = discovery_downlink
    comm.discovery_uplink_payload_bytes = discovery_uplink

    weighting = str(cfg["saliency-weighting"])
    if weighting == "uniform":
        avg_scores = get_mean_saliency_scores(score_dicts)
    elif weighting == "num-examples":
        avg_scores = get_weighted_mean_saliency_scores(score_dicts, sample_weights)
    else:
        raise ValueError(f"Unsupported saliency-weighting: {weighting}")

    masks, layer_density = create_mask_from_scores(
        avg_scores, keep_ratio=dense_ratio, device="cpu"
    )
    digest = mask_digest(masks)
    global_density = _global_mask_density(masks)
    mean_layer_density = sum(layer_density.values()) / max(len(layer_density), 1)
    log(
        INFO,
        "Created global mask digest=%s global-density=%.4f "
        "mean-layer-density=%.4f keep-ratio=%.4f",
        digest,
        global_density,
        mean_layer_density,
        dense_ratio,
    )
    mask_metrics = {
        "event": "mask_created",
        "server_round": 0,
        "mask_digest": digest,
        "mask/global_density": global_density,
        "mask/mean_layer_density": mean_layer_density,
        "mask/keep_ratio": dense_ratio,
        "comm/discovery_downlink_payload_bytes": discovery_downlink,
        "comm/discovery_uplink_payload_bytes": discovery_uplink,
    }
    if save_metrics:
        metrics_history.append(mask_metrics)
        _append_jsonl(checkpoint_dir / "metrics.jsonl", mask_metrics)

    log(INFO, "Installing mask on all clients...")
    mask_downlink = _install_masks(
        grid, masks=masks, digest=digest, node_ids=node_ids, timeout=timeout
    )
    comm.mask_downlink_payload_bytes = mask_downlink
    mask_install_metrics = {
        "event": "mask_installed",
        "server_round": 0,
        "comm/mask_downlink_payload_bytes": mask_downlink,
    }
    if save_metrics:
        metrics_history.append(mask_install_metrics)
        _append_jsonl(checkpoint_dir / "metrics.jsonl", mask_install_metrics)

    # One W&B step-0 record for discovery + mask install. Keep the step open
    # when centralized eval will also log at round 0.
    evaluate_every = int(cfg["evaluate-every"])
    init_wandb = {
        **{key: value for key, value in mask_metrics.items() if key != "event"},
        **{key: value for key, value in mask_install_metrics.items() if key != "event"},
        "event": "init",
    }
    wandb_session.log(init_wandb, step=0, commit=evaluate_every <= 0)

    # Apply mask to the initial global model before training rounds.
    masked_init = apply_mask_to_state_dict(arrays.to_torch_state_dict(), masks)
    arrays = ArrayRecord(masked_init)

    if save_checkpoints or save_model:
        _save_checkpoint(
            checkpoint_dir,
            arrays=arrays,
            masks=masks,
            digest=digest,
            server_round="init",
        )
        log(INFO, "Saved init checkpoint under %s", checkpoint_dir)

    transport = str(cfg["transport"])
    log(INFO, "Training transport mode: %s", transport)
    # For paper-scale frac=0.1 with 100 clients, require at least 2 train nodes,
    # but prefer the natural sample size from fraction_train.
    min_train = 2 if num_clients >= 2 else 1
    strategy = SSFLStrategy(
        node_to_client_id=node_to_client,
        masks=masks,
        mask_version=digest,
        sample_seed=seed,
        transport=transport,
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
        min_train_nodes=min_train,
        min_evaluate_nodes=0 if float(cfg["fraction-evaluate"]) == 0.0 else 2,
        min_available_nodes=min(2, num_clients),
    )

    if evaluate_every > 0:

        def _evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if server_round != 0 and server_round % evaluate_every != 0:
                return None
            eval_model = create_model(model_name, num_classes_for_dataset(dataset_name))
            state = arrays.to_torch_state_dict()
            eval_model.load_state_dict(state)
            testloader = load_centralized_testloader(
                dataset_name, data_path=str(cfg.get("data-path", ""))
            )
            metrics = evaluate_model(
                eval_model, testloader, device=_device(prefer_cpu=True), masks=masks
            )
            log(
                INFO,
                "Centralized eval round=%s accuracy=%.4f loss=%.4f",
                server_round,
                metrics["accuracy"],
                metrics["loss"],
            )
            eval_record = {
                "event": "eval",
                "server_round": server_round,
                "eval/accuracy": metrics["accuracy"],
                "eval/loss": metrics["loss"],
                "eval/num_examples": metrics["num-examples"],
            }
            wandb_session.log(eval_record, step=server_round, commit=True)
            if save_metrics:
                metrics_history.append(eval_record)
                _append_jsonl(checkpoint_dir / "metrics.jsonl", eval_record)
            if save_checkpoints:
                _save_checkpoint(
                    checkpoint_dir,
                    arrays=arrays,
                    masks=masks,
                    digest=digest,
                    server_round=server_round,
                )
            return MetricRecord(
                {
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                    "num-examples": metrics["num-examples"],
                }
            )

        evaluate_fn = _evaluate_fn
    else:
        evaluate_fn = None

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        timeout=timeout,
        train_config=ConfigRecord({"lr": float(cfg["learning-rate"])}),
        evaluate_fn=evaluate_fn,
    )

    # Aggregate reported training traffic from strategy metrics when present.
    comm.train_downlink_payload_bytes = int(strategy.train_downlink_payload_bytes)
    for round_idx, round_metrics in sorted(result.train_metrics_clientapp.items()):
        if "arrayrecord_payload_bytes" in round_metrics:
            comm.train_uplink_payload_bytes += int(
                _metric_float(round_metrics["arrayrecord_payload_bytes"])
            )
        if "comm_params" in round_metrics:
            comm.train_comm_params += int(_metric_float(round_metrics["comm_params"]))
        if save_metrics:
            train_record = {
                "event": "train",
                "server_round": int(round_idx),
                "comm/train_downlink_payload_bytes": float(
                    strategy.train_downlink_by_round.get(int(round_idx), 0)
                ),
                **{
                    f"train/{key}": (
                        float(value) if isinstance(value, (int, float)) else value
                    )
                    for key, value in dict(round_metrics).items()
                },
            }
            metrics_history.append(train_record)
            _append_jsonl(checkpoint_dir / "metrics.jsonl", train_record)

    for line in comm.summary_lines():
        log(INFO, "%s", line)
    wandb_session.log(comm.as_dict())

    if save_model:
        _save_checkpoint(
            checkpoint_dir,
            arrays=result.arrays,
            masks=masks,
            digest=digest,
            server_round="final",
        )
        log(INFO, "Saved final checkpoint under %s", checkpoint_dir)

    if save_metrics:
        eval_points = [row for row in metrics_history if row.get("event") == "eval"]
        final_eval = eval_points[-1] if eval_points else {}
        summary = {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "num_clients": num_clients,
            "num_server_rounds": num_rounds,
            "dense_ratio": dense_ratio,
            "transport": transport,
            "mask_digest": digest,
            "mask_global_density": global_density,
            "checkpoint_dir": str(checkpoint_dir),
            "final_eval_accuracy": final_eval.get("eval/accuracy"),
            "final_eval_loss": final_eval.get("eval/loss"),
            "final_eval_round": final_eval.get("server_round"),
            "communication": comm.as_dict(),
            "config": {str(k): cfg[k] for k in dict(cfg)},
        }
        _write_json(checkpoint_dir / "summary.json", summary)

        log(
            INFO,
            "Wrote metrics to %s (final accuracy=%s)",
            checkpoint_dir / "summary.json",
            summary["final_eval_accuracy"],
        )

    wandb_session.finish()
    log(INFO, "SSFL ServerApp finished.")
