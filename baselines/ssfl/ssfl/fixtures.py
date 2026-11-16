"""
Phase-0 fixture recipes for algorithm parity.

These helpers regenerate deterministic stage-level fixtures without depending
on the gitignored scripts/ directory. They use the Flower-port helpers, which
are intentionally copied from the legacy SSFL algorithms.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ssfl.aggregation import fedavg_weighted
from ssfl.mask import (
    apply_mask_to_state_dict,
    create_mask_from_scores,
    get_mean_saliency_scores,
    mask_digest,
)
from ssfl.model import create_model, num_classes_for_dataset
from ssfl.partitioner import partition_data_dirichlet
from ssfl.reproducibility import seed_everything
from ssfl.saliency import calculate_ssfl_scores
from ssfl.training import train_local


def generate_small_mask_fixture(
    *,
    seed: int = 550,
    n_clients: int = 4,
    dense_ratio: float = 0.5,
    batch_size: int = 16,
) -> dict:
    """
    CPU-friendly fixture: synthetic CIFAR-shaped tensors + partition + mask.

    This does not download CIFAR. It validates saliency→mask determinism on a
    fixed synthetic dataset that mirrors the CIFAR-10 label space.
    """
    seed_everything(seed)
    n_samples = n_clients * 64
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.arange(n_samples) % 10
    y = labels.numpy()

    mapping, counts = partition_data_dirichlet(
        y, n_clients=n_clients, alpha=0.3, seed=seed
    )

    model = create_model("resnet18", num_classes_for_dataset("cifar10"))
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    score_dicts = []
    client_batches = []
    for client_id in range(n_clients):
        idxs = mapping[client_id][:batch_size]
        batch = (images[idxs].clone(), labels[idxs].clone())
        client_batches.append(batch)
        model.load_state_dict(init_state)
        score_dicts.append(
            calculate_ssfl_scores(model, batch, device=torch.device("cpu"))
        )

    avg = get_mean_saliency_scores(score_dicts)
    masks, density = create_mask_from_scores(avg, keep_ratio=dense_ratio, device="cpu")
    return {
        "seed": seed,
        "n_clients": n_clients,
        "dense_ratio": dense_ratio,
        "partition_counts": counts,
        "partition_map": mapping,
        "client_batches": client_batches,
        "init_state": init_state,
        "local_scores": score_dicts,
        "avg_scores": avg,
        "masks": masks,
        "mask_digest": mask_digest(masks),
        "layer_density": density,
        "active_params": int(sum(int(m.sum().item()) for m in masks.values())),
    }


def generate_stage_oracle(
    *,
    seed: int = 550,
    n_clients: int = 2,
    dense_ratio: float = 0.5,
    batch_size: int = 16,
    local_epochs: int = 1,
    lr: float = 0.1,
    weight_decay: float = 0.0005,
) -> dict:
    """
    Generate a full stage-level oracle for Phase-0/1 parity gates.

    init model → local saliency → global mask → masked local updates → FedAvg.
    """
    fixture = generate_small_mask_fixture(
        seed=seed,
        n_clients=n_clients,
        dense_ratio=dense_ratio,
        batch_size=batch_size,
    )
    masks = fixture["masks"]
    init_state = fixture["init_state"]
    mapping = fixture["partition_map"]
    masked_init = apply_mask_to_state_dict(init_state, masks)

    local_updates: list[tuple[float, dict[str, torch.Tensor]]] = []
    local_states = []

    for client_id in range(n_clients):
        model = create_model("resnet18", num_classes_for_dataset("cifar10"))
        model.load_state_dict(masked_init)
        batch_x, batch_y = fixture["client_batches"][client_id]
        xs = batch_x.repeat(2, 1, 1, 1)
        ys = batch_y.repeat(2)
        loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False)
        train_local(
            model,
            loader,
            epochs=local_epochs,
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
            max_grad_norm=10.0,
            round_idx=1,
            lr_scheduler_name="default",
            lr_decay=0.998,
            device=torch.device("cpu"),
            masks=masks,
        )
        state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        state = apply_mask_to_state_dict(state, masks)
        n_examples = float(len(mapping[client_id]))
        local_states.append(state)
        local_updates.append((n_examples, state))

    aggregated = fedavg_weighted(local_updates)
    aggregated = apply_mask_to_state_dict(aggregated, masks)

    return {
        **fixture,
        "masked_init": masked_init,
        "local_states": local_states,
        "aggregated": aggregated,
        "local_epochs": local_epochs,
        "lr": lr,
    }


def save_fixture(
    fixture: dict, out_dir: str | Path, name: str = "small_mask_fixture.pt"
) -> Path:
    """Save the reproducibility subset of a fixture."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    payload = {
        "seed": fixture["seed"],
        "n_clients": fixture["n_clients"],
        "dense_ratio": fixture["dense_ratio"],
        "partition_counts": fixture["partition_counts"],
        "init_state": fixture["init_state"],
        "masks": fixture["masks"],
        "mask_digest": fixture.get("mask_digest"),
        "layer_density": fixture["layer_density"],
        "active_params": fixture["active_params"],
        "avg_scores": fixture.get("avg_scores"),
        "local_scores": fixture.get("local_scores"),
        "client_batches": fixture.get("client_batches"),
        "masked_init": fixture.get("masked_init"),
        "local_states": fixture.get("local_states"),
        "aggregated": fixture.get("aggregated"),
    }
    torch.save(payload, path)
    return path


def legacy_fixture_recipes() -> dict[str, str]:
    """Documented legacy CLI recipes for full CIFAR oracles (run from repo root)."""
    return {
        "mask_only": (
            "python main.py algorithm.name=ssfl algorithm.params.mode=static "
            "model.name=resnet18 dataset.name=cifar10 dataset.partition_alpha=0.3 "
            "model.dense_ratio=0.5 training.client_num_in_total=100 "
            "training.comm_round=1 experiment.seed=550 wandb.mode=offline "
            "wandb.exp_name=flower_oracle_mask"
        ),
        "one_train_round": (
            "python main.py algorithm.name=ssfl algorithm.params.mode=static "
            "model.name=resnet18 dataset.name=cifar10 dataset.partition_alpha=0.3 "
            "model.dense_ratio=0.5 training.client_num_in_total=100 training.frac=0.1 "
            "training.epochs=5 training.batch_size=16 training.comm_round=2 "
            "experiment.seed=550 wandb.mode=offline "
            "wandb.exp_name=flower_oracle_round1"
        ),
    }
