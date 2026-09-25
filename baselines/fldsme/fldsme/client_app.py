"""fldsme: Flower Baseline - FL over IEEE 802.15.4e DSME IoT Networks."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fldsme.dataset import load_data
from fldsme.model import Net, get_device
from fldsme.model import test as test_fn
from fldsme.model import train as train_fn
from fldsme.null_mac_model import build_mac_model

# Approximate size of Net() in KB: 61,770 params * 4 bytes / 1024
MODEL_SIZE_KB = 61770 * 4 / 1024

app = ClientApp()


def _build_mac_model(context: Context, num_partitions: int):
    """Build the MAC model selected by ``dsme-enabled`` in the run config.

    Returns a DSMEMACModel normally, or a NullMACModel for the control arm.
    Both satisfy the same interface, so nothing downstream branches on which.
    """
    cfg = context.run_config
    return build_mac_model(
        cfg,
        bo=int(cfg["bo"]),
        mo=int(cfg["mo"]),
        so=int(cfg["so"]),
        num_clients=num_partitions,
        num_clusters=int(cfg["num-clusters"]),
        energy_budget=float(cfg["energy-budget-mj"]),
        bandwidth_frac=float(cfg["bandwidth-fraction"]),
        seed=int(cfg.get("seed", 0)),
    )


def _apply_gts_mask(trained_state, global_state, bw_frac: float) -> dict:
    """Top-k sparsify the update delta to model GTS bandwidth limiting.

    We sparsify the UPDATE (trained - global), not the absolute weights.
    Zeroing absolute weights would corrupt the global model during FedAvg
    aggregation (P1 bot review fix). The transmitted tensor is reconstructed
    as ``global + sparse_update``.
    """
    masked = {}
    for key in trained_state:
        trained_tensor = trained_state[key].float()
        # Move global tensor to the same device as the trained tensor to avoid
        # a device mismatch when train_fn runs on CUDA or MPS.
        global_tensor = global_state[key].float().to(trained_tensor.device)
        update = (trained_tensor - global_tensor).flatten().clone()
        n_keep = max(1, int(len(update) * bw_frac))
        if n_keep < len(update):
            # Keep only the largest-magnitude update components
            _, idx = torch.topk(update.abs(), n_keep)
            sparse_update = torch.zeros_like(update)
            sparse_update[idx] = update[idx]
        else:
            sparse_update = update
        masked[key] = (
            global_tensor + sparse_update.reshape(global_tensor.shape)
        ).to(trained_state[key].dtype)
    return masked


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data, gated by DSME MAC-layer constraints."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    fl_round = int(msg.content.config_records["config"]["server-round"])

    local_epochs = int(context.run_config["local-epochs"])

    mac_model = _build_mac_model(context, num_partitions)
    profile = mac_model.get_client_profile(
        client_id=partition_id,
        fl_round=fl_round,
        model_size_kb=MODEL_SIZE_KB,
        n_local_epochs=local_epochs,
    )
    eligible = mac_model.is_eligible(
        profile,
        model_size_kb=MODEL_SIZE_KB,
        n_local_epochs=local_epochs,
    )

    model = Net()
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()

    # Pre-compute values used in both branches so keys are always identical
    energy_cost = mac_model.energy_per_round_mj(
        MODEL_SIZE_KB, local_epochs, profile.cap_mode
    )
    bw_frac = mac_model.effective_bandwidth_fraction(
        partition_id, fl_round, profile.cap_mode
    )

    if not eligible:
        # Energy-depleted: return unchanged weights, num-examples=0
        # ALL metric keys must match the active-client branch exactly
        metrics = {
            "train_loss": 0.0,
            "num-examples": 0,
            "skipped": 1,
            "energy_used_mj": float(energy_cost),
            "bandwidth_frac": float(bw_frac),
            "cluster_id": float(profile.cluster_id),
            "gts_slots": float(profile.gts_slots),
            # Reported so the server can track per-node energy state.
            # Negative means no budget model is in effect (control arm).
            "client_id": float(partition_id),
            "residual_mj": float(profile.energy_budget_mj),
            "cap_ncr": float(profile.cap_mode == "NCR"),
        }
        metric_record = MetricRecord(metrics)
        model_record = ArrayRecord(model.state_dict())
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

    # Load local data and train.
    # Seed per (run seed, partition, round) so the DataLoader shuffle and any
    # other client-side randomness are reproducible. Without this, two runs
    # with the same seed still train on differently-ordered batches, which is
    # why round-1 train_loss did not match across arms.
    client_seed = (
        int(context.run_config.get("seed", 0)) * 100_003
        + partition_id * 101
        + fl_round
    )
    torch.manual_seed(client_seed)
    trainloader, _ = load_data(partition_id, num_partitions)
    train_loss = train_fn(model, trainloader, local_epochs, device)

    # Apply GTS bandwidth mask (top-k sparsification on the update delta).
    # We sparsify the UPDATE (trained - global), not the absolute weights.
    # Zeroing absolute weights would corrupt the model (P1 bot review fix).
    # Reconstruct as: global_weights + sparse_update for FedAvg aggregation.
    global_state = arrays.to_torch_state_dict()   # received global weights
    trained_state = model.state_dict()             # locally trained weights

    # Full uplink (control arm, or a DSME round where this client won all its
    # GTS slots): nothing to drop, so skip the sort entirely.
    if bw_frac < 1.0:
        masked = _apply_gts_mask(trained_state, global_state, bw_frac)
        model.load_state_dict(masked)

    # ALL metric keys must match the skipped-client branch exactly
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": int(len(trainloader.dataset)),
        "skipped": 0,
        "energy_used_mj": float(energy_cost),
        "bandwidth_frac": float(bw_frac),
        "cluster_id": float(profile.cluster_id),
        "gts_slots": float(profile.gts_slots),
        "client_id": float(partition_id),
        "residual_mj": float(profile.energy_budget_mj),
        "cap_ncr": float(profile.cap_mode == "NCR"),
    }
    metric_record = MetricRecord(metrics)
    model_record = ArrayRecord(model.state_dict())
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net()
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader = load_data(partition_id, num_partitions)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": int(len(valloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
