"""fl_dsme_iot: Flower Baseline - FL over IEEE 802.15.4e DSME IoT Networks."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_dsme_iot.dataset import load_data
from fl_dsme_iot.dsme_model import DSMEMACModel
from fl_dsme_iot.model import Net
from fl_dsme_iot.model import test as test_fn
from fl_dsme_iot.model import train as train_fn

# Approximate size of Net() in KB: 61,770 params * 4 bytes / 1024
MODEL_SIZE_KB = 61770 * 4 / 1024

app = ClientApp()


def _build_mac_model(context: Context, num_partitions: int) -> DSMEMACModel:
    """Build DSMEMACModel from run config."""
    cfg = context.run_config
    return DSMEMACModel(
        bo=int(cfg["bo"]),
        mo=int(cfg["mo"]),
        so=int(cfg["so"]),
        num_clients=num_partitions,
        num_clusters=int(cfg["num-clusters"]),
        energy_budget=float(cfg["energy-budget-mj"]),
        bandwidth_frac=float(cfg["bandwidth-fraction"]),
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data, gated by DSME MAC-layer constraints."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    fl_round = int(msg.content.config_records["config"]["server-round"])

    mac_model = _build_mac_model(context, num_partitions)
    profile = mac_model.get_client_profile(
        client_id=partition_id,
        fl_round=fl_round,
        model_size_kb=MODEL_SIZE_KB,
    )
    eligible = mac_model.is_eligible(profile, model_size_kb=MODEL_SIZE_KB)

    model = Net()
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Pre-compute values used in both branches so keys are always identical
    energy_cost = mac_model.energy_per_round_mj(
        MODEL_SIZE_KB, int(context.run_config["local-epochs"]), profile.cap_mode
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
        }
        metric_record = MetricRecord(metrics)
        model_record = ArrayRecord(model.state_dict())
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

    # Load local data and train
    trainloader, _ = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    train_loss = train_fn(model, trainloader, local_epochs, device)

    # Apply GTS bandwidth mask (top-k sparsification)
    state_dict = model.state_dict()
    masked = {}
    for key, tensor in state_dict.items():
        flat = tensor.flatten().clone()
        n_keep = max(1, int(len(flat) * bw_frac))
        if n_keep < len(flat):
            _, idx = torch.topk(flat.abs(), n_keep)
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0
            masked[key] = (flat * mask).reshape(tensor.shape)
        else:
            masked[key] = tensor.clone()
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
