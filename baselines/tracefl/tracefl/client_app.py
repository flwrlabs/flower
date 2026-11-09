"""tracefl-baseline: A Flower Baseline."""

import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

try:
    from transformers import DefaultDataCollator
except ImportError:
    # Fallback for older transformers versions
    from transformers.data.data_collator import (
        default_data_collator as DefaultDataCollator,
    )

from tracefl.config import create_tracefl_config
from tracefl.dataset import get_clients_server_data
from tracefl.model import initialize_model
from tracefl.model import test as test_fn
from tracefl.model import train as train_fn

# Flower ClientApp
app = ClientApp()

# Global variable to store client data (loaded once per client)
_CLIENT_DATA = None


def _load_client_data(context):
    """Load client data using TraceFL dataset preparation."""
    global _CLIENT_DATA  # pylint: disable=global-statement
    if _CLIENT_DATA is None:
        # Create TraceFL config from Flower context
        cfg = create_tracefl_config(context)

        # Load dataset
        ds_dict = get_clients_server_data(cfg)

        # Get client ID from context (partition-id)
        client_id = context.node_config.get("partition-id", "0")

        # Client ID resolution (with fallback for robustness)
        # Log available clients to help diagnose partition mismatches
        logging.debug("Available client IDs: %s", list(ds_dict["client2data"].keys()))
        logging.debug("Requested client ID: %s (type: %s)", client_id, type(client_id))

        # Check if client ID exists (handle both string and int types)
        if str(client_id) not in [str(k) for k in ds_dict["client2data"].keys()]:
            print(
                f"❌ Client ID {client_id} not found! Available: "
                f"{list(ds_dict['client2data'].keys())}"
            )
            # Use the first available client ID as fallback
            client_id = list(ds_dict["client2data"].keys())[0]
            print(f"🔄 Using fallback client ID: {client_id}")
        else:
            # Convert to string to match dataset keys
            client_id = str(client_id)

        # Store client data
        _CLIENT_DATA = {
            "train_data": ds_dict["client2data"][client_id],
            "test_data": ds_dict["client2data"][client_id],  # Use same data for now
            "client_id": client_id,
        }

        print(
            f"📊 Client {client_id} loaded "
            f"{len(_CLIENT_DATA['train_data'])} training samples"
        )

    return _CLIENT_DATA


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Get TraceFL config
    cfg = create_tracefl_config(context)

    # Initialize model based on configuration
    model_dict = initialize_model(cfg.data_dist.model_name, cfg.data_dist)
    model = model_dict["model"]
    arrays = msg.content.get("arrays")
    if arrays and hasattr(arrays, "to_torch_state_dict"):
        model.load_state_dict(arrays.to_torch_state_dict())
    else:
        logging.warning("No valid arrays found in message content")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load TraceFL client data
    client_data = _load_client_data(context)
    train_data = client_data["train_data"]

    # Convert to DataLoader with DefaultDataCollator for proper tensor batching
    from torch.utils.data import DataLoader  # pylint: disable=import-outside-toplevel

    trainloader = DataLoader(
        train_data,
        batch_size=cfg.data_dist.batch_size,
        shuffle=True,
        collate_fn=DefaultDataCollator(),
    )

    local_epochs = context.run_config["local-epochs"]

    # Determine model type for training using architecture detection
    model_type = (
        "transformer" if cfg.data_dist.model_architecture == "transformer" else "cnn"
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        device,
        model_type,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(train_data),
        "client_id": int(client_data["client_id"]),  # Include actual client ID
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Get TraceFL config
    cfg = create_tracefl_config(context)

    # Initialize model based on configuration
    model_dict = initialize_model(cfg.data_dist.model_name, cfg.data_dist)
    model = model_dict["model"]
    arrays = msg.content.get("arrays")
    if arrays and hasattr(arrays, "to_torch_state_dict"):
        model.load_state_dict(arrays.to_torch_state_dict())
    else:
        logging.warning("No valid arrays found in message content")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load TraceFL client data
    client_data = _load_client_data(context)
    test_data = client_data["test_data"]

    # Convert to DataLoader with DefaultDataCollator for proper tensor batching
    from torch.utils.data import DataLoader  # pylint: disable=import-outside-toplevel

    valloader = DataLoader(
        test_data,
        batch_size=cfg.data_dist.batch_size,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )

    # Determine model type for evaluation using architecture detection
    model_type = (
        "transformer" if cfg.data_dist.model_architecture == "transformer" else "cnn"
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device, model_type)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(test_data),
        # Note: client_id is a string, so we can't include it in Flower metrics
        # Flower MetricRecord only accepts numeric values
        # (int, float, list[int], list[float])
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
