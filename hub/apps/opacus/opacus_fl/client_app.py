"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from opacus import PrivacyEngine
from opacus_fl.task import Net, load_data, test, train

warnings.filterwarnings("ignore", category=UserWarning)


app = ClientApp()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _partition_loaders(context: Context):
    pid = context.node_config["partition-id"]
    n = context.node_config["num-partitions"]
    noise = 1.0 if pid % 2 == 0 else 1.5
    train_loader, test_loader = load_data(partition_id=pid, num_partitions=n)
    return pid, noise, train_loader, test_loader


def _unwrap_state_dict(model: torch.nn.Module) -> dict:
    # NOTE: this is to return plain or unwrapped state_dict even if Opacus wrapped the model.
    return (
        model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    )


@app.train()
def train_message(msg: Message, context: Context) -> Message:
    pid, noise_multiplier, train_loader, _ = _partition_loaders(context)
    device = _device()

    target_delta = float(context.run_config["target-delta"])
    max_grad_norm = float(context.run_config["max-grad-norm"])

    model = Net().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    privacy_engine = PrivacyEngine(secure_mode=False)
    private_model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    epsilon = train(
        private_model,
        private_train_loader,
        privacy_engine,
        optimizer,
        target_delta,
        device=device,
        epochs=1,
    )

    out_arrays = ArrayRecord(_unwrap_state_dict(private_model))
    out_metrics = MetricRecord(
        {
            "num-examples": len(private_train_loader.dataset),
            "epsilon": float(epsilon),
            "target_delta": float(target_delta),
            "noise_multiplier": float(noise_multiplier),
            "max_grad_norm": float(max_grad_norm),
        }
    )

    print(
        f"[client {pid}] epsilon(delta={target_delta})={epsilon:.2f}, noise={noise_multiplier}"
    )

    return Message(
        content=RecordDict({"arrays": out_arrays, "metrics": out_metrics}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_message(msg: Message, context: Context) -> Message:
    pid, _, _, test_loader = _partition_loaders(context)
    device = _device()

    model = Net().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    loss, accuracy = test(model, test_loader, device)

    out_metrics = MetricRecord(
        {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "num-examples": len(test_loader.dataset),
        }
    )

    print(f"[client {pid}] eval loss={loss:.4f}, acc={accuracy:.4f}")

    return Message(content=RecordDict({"metrics": out_metrics}), reply_to=msg)
