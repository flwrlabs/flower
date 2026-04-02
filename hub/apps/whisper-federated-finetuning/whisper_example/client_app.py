"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

import time

time.sleep(5)
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
from torch.utils.data import DataLoader

from whisper_example.dataset import load_data
from whisper_example.model import construct_balanced_sampler, get_model, train_one_epoch

torch.set_float32_matmul_precision(
    "high"
)  #  If “high” or “medium” are set then the TensorFloat32 is used

og_threads = torch.get_num_threads()


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the run config
    partition_id = context.node_config["partition-id"]
    num_classes = context.run_config["num-classes"]
    batch_size = context.run_config["batch-size"]
    disable_tqdm = context.run_config["disable-tqdm"]
    compile_model = context.run_config["compile-model"]

    # Load model and initialize it with the received weights
    # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, classifier = get_model(device, num_classes, compile_model)
    classifier.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    # Some systems seem to need this, else .map stages will hang
    # Doesn't seem to be required on macOS; but it's on Ubuntu
    # even if the latter has more CPUs...
    # ! Open a PR if you know how to improve this!
    og_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    partition = load_data(
        partition_id=partition_id,
        remove_cols=context.run_config["remove-cols"],
    )
    trainset = partition.with_format("torch", columns=["data", "targets"])
    torch.set_num_threads(og_threads)

    # construct sampler in order to have balanced batches
    sampler = None
    if len(trainset) > batch_size:
        sampler = construct_balanced_sampler(trainset)

    # Construct dataloader
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler,
        drop_last=True,
    )

    # Define optimizer and criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Don't train if trainset is very small
    run_training = len(train_loader) > 1
    config_record = ConfigRecord(
        {"trained": run_training}
    )  # will be used for aggregation
    train_metrics = {}
    if run_training:
        # Train
        avg_loss, avg_acc = train_one_epoch(
            encoder,
            classifier,
            optimizer,
            criterion,
            train_loader,
            device,
            disable_tqdm=disable_tqdm,
        )
        train_metrics = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "num-examples": len(train_loader.dataset),
        }

    # Construct and return reply Message
    model_record = ArrayRecord(classifier.state_dict())
    metric_record = MetricRecord(train_metrics)
    content = RecordDict(
        {"arrays": model_record, "metrics": metric_record, "config": config_record}
    )
    return Message(content=content, reply_to=msg)
