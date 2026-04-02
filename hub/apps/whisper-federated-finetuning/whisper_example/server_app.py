"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

from logging import INFO
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from whisper_example.dataset import get_encoding_fn
from whisper_example.model import eval_model, get_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    num_classes = context.run_config["num-classes"]
    fraction_train = context.run_config["fraction-train"]

    # Initialize global model parameters. Recall we are
    # only federating the classification head
    _, classifier = get_model("cpu", num_classes, False)
    arrays = ArrayRecord(classifier.state_dict())

    eval_fn = None
    if context.run_config["central-eval"]:
        # The ServerApp will use the validation set to assess the performance of the global
        # model after each round. Then, the test set will be used for evaluating the global
        # model after the last round
        sc_val = load_dataset(
            "speech_commands", "v0.02", split="validation", token=False
        )
        sc_test = load_dataset("speech_commands", "v0.02", split="test", token=False)

        # Processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

        eval_fn = get_evaluate_fn(sc_val, sc_test, processor, context.run_config)

    # Initialize FedAvg strategy
    strategy = ExclusiveFedAvg(fraction_train=fraction_train, fraction_evaluate=0.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=eval_fn,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_evaluate_fn(
    val_set, test_set, processor: WhisperProcessor, run_config: UserConfig
):
    """Return a callback that the strategy will call after models are aggregated."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        num_rounds = run_config["num-server-rounds"]
        num_classes = run_config["num-classes"]
        remove_cols = run_config["remove-cols"]
        remove_cols = remove_cols.split(",")

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare model
        encoder, classifier = get_model(device, num_classes)
        classifier.load_state_dict(arrays.to_torch_state_dict())

        # prepare dataset
        og_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        encoding_fn = get_encoding_fn(processor)
        if server_round == num_rounds:
            prefix = "test"
            encoded = test_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)
        else:
            prefix = "val"
            encoded = val_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)

        torch.set_num_threads(og_threads)
        val_encoded = encoded.with_format("torch", columns=["data", "targets"])
        val_loader = DataLoader(val_encoded, batch_size=64, num_workers=4)

        # Run global evaluation
        criterion = torch.nn.CrossEntropyLoss()
        loss, accuracy = eval_model(encoder, classifier, criterion, val_loader, device)

        return MetricRecord({f"{prefix}_loss": loss, f"{prefix}_accuracy": accuracy})

    return global_evaluate


class ExclusiveFedAvg(FedAvg):

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        # Clients with not enough training examples to have a single full batch
        # didn't train the classification head. We need to exclude it from aggregation

        filtered_replies = []

        for reply in replies:
            if reply.has_content():
                # Here the assumption is that there is only one config record in the reply
                record_key = next(iter(reply.content.config_records.keys()))
                is_trained = reply.content[record_key]["trained"]
                if not isinstance(is_trained, bool):
                    raise ValueError(
                        f"Expected 'trained' to be of type bool, but got {type(is_trained)}"
                    )
                if is_trained:
                    filtered_replies.append(reply)
        log(
            INFO,
            f"{len(filtered_replies)}/{len(replies)} models included for aggregation.",
        )

        return super().aggregate_train(server_round, filtered_replies)
