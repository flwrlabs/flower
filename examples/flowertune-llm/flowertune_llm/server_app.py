"""flowertune-llm: A Flower / FlowerTune app."""

import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.benchmark import evaluate_instruction_model
from flowertune_llm.models import (
    get_model,
    is_communicated_adapter_parameter,
)
from flowertune_llm.pretraining import (
    CODE_VALIDATION_URL,
    evaluate_continued_pretraining,
)
from flowertune_llm.fedavgstreaming import FedAvgStreaming

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    strategy_timeout = float(context.run_config.get("strategy.timeout", 3600))
    if strategy_timeout <= 0:
        raise ValueError("strategy.timeout must be greater than 0 seconds")
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model, for_server=True)
    init_state_dict = init_model.state_dict()
    trainer_backend = str(context.run_config.get("trainer.backend", "none"))
    adapter_only_backend = trainer_backend in {
        "benchmark-lora",
        "benchmark-qlora",
        "benchmark-dora",
        "benchmark-qdora",
        "benchmark-rslora",
        "benchmark-qffa",
        "pretraining-lora",
        "pretraining-qlora",
        "pretraining-dora",
        "pretraining-qdora",
        "pretraining-rslora",
        "pretraining-ffa",
        "pretraining-qffa",
    }
    if adapter_only_backend:
        # A conventional LoRA federation communicates adapters only.  Every
        # participant independently loads the same frozen pretrained base.
        init_state_dict = {
            name: tensor
            for name, tensor in init_state_dict.items()
            if is_communicated_adapter_parameter(
                name, freeze_a=bool(cfg.model.lora.get("freeze_a", False))
            )
        }
    arrays = ArrayRecord()

    # Define strategy
    strategy = FedAvgStreaming(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        initial_state_dict=init_state_dict,
    )

    validation_enabled = bool(context.run_config.get("validation.enabled", False))
    validation_input_ids = None
    validation_logits = None
    if validation_enabled:
        parameter = next(init_model.parameters())
        generator = torch.Generator(device=parameter.device).manual_seed(7)
        validation_input_ids = torch.randint(
            0,
            int(init_model.config.vocab_size),
            (1, int(context.run_config.get("validation.sequence-length", 32))),
            generator=generator,
            device=parameter.device,
        )
        init_model.eval()
        with torch.inference_mode():
            validation_logits = (
                init_model(input_ids=validation_input_ids, use_cache=False)
                .logits.float()
                .cpu()
            )

    # Start strategy, run FedAvg for `num_rounds`.
    # If federated evaluation is disabled, skip server-side global evaluation too.
    evaluate_fn = None
    benchmark_validation = bool(
        context.run_config.get("train.benchmark.validation-enabled", False)
    )
    pretraining_validation = bool(
        context.run_config.get("train.pretraining.validation-enabled", False)
    )
    if pretraining_validation:
        evaluation_history: list[dict[str, object]] = []

        def pretraining_evaluate(
            server_round: int, round_arrays: ArrayRecord
        ) -> MetricRecord:
            round_state = round_arrays.to_torch_state_dict()
            init_model.load_state_dict(
                round_state,
                strict=not adapter_only_backend,
            )
            values = evaluate_continued_pretraining(
                init_model,
                corpus_url=str(
                    context.run_config.get(
                        "train.pretraining.corpus-url",
                        "https://huggingface.co/datasets/bigcode/"
                        "the-stack-smol-xs/resolve/main/data/python/data.json",
                    )
                ),
                validation_corpus_url=str(
                    context.run_config.get(
                        "train.pretraining.validation-corpus-url",
                        CODE_VALIDATION_URL,
                    )
                ),
                validation_documents=int(
                    context.run_config.get(
                        "train.pretraining.validation-documents", 20
                    )
                ),
                validation_windows=int(
                    context.run_config.get("train.pretraining.validation-windows", 16)
                ),
                seq_length=int(
                    context.run_config.get("train.pretraining.seq-length", 128)
                ),
                gpu_lock_path=str(
                    context.run_config.get(
                        "train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"
                    )
                ),
                evaluation_batch_size=int(
                    context.run_config.get(
                        "train.pretraining.validation-batch-size", 8
                    )
                ),
            )
            if bool(
                context.run_config.get(
                    "train.pretraining.quantized-validation-enabled", False
                )
            ) and (
                server_round == num_rounds
                or (
                    server_round == 0
                    and bool(
                        context.run_config.get(
                            "train.pretraining.quantized-validation-initial", False
                        )
                    )
                )
            ):
                quantized_model = get_model(cfg.model)
                quantized_model.load_state_dict(round_state, strict=False)
                quantized_values = evaluate_continued_pretraining(
                    quantized_model,
                    corpus_url=str(
                        context.run_config.get("train.pretraining.corpus-url")
                    ),
                    validation_corpus_url=str(
                        context.run_config.get(
                            "train.pretraining.validation-corpus-url",
                            CODE_VALIDATION_URL,
                        )
                    ),
                    validation_documents=int(
                        context.run_config.get(
                            "train.pretraining.validation-documents", 20
                        )
                    ),
                    validation_windows=int(
                        context.run_config.get(
                            "train.pretraining.validation-windows", 16
                        )
                    ),
                    seq_length=int(
                        context.run_config.get(
                            "train.pretraining.seq-length", 128
                        )
                    ),
                    gpu_lock_path=str(
                        context.run_config.get(
                            "train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"
                        )
                    ),
                    evaluation_batch_size=int(
                        context.run_config.get(
                            "train.pretraining.validation-batch-size", 8
                        )
                    ),
                )
                values.update(
                    {f"quantized_{key}": value for key, value in quantized_values.items()}
                )
                del quantized_model
                torch.cuda.empty_cache()
            evaluation_history.append({"round": server_round, **values})
            output_path = str(
                context.run_config.get("train.pretraining.validation-output", "")
            ).strip()
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(evaluation_history, output_file, indent=2)
                    output_file.write("\n")
            return MetricRecord(
                {
                    key: value
                    for key, value in values.items()
                    if isinstance(value, (bool, int, float, str, bytes))
                }
            )

        evaluate_fn = pretraining_evaluate
    elif benchmark_validation:
        evaluation_history: list[dict[str, float | int]] = []

        def benchmark_evaluate(
            server_round: int, round_arrays: ArrayRecord
        ) -> MetricRecord:
            round_state = round_arrays.to_torch_state_dict()
            init_model.load_state_dict(
                round_state,
                strict=not adapter_only_backend,
            )
            values = evaluate_instruction_model(
                init_model,
                dataset_name=str(
                    context.run_config.get(
                        "train.benchmark.dataset-name", "vicgalle/alpaca-gpt4"
                    )
                ),
                validation_examples=int(
                    context.run_config.get("train.benchmark.validation-examples", 32)
                ),
                seq_length=int(
                    context.run_config.get("train.benchmark.seq-length", 128)
                ),
                gpu_lock_path=str(
                    context.run_config.get(
                        "train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"
                    )
                ),
                evaluation_batch_size=int(
                    context.run_config.get("train.benchmark.validation-batch-size", 8)
                ),
            )
            if bool(
                context.run_config.get(
                    "train.benchmark.quantized-validation-enabled", False
                )
            ) and (
                server_round == num_rounds
                or (
                    server_round == 0
                    and bool(
                        context.run_config.get(
                            "train.benchmark.quantized-validation-initial", False
                        )
                    )
                )
            ):
                quantized_model = get_model(cfg.model)
                quantized_model.load_state_dict(round_state, strict=False)
                quantized_values = evaluate_instruction_model(
                    quantized_model,
                    dataset_name=str(
                        context.run_config.get(
                            "train.benchmark.dataset-name", "vicgalle/alpaca-gpt4"
                        )
                    ),
                    validation_examples=int(
                        context.run_config.get(
                            "train.benchmark.validation-examples", 32
                        )
                    ),
                    seq_length=int(
                        context.run_config.get("train.benchmark.seq-length", 128)
                    ),
                    gpu_lock_path=str(
                        context.run_config.get(
                            "train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"
                        )
                    ),
                    evaluation_batch_size=int(
                        context.run_config.get(
                            "train.benchmark.validation-batch-size", 8
                        )
                    ),
                )
                values.update(
                    {f"quantized_{key}": value for key, value in quantized_values.items()}
                )
                del quantized_model
                torch.cuda.empty_cache()
            evaluation_history.append({"round": server_round, **values})
            output_path = str(
                context.run_config.get("train.benchmark.validation-output", "")
            ).strip()
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(evaluation_history, output_file, indent=2)
                    output_file.write("\n")
            return MetricRecord(
                {
                    key: value
                    for key, value in values.items()
                    if isinstance(value, (bool, int, float, str, bytes))
                }
            )

        evaluate_fn = benchmark_evaluate
    elif float(cfg.strategy.fraction_evaluate) > 0.0:
        evaluate_fn = get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        )

    train_cfg = dict(context.run_config)
    train_cfg["save_path"] = save_path
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        num_rounds=num_rounds,
        timeout=strategy_timeout,
        evaluate_fn=evaluate_fn,
    )
    final_state_output = str(
        context.run_config.get("train.pretraining.final-state-output", "")
    ).strip()
    if final_state_output:
        os.makedirs(
            os.path.dirname(os.path.abspath(final_state_output)), exist_ok=True
        )
        torch.save(result.arrays.to_torch_state_dict(), final_state_output)
    if validation_logits is not None and validation_input_ids is not None:
        init_model.load_state_dict(result.arrays.to_torch_state_dict(), strict=True)
        init_model.eval()
        with torch.inference_mode():
            final_logits = (
                init_model(input_ids=validation_input_ids, use_cache=False)
                .logits.float()
                .cpu()
            )
        metrics = {
            "logits_cosine": float(
                F.cosine_similarity(
                    validation_logits.flatten(), final_logits.flatten(), dim=0
                ).item()
            ),
            "logits_relative_rmse": float(
                torch.sqrt(torch.mean((final_logits - validation_logits) ** 2))
                / torch.sqrt(torch.mean(validation_logits**2)).clamp_min(1e-12)
            ),
            "top1_token_agreement": float(
                (
                    validation_logits.argmax(-1) == final_logits.argmax(-1)
                ).float().mean().item()
            ),
            "next_token_match": bool(
                validation_logits[0, -1].argmax() == final_logits[0, -1].argmax()
            ),
        }
        output_path = str(context.run_config.get("validation.output", "")).strip()
        if not output_path:
            output_path = os.path.join(save_path, "validation.json")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, indent=2)
            output_file.write("\n")
        print(f"FEDERATED_TURBOQUANT_VALIDATION={json.dumps(metrics, sort_keys=True)}")


# Get function that will be executed by the strategy
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            model.load_state_dict(arrays.to_torch_state_dict())

            model.save_pretrained(f"{save_path}/{server_round}")

        return MetricRecord()

    return evaluate
