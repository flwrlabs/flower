"""Controlled dense-versus-LoRA instruction-tuning benchmark helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import math
import os
from typing import Any, Iterator

import torch
import torch.nn.functional as F


@lru_cache(maxsize=4)
def _load_dataset(dataset_name: str):  # type: ignore[no-untyped-def]
    """Load and cache one Hugging Face instruction dataset."""
    from datasets import load_dataset

    return load_dataset(dataset_name, split="train")


def _example_fields(example: dict[str, Any]) -> tuple[str, str, str]:
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    response = str(example.get("output", example.get("response", ""))).strip()
    if not instruction or not response:
        raise ValueError("Instruction benchmark rows need instruction and output fields")
    return instruction, user_input, response


def _prompt_and_response(example: dict[str, Any]) -> tuple[str, str]:
    instruction, user_input, response = _example_fields(example)
    prompt = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n"
    )
    if user_input:
        prompt += f"\n### Input:\n{user_input}\n"
    prompt += "\n### Response:\n"
    return prompt, response


def _encode_example(tokenizer, example: dict[str, Any], seq_length: int):  # type: ignore[no-untyped-def]
    """Encode one example and mask prompt tokens from causal-LM loss."""
    prompt, response = _prompt_and_response(example)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids.append(int(eos_id))

    # Reserve at least half the window for the supervised response where possible.
    max_prompt = max(1, seq_length // 2)
    prompt_ids = prompt_ids[-max_prompt:]
    response_ids = response_ids[: max(1, seq_length - len(prompt_ids))]
    input_ids = (prompt_ids + response_ids)[:seq_length]
    labels = [-100] * len(prompt_ids) + response_ids
    labels = labels[: len(input_ids)]
    if not any(label != -100 for label in labels):
        raise ValueError("Encoded benchmark example contains no response tokens")
    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long).unsqueeze(0),
    )


def _training_indices(
    dataset_size: int,
    *,
    node_id: int,
    server_round: int,
    num_steps: int,
    num_partitions: int,
    validation_examples: int,
    initial_offset: int = 0,
) -> list[int]:
    """Return disjoint, repeatable per-node indices excluding held-out rows."""
    train_size = dataset_size - validation_examples
    if train_size <= num_partitions:
        raise ValueError("Dataset is too small for the requested held-out split")
    partition_id = node_id % num_partitions
    partition = list(range(partition_id, train_size, num_partitions))
    offset = (initial_offset + (server_round - 1) * num_steps) % len(partition)
    return [partition[(offset + step) % len(partition)] for step in range(num_steps)]


@contextmanager
def _gpu_lock(path: str) -> Iterator[None]:
    """Serialize GPU work when multiple local SuperNodes share one accelerator."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lock_file = open(path, "a+", encoding="utf-8")  # pylint: disable=R1732
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        lock_file.close()


def _configure_trainable(
    model: torch.nn.Module, *, method: str, last_layers: int
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if method == "dense":
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None or len(layers) < last_layers:
            raise ValueError("Dense benchmark model does not expose enough layers")
        if last_layers == 0:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        else:
            for layer in layers[-last_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            final_norm = getattr(backbone, "norm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad_(True)
    elif method in {"lora", "qlora", "dora", "qdora", "rslora", "qffa"}:
        for name, parameter in model.named_parameters():
            if "lora_" in name and (method != "qffa" or "lora_B" in name):
                parameter.requires_grad_(True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        raise ValueError(f"Unsupported benchmark method: {method}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters found for benchmark method {method}")
    return trainable


def run_instruction_training(
    model: torch.nn.Module,
    context,
    *,
    server_round: int,
    method: str,
) -> tuple[float, float, int, int, int, int]:
    """Run a deterministic instruction-tuning window for one federated client."""
    if not torch.cuda.is_available():
        raise RuntimeError("Instruction benchmark training requires CUDA")

    config = context.run_config
    num_steps = int(config.get("train.benchmark.steps", 25))
    seq_length = int(config.get("train.benchmark.seq-length", 128))
    learning_rate = float(config.get("train.benchmark.learning-rate", 5e-4))
    last_layers = int(config.get("train.benchmark.last-layers", 2))
    num_partitions = int(config.get("train.benchmark.num-partitions", 2))
    validation_examples = int(config.get("train.benchmark.validation-examples", 32))
    dataset_name = str(config.get("train.benchmark.dataset-name", "vicgalle/alpaca-gpt4"))
    lock_path = str(config.get("train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"))
    if min(num_steps, seq_length, num_partitions) < 1 or last_layers < 0:
        raise ValueError(
            "Benchmark steps, sequence length, and partitions must be positive; "
            "last-layers must be nonnegative"
        )

    dataset = _load_dataset(dataset_name)
    partition_id = int(
        context.node_config.get("partition-id", int(context.node_id) % num_partitions)
    )
    indices = _training_indices(
        len(dataset),
        node_id=partition_id,
        server_round=server_round,
        num_steps=num_steps,
        num_partitions=num_partitions,
        validation_examples=validation_examples,
        initial_offset=int(config.get("train.benchmark.data-seed", 0)) * 1_000_003,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=True)
    with _gpu_lock(lock_path):
        torch.cuda.reset_peak_memory_stats()
        trainable = _configure_trainable(model, method=method, last_layers=last_layers)
        trainable_count = sum(parameter.numel() for parameter in trainable)
        measure_delta_norm = bool(
            config.get("train.benchmark.measure-delta-norm", False)
        )
        before = (
            [parameter.detach().cpu().float().clone() for parameter in trainable]
            if measure_delta_norm
            else []
        )
        device = torch.device("cuda")
        quantized = bool(getattr(model, "is_loaded_in_4bit", False))
        if not quantized:
            model.to(device)
        model.train()
        model.config.use_cache = False
        if bool(config.get("train.benchmark.gradient-checkpointing", True)):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        optimizer_name = str(
            config.get(
                "train.benchmark.optimizer",
                "paged_adamw8bit" if method == "dense" else "adamw",
            )
        ).lower()
        if optimizer_name in {"adamw8bit", "paged_adamw8bit"}:
            import bitsandbytes as bnb

            optimizer_class = (
                bnb.optim.PagedAdamW8bit
                if optimizer_name == "paged_adamw8bit"
                else bnb.optim.AdamW8bit
            )
            optimizer = optimizer_class(
                trainable,
                lr=learning_rate,
                weight_decay=float(config.get("train.benchmark.weight-decay", 0.0)),
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=float(config.get("train.benchmark.weight-decay", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported benchmark optimizer: {optimizer_name}")
        total_loss = 0.0
        supervised_tokens = 0
        for index in indices:
            input_ids, labels = _encode_example(tokenizer, dataset[int(index)], seq_length)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().float().item())
            supervised_tokens += int(torch.count_nonzero(labels != -100).item())

        squared_delta = -1.0
        if measure_delta_norm:
            squared_delta = 0.0
            for parameter, original in zip(trainable, before, strict=True):
                difference = parameter.detach().cpu().float() - original
                squared_delta += float(torch.sum(difference * difference).item())
        optimizer.zero_grad(set_to_none=True)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        del optimizer, before, trainable
        if not quantized:
            model.to("cpu")
        torch.cuda.empty_cache()
    return (
        total_loss / num_steps,
        squared_delta**0.5 if squared_delta >= 0 else -1.0,
        trainable_count,
        supervised_tokens,
        peak_allocated,
        peak_reserved,
    )


def evaluate_instruction_model(
    model: torch.nn.Module,
    *,
    dataset_name: str,
    validation_examples: int,
    seq_length: int,
    gpu_lock_path: str,
    evaluation_batch_size: int = 8,
) -> dict[str, float | list[float]]:
    """Evaluate response-only held-out loss and next-token accuracy."""
    dataset = _load_dataset(dataset_name)
    if validation_examples < 1 or validation_examples >= len(dataset):
        raise ValueError("validation_examples must be between 1 and dataset size - 1")
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=True)
    rows = list(range(len(dataset) - validation_examples, len(dataset)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0.0
    window_losses: list[float] = []
    correct_tokens = 0
    total_tokens = 0
    with _gpu_lock(gpu_lock_path):
        quantized = bool(getattr(model, "is_loaded_in_4bit", False))
        if not quantized:
            model.to(device)
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(rows), evaluation_batch_size):
                encoded = [
                    _encode_example(tokenizer, dataset[int(index)], seq_length)
                    for index in rows[offset : offset + evaluation_batch_size]
                ]
                lengths = torch.tensor(
                    [input_ids.shape[1] for input_ids, _ in encoded],
                    device=device,
                )
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [item[0].squeeze(0) for item in encoded],
                    batch_first=True,
                    padding_value=int(pad_token_id),
                ).to(device)
                labels = torch.nn.utils.rnn.pad_sequence(
                    [item[1].squeeze(0) for item in encoded],
                    batch_first=True,
                    padding_value=-100,
                ).to(device)
                positions = torch.arange(input_ids.shape[1], device=device)
                attention_mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).long()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = output.logits[:, :-1].float()
                targets = labels[:, 1:]
                mask = targets != -100
                token_losses = F.cross_entropy(
                    logits.transpose(1, 2),
                    targets,
                    ignore_index=-100,
                    reduction="none",
                )
                loss_sums = torch.sum(token_losses * mask, dim=1)
                token_counts = torch.sum(mask, dim=1).clamp_min(1)
                batch_losses = (loss_sums / token_counts).cpu().tolist()
                window_losses.extend(float(loss) for loss in batch_losses)
                total_loss += sum(float(loss) for loss in batch_losses)
                predictions = logits.argmax(dim=-1)
                correct_tokens += int(torch.sum((predictions == targets) & mask).item())
                total_tokens += int(torch.sum(mask).item())
        if not quantized:
            model.to("cpu")
        torch.cuda.empty_cache()
    mean_loss = total_loss / validation_examples
    return {
        "heldout_loss": mean_loss,
        "heldout_perplexity": math.exp(min(mean_loss, 20.0)),
        "heldout_token_accuracy": correct_tokens / max(1, total_tokens),
        "heldout_supervised_tokens": float(total_tokens),
        "heldout_window_losses": window_losses,
    }
