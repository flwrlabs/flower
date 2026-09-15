"""Domain-adaptive continued-pretraining benchmark helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import math
import os
from typing import Iterator

import torch

CODE_CORPUS_URL = (
    "https://huggingface.co/datasets/bigcode/the-stack-smol-xs/resolve/main/"
    "data/python/data.json"
)
CODE_VALIDATION_URL = (
    "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid/resolve/"
    "main/file-000000000054.json.gz"
)


@contextmanager
def _gpu_lock(path: str) -> Iterator[None]:
    """Serialize GPU use by local SuperNodes and the ServerApp."""
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


@lru_cache(maxsize=2)
def _load_code_corpus(url: str = CODE_CORPUS_URL):  # type: ignore[no-untyped-def]
    from datasets import load_dataset

    return load_dataset("json", data_files=url, split="train")


@lru_cache(maxsize=1)
def _load_general_validation():  # type: ignore[no-untyped-def]
    from datasets import load_dataset

    return load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )


def _tokenize_documents(tokenizer, documents: list[str]) -> list[int]:  # type: ignore[no-untyped-def]
    eos_id = tokenizer.eos_token_id
    tokens: list[int] = []
    for document in documents:
        document_tokens = tokenizer.encode(document, add_special_tokens=False)
        tokens.extend(document_tokens)
        if eos_id is not None:
            tokens.append(int(eos_id))
    return tokens


def _training_windows(
    tokens: list[int],
    *,
    server_round: int,
    steps: int,
    seq_length: int,
    initial_offset: int = 0,
) -> list[list[int]]:
    """Return deterministic non-overlapping round windows from a token stream."""
    window = seq_length
    if len(tokens) < window:
        raise ValueError("Continued-pretraining token stream is too short")
    span = len(tokens) - window + 1
    round_offset = (initial_offset + (server_round - 1) * steps * window) % span
    return [
        tokens[(round_offset + step * window) % span :][:window]
        if round_offset + step * window + window <= len(tokens)
        else (
            tokens[(round_offset + step * window) % span :]
            + tokens[:window]
        )[:window]
        for step in range(steps)
    ]


def _configure_trainable(model: torch.nn.Module, method: str):  # type: ignore[no-untyped-def]
    if method in {"dense", "galore"}:
        for parameter in model.parameters():
            parameter.requires_grad_(parameter.is_floating_point())
    elif method in {
        "lora",
        "relora",
        "qlora",
        "dora",
        "qdora",
        "rslora",
        "ffa",
        "qffa",
    }:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                ".lora_B." in name if method in {"ffa", "qffa"} else "lora_" in name
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        raise ValueError(f"Unsupported continued-pretraining method: {method}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters found for {method}")
    return trainable


def run_continued_pretraining(
    model: torch.nn.Module,
    context,
    *,
    server_round: int,
    method: str,
) -> tuple[float, int, int, int, int, int]:
    """Train one deterministic raw-code token window."""
    if not torch.cuda.is_available():
        raise RuntimeError("Continued-pretraining benchmark requires CUDA")
    config = context.run_config
    steps = int(config.get("train.pretraining.steps", 50))
    seq_length = int(config.get("train.pretraining.seq-length", 128))
    learning_rate = float(config.get("train.pretraining.learning-rate", 2e-5))
    validation_documents = int(
        config.get("train.pretraining.validation-documents", 20)
    )
    num_partitions = int(config.get("train.pretraining.num-partitions", 2))
    corpus_url = str(config.get("train.pretraining.corpus-url", CODE_CORPUS_URL))
    max_documents = int(config.get("train.pretraining.max-documents", 0))
    node_config = getattr(context, "node_config", {})
    lock_path = str(
        node_config.get(
            "client.gpu-lock",
            config.get("train.substantial-gpu-lock", "/tmp/flwr-gpu.lock"),
        )
    )
    if min(steps, seq_length, validation_documents, num_partitions) < 1:
        raise ValueError("Continued-pretraining settings must be positive")

    from transformers import AutoTokenizer

    dataset = _load_code_corpus(corpus_url)
    validation_url = str(
        config.get("train.pretraining.validation-corpus-url", corpus_url)
    )
    train_size = len(dataset)
    if validation_url == corpus_url:
        train_size -= validation_documents
    if max_documents > 0:
        train_size = min(train_size, max_documents)
    partition_id = int(
        node_config.get("partition-id", int(context.node_id) % num_partitions)
    )
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition-id must be in [0, num-partitions)")
    documents = [
        str(dataset[index]["content"])
        for index in range(partition_id, train_size, num_partitions)
    ]
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=True)
    tokens = _tokenize_documents(tokenizer, documents)
    windows = _training_windows(
        tokens,
        server_round=server_round,
        steps=steps,
        seq_length=seq_length,
        initial_offset=(
            int(config.get("train.pretraining.data-seed", 0)) * 1_000_003
            + partition_id * 97_409
        ),
    )

    with _gpu_lock(lock_path):
        torch.cuda.reset_peak_memory_stats()
        trainable = _configure_trainable(model, method)
        trainable_count = sum(parameter.numel() for parameter in trainable)
        model.config.use_cache = False
        if bool(config.get("train.pretraining.gradient-checkpointing", True)):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        device = torch.device("cuda")
        model.to(device)
        model.train()
        optimizer_name = str(
            config.get(
                "train.pretraining.optimizer",
                "adamw8bit" if method == "dense" else "adamw",
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
                trainable, lr=learning_rate, weight_decay=0.0
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable, lr=learning_rate, weight_decay=0.0
            )
        elif optimizer_name in {"galore_adamw8bit", "galore-adamw8bit"}:
            from galore_torch import GaLoreAdamW8bit

            projected = [parameter for parameter in trainable if parameter.ndim == 2]
            projected_ids = {id(parameter) for parameter in projected}
            regular = [
                parameter for parameter in trainable if id(parameter) not in projected_ids
            ]
            groups: list[dict[str, object]] = []
            if regular:
                groups.append({"params": regular})
            if projected:
                groups.append(
                    {
                        "params": projected,
                        "rank": int(config.get("train.pretraining.galore-rank", 256)),
                        "update_proj_gap": int(
                            config.get("train.pretraining.galore-update-gap", 50)
                        ),
                        "scale": float(
                            config.get("train.pretraining.galore-scale", 1.0)
                        ),
                        "proj_type": "std",
                    }
                )
            optimizer = GaLoreAdamW8bit(groups, lr=learning_rate, weight_decay=0.0)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        total_loss = 0.0
        for window in windows:
            input_ids = torch.tensor(window, device=device).unsqueeze(0)
            labels = input_ids.clone()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().float().item())
        optimizer.zero_grad(set_to_none=True)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        del optimizer, trainable
        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            model.to("cpu")
        torch.cuda.empty_cache()
    return (
        total_loss / steps,
        trainable_count,
        steps * (seq_length - 1),
        0,
        peak_allocated,
        peak_reserved,
    )


def _evaluation_windows(
    tokenizer, documents: list[str], *, seq_length: int, max_windows: int  # type: ignore[no-untyped-def]
) -> list[list[int]]:
    tokens = _tokenize_documents(tokenizer, documents)
    width = seq_length
    return [
        tokens[start : start + width]
        for start in range(0, max(0, len(tokens) - width + 1), width)
    ][:max_windows]


def _evaluate_windows(  # type: ignore[no-untyped-def]
    model, windows: list[list[int]], device, *, batch_size: int = 1
):
    if not windows:
        raise ValueError("No complete continued-pretraining evaluation windows")
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    losses: list[float] = []
    correct = 0
    tokens = 0
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            input_ids = torch.tensor(batch, device=device)
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            targets = input_ids[:, 1:]
            token_losses = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].float().transpose(1, 2),
                targets,
                reduction="none",
            )
            losses.extend(
                float(value) for value in token_losses.mean(dim=1).cpu().tolist()
            )
            predictions = output.logits[:, :-1].argmax(dim=-1)
            correct += int(torch.sum(predictions == targets).item())
            tokens += int(targets.numel())
    loss = sum(losses) / len(losses)
    variance = sum((value - loss) ** 2 for value in losses) / max(
        1, len(losses) - 1
    )
    loss_std = math.sqrt(variance)
    loss_se = loss_std / math.sqrt(len(losses))
    return (
        loss,
        math.exp(min(loss, 20.0)),
        correct / max(1, tokens),
        tokens,
        loss_std,
        loss_se,
        losses,
    )


def merge_relora_state_dict(
    state_dict: dict[str, torch.Tensor], *, rank: int, alpha: int, seed: int
) -> int:
    """Merge aggregated LoRA factors into base weights and restart adapters."""
    if rank < 1:
        raise ValueError("ReLoRA rank must be positive")
    suffix = ".lora_A.default.weight"
    merged = 0
    scale = float(alpha) / float(rank)
    with torch.no_grad():
        for adapter_a_name in [name for name in state_dict if name.endswith(suffix)]:
            prefix = adapter_a_name[: -len(suffix)]
            adapter_b_name = f"{prefix}.lora_B.default.weight"
            base_name = next(
                (
                    candidate
                    for candidate in (
                        f"{prefix}.base_layer.weight",
                        f"{prefix}.weight",
                    )
                    if candidate in state_dict
                ),
                None,
            )
            if adapter_b_name not in state_dict or base_name is None:
                continue
            adapter_a = state_dict[adapter_a_name]
            adapter_b = state_dict[adapter_b_name]
            base = state_dict[base_name]
            update = adapter_b.float() @ adapter_a.float()
            base.add_((update * scale).to(dtype=base.dtype))
            generator = torch.Generator(device=adapter_a.device).manual_seed(
                seed + merged
            )
            adapter_a.normal_(
                mean=0.0,
                std=1.0 / math.sqrt(max(1, adapter_a.shape[1])),
                generator=generator,
            )
            adapter_b.zero_()
            merged += 1
    if merged == 0:
        raise ValueError("ReLoRA state contains no mergeable default adapters")
    return merged


def evaluate_continued_pretraining(
    model: torch.nn.Module,
    *,
    corpus_url: str,
    validation_corpus_url: str,
    validation_documents: int,
    validation_windows: int,
    seq_length: int,
    gpu_lock_path: str,
    evaluation_batch_size: int = 8,
) -> dict[str, float | list[float]]:
    """Measure code-domain learning and general-text retention."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=True)
    code = _load_code_corpus(validation_corpus_url)
    if validation_corpus_url == corpus_url:
        validation_indices = range(len(code) - validation_documents, len(code))
    else:
        validation_indices = range(min(validation_documents, len(code)))
    code_documents = [str(code[index]["content"]) for index in validation_indices]
    general = _load_general_validation()
    general_documents = [
        str(row["text"]) for row in general if str(row["text"]).strip()
    ]
    code_windows = _evaluation_windows(
        tokenizer,
        code_documents,
        seq_length=seq_length,
        max_windows=validation_windows,
    )
    general_windows = _evaluation_windows(
        tokenizer,
        general_documents,
        seq_length=seq_length,
        max_windows=validation_windows,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with _gpu_lock(gpu_lock_path):
        model.to(device)
        model.eval()
        (
            code_loss,
            code_ppl,
            code_accuracy,
            code_tokens,
            code_loss_std,
            code_loss_se,
            code_window_losses,
        ) = _evaluate_windows(
            model, code_windows, device, batch_size=evaluation_batch_size
        )
        (
            general_loss,
            general_ppl,
            general_accuracy,
            general_tokens,
            general_loss_std,
            general_loss_se,
            general_window_losses,
        ) = _evaluate_windows(
            model, general_windows, device, batch_size=evaluation_batch_size
        )
        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            model.to("cpu")
        torch.cuda.empty_cache()
    return {
        "code_loss": code_loss,
        "code_perplexity": code_ppl,
        "code_token_accuracy": code_accuracy,
        "code_tokens": float(code_tokens),
        "code_loss_std": code_loss_std,
        "code_loss_se": code_loss_se,
        "general_loss": general_loss,
        "general_perplexity": general_ppl,
        "general_token_accuracy": general_accuracy,
        "general_tokens": float(general_tokens),
        "general_loss_std": general_loss_std,
        "general_loss_se": general_loss_se,
        "code_window_losses": code_window_losses,
        "general_window_losses": general_window_losses,
    }
