"""Tests for controlled dense-versus-LoRA benchmark helpers."""

from types import SimpleNamespace

import pytest
import torch

import flowertune_llm.benchmark as benchmark
from flowertune_llm.benchmark import (
    _configure_trainable,
    _encode_example,
    _training_indices,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = None

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [3 + (ord(char) % 17) for char in text]


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        )
        self.norm = torch.nn.LayerNorm(2)


class _DenseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.head = torch.nn.Linear(2, 2)


class _LoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Linear(2, 1, bias=False)
        self.lora_B = torch.nn.Linear(1, 2, bias=False)

    def enable_input_require_grads(self) -> None:
        pass


class _CausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 32, bias=False)
        self.config = SimpleNamespace(_name_or_path="test-model")

    def forward(self, input_ids, attention_mask=None, use_cache=False):  # type: ignore[no-untyped-def]
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def test_training_indices_are_disjoint_repeatable_and_held_out() -> None:
    left = _training_indices(
        100, node_id=2, server_round=2, num_steps=10, num_partitions=2,
        validation_examples=20,
    )
    right = _training_indices(
        100, node_id=3, server_round=2, num_steps=10, num_partitions=2,
        validation_examples=20,
    )
    assert set(left).isdisjoint(right)
    assert max(left + right) < 80
    assert left == _training_indices(
        100, node_id=2, server_round=2, num_steps=10, num_partitions=2,
        validation_examples=20,
    )


def test_encode_example_masks_prompt_and_keeps_response() -> None:
    input_ids, labels = _encode_example(
        _Tokenizer(),
        {"instruction": "Add", "input": "1+1", "output": "2"},
        32,
    )
    assert input_ids.shape == labels.shape
    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] <= 32
    assert torch.any(labels == -100)
    assert torch.any(labels != -100)


def test_configure_trainable_selects_only_requested_parameters() -> None:
    dense = _DenseModel()
    dense_trainable = _configure_trainable(dense, method="dense", last_layers=1)
    expected_dense = sum(
        parameter.numel()
        for parameter in [*dense.model.layers[-1].parameters(), *dense.model.norm.parameters()]
    )
    assert sum(parameter.numel() for parameter in dense_trainable) == expected_dense
    assert not any(parameter.requires_grad for parameter in dense.head.parameters())

    full_dense = _DenseModel()
    full_dense_trainable = _configure_trainable(
        full_dense, method="dense", last_layers=0
    )
    assert sum(parameter.numel() for parameter in full_dense_trainable) == sum(
        parameter.numel() for parameter in full_dense.parameters()
    )

    lora = _LoraModel()
    lora_trainable = _configure_trainable(lora, method="lora", last_layers=1)
    assert sum(parameter.numel() for parameter in lora_trainable) == 4
    assert not any(parameter.requires_grad for parameter in lora.base.parameters())

    frozen_a = _LoraModel()
    frozen_a_trainable = _configure_trainable(
        frozen_a, method="qffa", last_layers=1
    )
    assert sum(parameter.numel() for parameter in frozen_a_trainable) == 2
    assert not any(parameter.requires_grad for parameter in frozen_a.lora_A.parameters())


def test_batched_instruction_evaluation_matches_single_example_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    dataset = [
        {"instruction": "Add", "input": str(index), "output": str(index + 1)}
        for index in range(7)
    ]
    monkeypatch.setattr(benchmark, "_load_dataset", lambda _name: dataset)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    model = _CausalModel()
    common = {
        "dataset_name": "unused",
        "validation_examples": 4,
        "seq_length": 32,
        "gpu_lock_path": str(tmp_path / "gpu.lock"),
    }
    unbatched = benchmark.evaluate_instruction_model(
        model, evaluation_batch_size=1, **common
    )
    batched = benchmark.evaluate_instruction_model(
        model, evaluation_batch_size=3, **common
    )
    assert batched["heldout_loss"] == pytest.approx(unbatched["heldout_loss"])
    assert batched["heldout_token_accuracy"] == pytest.approx(
        unbatched["heldout_token_accuracy"]
    )
    assert batched["heldout_window_losses"] == pytest.approx(
        unbatched["heldout_window_losses"]
    )
