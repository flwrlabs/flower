"""Tests for continued-pretraining benchmark helpers."""

import torch

from flowertune_llm.pretraining import (
    _evaluate_windows,
    _training_windows,
    merge_relora_state_dict,
)


def test_training_windows_are_repeatable_and_advance_by_round() -> None:
    tokens = list(range(100))
    first = _training_windows(tokens, server_round=1, steps=2, seq_length=8)
    second = _training_windows(tokens, server_round=2, steps=2, seq_length=8)
    assert first == [list(range(8)), list(range(8, 16))]
    assert second == [list(range(16, 24)), list(range(24, 32))]
    assert first == _training_windows(
        tokens, server_round=1, steps=2, seq_length=8
    )
    shifted = _training_windows(
        tokens, server_round=1, steps=2, seq_length=8, initial_offset=24
    )
    assert shifted == [list(range(24, 32)), list(range(32, 40))]


def test_merge_relora_state_dict_updates_base_and_restarts_adapter() -> None:
    prefix = "base_model.model.layer.proj"
    state = {
        f"{prefix}.weight": torch.zeros((2, 2)),
        f"{prefix}.lora_A.default.weight": torch.ones((1, 2)),
        f"{prefix}.lora_B.default.weight": torch.full((2, 1), 2.0),
    }
    merged = merge_relora_state_dict(state, rank=1, alpha=2, seed=7)
    assert merged == 1
    assert torch.equal(
        state[f"{prefix}.weight"], torch.full((2, 2), 4.0)
    )
    assert torch.count_nonzero(state[f"{prefix}.lora_A.default.weight"]) > 0
    assert torch.count_nonzero(state[f"{prefix}.lora_B.default.weight"]) == 0


def test_merge_relora_supports_current_peft_base_layer_names() -> None:
    prefix = "base_model.model.layer.proj"
    state = {
        f"{prefix}.base_layer.weight": torch.zeros((2, 2)),
        f"{prefix}.lora_A.default.weight": torch.ones((1, 2)),
        f"{prefix}.lora_B.default.weight": torch.full((2, 1), 2.0),
    }
    assert merge_relora_state_dict(state, rank=1, alpha=2, seed=7) == 1
    assert torch.equal(
        state[f"{prefix}.base_layer.weight"], torch.full((2, 2), 4.0)
    )


class _EvalModel(torch.nn.Module):
    def forward(self, *, input_ids, labels, use_cache):  # type: ignore[no-untyped-def]
        del labels, use_cache
        vocab = 8
        logits = torch.nn.functional.one_hot(
            input_ids % vocab, num_classes=vocab
        ).float()
        return type("Output", (), {"loss": input_ids.float().mean(), "logits": logits})


def test_evaluate_windows_retains_paired_window_losses() -> None:
    values = _evaluate_windows(
        _EvalModel(), [[1, 2, 3], [3, 4, 5]], torch.device("cpu")
    )
    batched = _evaluate_windows(
        _EvalModel(),
        [[1, 2, 3], [3, 4, 5]],
        torch.device("cpu"),
        batch_size=2,
    )
    assert len(values[-1]) == 2
    assert values[0] == sum(values[-1]) / len(values[-1])
    assert batched[-1] == values[-1]
