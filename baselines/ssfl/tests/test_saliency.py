"""Unit tests for SSFL saliency scores."""

from __future__ import annotations

import torch
import torch.nn as nn

from ssfl.model import SparseModel, prunable_parameter_names
from ssfl.saliency import calculate_ssfl_scores


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(4 * 8 * 8, 3)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 8)
        return self.fc(x.view(x.size(0), -1))


def test_ssfl_scores_keys_and_shape():
    torch.manual_seed(0)
    model = SparseModel(TinyNet())
    x = torch.randn(2, 3, 16, 16)
    y = torch.tensor([0, 1])
    scores = calculate_ssfl_scores(model, (x, y), device=torch.device("cpu"))

    prunable = prunable_parameter_names(model)
    expected_stems = {name.replace(".weight", "") for name in prunable}
    assert set(scores.keys()) == expected_stems
    for name, tensor in scores.items():
        weight = model.state_dict()[f"{name}.weight"]
        assert tensor.shape == weight.shape
        assert torch.all(tensor >= 0)


def test_ssfl_scores_match_manual_formula():
    torch.manual_seed(1)
    model = SparseModel(TinyNet())
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 3, (4,))

    model.zero_grad()
    model.eval()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    manual = {}
    for name, param in model.named_parameters():
        if param.grad is not None and name.endswith(".weight"):
            stem = name.replace(".weight", "")
            manual[stem] = torch.abs(param.grad.detach().cpu() * param.detach().cpu())

    scores = calculate_ssfl_scores(model, (x, y), device=torch.device("cpu"))
    for stem, expected in manual.items():
        assert torch.allclose(scores[stem], expected, atol=1e-6)
