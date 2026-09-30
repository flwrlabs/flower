import random
import numpy as np
import torch


class Weldon(torch.nn.Module):
    """Weldon module."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        n_top: int = 10,
        n_bottom: int = 10,
    ):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        super().__init__()

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = 1
        self.score_model = torch.nn.Linear(
            in_features, bias=True, out_features=out_features
        )
        torch.nn.init.xavier_uniform_(self.score_model.weight)
        self.score_model.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        scores = self.score_model(x)
        top, _ = scores.topk(k=self.n_top, dim=self.dim)
        bottom, _ = scores.topk(k=self.n_bottom, largest=False, dim=self.dim)
        extreme_scores = torch.cat([top, bottom], dim=self.dim)
        output = torch.mean(extreme_scores, 1, keepdim=False).reshape(-1)
        output = torch.sigmoid(output)
        return output
