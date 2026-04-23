"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import Dataset
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from fedntd.models import Net, apply_transforms, test


def get_evaluate_fn(
    centralized_testset: Dataset,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    centralized_testset : Dataset
        The dataset to test the model with.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate_fn(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = Net(num_classes=10)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        testset = centralized_testset.with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn
