"""Utility functions for fjord."""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.nn import Module

from .logger import Logger


def get_parameters(net: Module) -> List[np.ndarray]:
    """Get statedict parameters as a list of numpy arrays.

    :param net: PyTorch model
    :return: List of numpy arrays
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: Module, parameters: List[np.ndarray]) -> None:
    """Load parameters into PyTorch model.

    :param net: PyTorch model
    :param parameters: List of numpy arrays
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


def save_model(
    model: torch.nn.Module,
    model_path: str,
    is_best: bool = False,
    cid: Optional[int] = None,
) -> None:
    """Checkpoint model.

    :param model: model to be saved
    :param model_path: path to save the model
    :param is_best: whether this is the best model
    :param cid: client id
    """
    suffix = "best" if is_best else "last"
    if cid:
        suffix += f"_{cid}"
    filename = os.path.join(model_path, f"model_{suffix}.checkpoint")
    Logger.get().info(f"Persisting model in {filename}")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), filename)
