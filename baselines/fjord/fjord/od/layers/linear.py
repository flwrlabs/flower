"""Liner layer using Ordered Dropout."""

from typing import Optional, Tuple, Union

import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from .utils import check_layer

__all__ = ["ODLinear"]


class ODLinear(nn.Linear):
    """Ordered Dropout Linear."""

    def __init__(self, *args, is_od: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_od = is_od
        self.width = self.out_features
        self.last_input_dim = None
        self.last_output_dim = None

    def forward(  # pylint: disable=arguments-differ
        self,
        input: Tensor,  # pylint: disable=redefined-builtin
        p: Optional[Union[Tuple[Module, float], float]] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
        :param input: Input tensor.
        :param p: Tuple of layer and p or p.
        :return: Output of forward pass.
        """
        if not self.is_od and p is not None:
            raise ValueError("p must be None if is_od is False")
        p = check_layer(self, p)
        in_dim = input.size(1)  # second dimension is input dimension
        self.last_input_dim = in_dim
        if not p:  # i.e., don't apply OD
            out_dim = self.width
        else:
            out_dim = int(np.ceil(self.width * p))
        self.last_output_dim = out_dim
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(input, weights_red, bias_red)  # pylint: disable=not-callable

    def get_slice(self, in_dim: int, out_dim: int) -> Tuple[Tensor, Tensor]:
        """Get slice of weights and bias.

        Args:
        :param layer: The layer.
        :param in_dim: The input dimension.
        :param out_dim: The output dimension.
        :return: The slice of weights and bias.
        """
        weight_slice = self.weight[:out_dim, :in_dim]
        bias_slice = self.bias[:out_dim] if self.bias is not None else None
        return weight_slice, bias_slice
