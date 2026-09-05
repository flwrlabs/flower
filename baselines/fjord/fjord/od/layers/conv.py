"""Convolutional layer using Ordered Dropout."""

from typing import Optional, Tuple, Union

import numpy as np
from torch import Tensor, nn
from torch.nn import Module

from .utils import check_layer

__all__ = ["ODConv1d", "ODConv2d", "ODConv3d"]


def od_conv_forward(
    layer: Module, x: Tensor, p: Optional[Union[Tuple[Module, float], float]] = None
) -> Tensor:
    """Ordered dropout forward pass for convolution networks.

    Args:
    :param layer: The layer being forwarded.
    :param x: Input tensor.
    :param p: Tuple of layer and p or p.
    :return: Output of forward pass.
    """
    p = check_layer(layer, p)
    if not layer.is_od and p is not None:
        raise ValueError("p must be None if is_od is False")
    in_dim = x.size(1)  # second dimension is input dimension
    layer.last_input_dim = in_dim
    if not p:  # i.e., don't apply OD
        out_dim = layer.width
    else:
        out_dim = int(np.ceil(layer.width * p))
    layer.last_output_dim = out_dim
    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(  # pylint: disable=protected-access
        x, weights_red, bias_red
    )


def get_slice(layer: Module, in_dim: int, out_dim: int) -> Tuple[Tensor, Tensor]:
    """Get slice of weights and bias.

    Args:
    :param layer: The layer.
    :param in_dim: The input dimension.
    :param out_dim: The output dimension.
    :return: The slice of weights and bias.
    """
    weight_slice = layer.weight[:out_dim, :in_dim]
    bias_slice = layer.bias[:out_dim] if layer.bias is not None else None
    return weight_slice, bias_slice


class ODConv1d(nn.Conv1d):
    """Ordered Dropout Conv1d."""

    def __init__(self, *args, is_od: bool = True, **kwargs) -> None:
        self.is_od = is_od
        super().__init__(*args, **kwargs)
        self.width = self.out_channels
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
        return od_conv_forward(self, input, p)

    def get_slice(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Get slice of weights and bias."""
        return get_slice(self, *args, **kwargs)


class ODConv2d(nn.Conv2d):
    """Ordered Dropout Conv2d."""

    def __init__(self, *args, is_od: bool = True, **kwargs) -> None:
        self.is_od = is_od
        super().__init__(*args, **kwargs)
        self.width = self.out_channels
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
        return od_conv_forward(self, input, p)

    def get_slice(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Get slice of weights and bias."""
        return get_slice(self, *args, **kwargs)


class ODConv3d(nn.Conv3d):
    """Ordered Dropout Conv3d."""

    def __init__(self, *args, is_od: bool = True, **kwargs) -> None:
        self.is_od = is_od
        super().__init__(*args, **kwargs)
        self.width = self.out_channels
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
        return od_conv_forward(self, input, p)

    def get_slice(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Get slice of weights and bias."""
        return get_slice(self, *args, **kwargs)
