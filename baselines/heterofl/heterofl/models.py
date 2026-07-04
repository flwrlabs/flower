"""Conv & resnet18 model architecture, training, testing functions.

Classes Conv, Block, Resnet18 are adopted from authors implementation.
"""

import copy
from typing import List, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import parameters_to_ndarrays
from torch import nn

from heterofl.utils import make_optimizer


class Conv(nn.Module):
    """Convolutional Neural Network architecture with sBN."""

    def __init__(
        self,
        model_config,
    ):
        super().__init__()
        self.model_config = model_config

        blocks = [
            nn.Conv2d(
                model_config["data_shape"][0], model_config["hidden_size"][0], 3, 1, 1
            ),
            self._get_scale(),
            self._get_norm(0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        for i in range(len(model_config["hidden_size"]) - 1):
            blocks.extend(
                [
                    nn.Conv2d(
                        model_config["hidden_size"][i],
                        model_config["hidden_size"][i + 1],
                        3,
                        1,
                        1,
                    ),
                    self._get_scale(),
                    self._get_norm(i + 1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(
                    model_config["hidden_size"][-1], model_config["classes_size"]
                ),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def _get_norm(self, j: int):
        """Return the relevant norm."""
        if self.model_config["norm"] == "bn":
            norm = nn.BatchNorm2d(
                self.model_config["hidden_size"][j],
                momentum=None,
                track_running_stats=self.model_config["track"],
            )
        elif self.model_config["norm"] == "in":
            norm = nn.GroupNorm(
                self.model_config["hidden_size"][j], self.model_config["hidden_size"][j]
            )
        elif self.model_config["norm"] == "ln":
            norm = nn.GroupNorm(1, self.model_config["hidden_size"][j])
        elif self.model_config["norm"] == "gn":
            norm = nn.GroupNorm(4, self.model_config["hidden_size"][j])
        elif self.model_config["norm"] == "none":
            norm = nn.Identity()
        else:
            raise ValueError("Not valid norm")

        return norm

    def _get_scale(self):
        """Return the relevant scaler."""
        if self.model_config["scale"]:
            scaler = _Scaler(self.model_config["rate"])
        else:
            scaler = nn.Identity()
        return scaler

    def forward(self, input_dict):
        """Forward pass of the Conv.

        Parameters
        ----------
        input_dict : Dict
            Conatins input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        # output = {"loss": torch.tensor(0, device=self.device, dtype=torch.float32)}
        output = {}
        out = self.blocks(input_dict["img"])
        if "label_split" in input_dict and self.model_config["mask"]:
            label_mask = torch.zeros(
                self.model_config["classes_size"], device=out.device
            )
            label_mask[input_dict["label_split"]] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output["score"] = out
        output["loss"] = F.cross_entropy(out, input_dict["label"], reduction="mean")
        return output


def conv(
    model_rate,
    model_config,
    device="cpu",
):
    """Create the Conv model."""
    model_config["hidden_size"] = [
        int(np.ceil(model_rate * x)) for x in model_config["hidden_layers"]
    ]
    scaler_rate = model_rate / model_config["global_model_rate"]
    model_config["rate"] = scaler_rate
    model = Conv(model_config)
    model.apply(_init_param)
    return model.to(device)


class Block(nn.Module):
    """Block."""

    expansion = 1

    def __init__(self, in_planes, planes, stride, model_config):
        super().__init__()
        if model_config["norm"] == "bn":
            n_1 = nn.BatchNorm2d(
                in_planes, momentum=None, track_running_stats=model_config["track"]
            )
            n_2 = nn.BatchNorm2d(
                planes, momentum=None, track_running_stats=model_config["track"]
            )
        elif model_config["norm"] == "in":
            n_1 = nn.GroupNorm(in_planes, in_planes)
            n_2 = nn.GroupNorm(planes, planes)
        elif model_config["norm"] == "ln":
            n_1 = nn.GroupNorm(1, in_planes)
            n_2 = nn.GroupNorm(1, planes)
        elif model_config["norm"] == "gn":
            n_1 = nn.GroupNorm(4, in_planes)
            n_2 = nn.GroupNorm(4, planes)
        elif model_config["norm"] == "none":
            n_1 = nn.Identity()
            n_2 = nn.Identity()
        else:
            raise ValueError("Not valid norm")
        self.n_1 = n_1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.n_2 = n_2
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        if model_config["scale"]:
            self.scaler = _Scaler(model_config["rate"])
        else:
            self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        """Forward pass of the Block.

        Parameters
        ----------
        x : Dict
            Dict that contains Input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        out = F.relu(self.n_1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n_2(self.scaler(out))))
        out += shortcut
        return out


# pylint: disable=too-many-instance-attributes
class ResNet(nn.Module):
    """Implementation of a Residual Neural Network (ResNet) model with sBN."""

    def __init__(
        self,
        model_config,
        block,
        num_blocks,
    ):
        self.model_config = model_config
        super().__init__()
        self.in_planes = model_config["hidden_size"][0]
        self.conv1 = nn.Conv2d(
            model_config["data_shape"][0],
            model_config["hidden_size"][0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.layer1 = self._make_layer(
            block,
            model_config["hidden_size"][0],
            num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block,
            model_config["hidden_size"][1],
            num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            model_config["hidden_size"][2],
            num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            model_config["hidden_size"][3],
            num_blocks[3],
            stride=2,
        )

        # self.layers = [layer1, layer2, layer3, layer4]

        if model_config["norm"] == "bn":
            n_4 = nn.BatchNorm2d(
                model_config["hidden_size"][3] * block.expansion,
                momentum=None,
                track_running_stats=model_config["track"],
            )
        elif model_config["norm"] == "in":
            n_4 = nn.GroupNorm(
                model_config["hidden_size"][3] * block.expansion,
                model_config["hidden_size"][3] * block.expansion,
            )
        elif model_config["norm"] == "ln":
            n_4 = nn.GroupNorm(1, model_config["hidden_size"][3] * block.expansion)
        elif model_config["norm"] == "gn":
            n_4 = nn.GroupNorm(4, model_config["hidden_size"][3] * block.expansion)
        elif model_config["norm"] == "none":
            n_4 = nn.Identity()
        else:
            raise ValueError("Not valid norm")
        self.n_4 = n_4
        if model_config["scale"]:
            self.scaler = _Scaler(model_config["rate"])
        else:
            self.scaler = nn.Identity()
        self.linear = nn.Linear(
            model_config["hidden_size"][3] * block.expansion,
            model_config["classes_size"],
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, self.model_config.copy()))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        """Forward pass of the ResNet.

        Parameters
        ----------
        input_dict : Dict
            Dict that contains Input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        output = {}
        x = input_dict["img"]
        out = self.conv1(x)
        # for layer in self.layers:
        #     out = layer(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n_4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if "label_split" in input_dict and self.model_config["mask"]:
            label_mask = torch.zeros(
                self.model_config["classes_size"], device=out.device
            )
            label_mask[input_dict["label_split"]] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output["score"] = out
        output["loss"] = F.cross_entropy(output["score"], input_dict["label"])
        return output


def resnet18(
    model_rate,
    model_config,
    device="cpu",
):
    """Create the ResNet18 model."""
    model_config["hidden_size"] = [
        int(np.ceil(model_rate * x)) for x in model_config["hidden_layers"]
    ]
    scaler_rate = model_rate / model_config["global_model_rate"]
    model_config["rate"] = scaler_rate
    model = ResNet(model_config, block=Block, num_blocks=[1, 1, 1, 2])
    model.apply(_init_param)
    return model.to(device)


class MLP(nn.Module):
    """Multi Layer Perceptron."""

    def __init__(self):
        super().__init__()
        self.layer_input = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [
            ["layer_input.weight", "layer_input.bias"],
            ["layer_hidden1.weight", "layer_hidden1.bias"],
            ["layer_hidden2.weight", "layer_hidden2.bias"],
            ["layer_hidden3.weight", "layer_hidden3.bias"],
            ["layer_out.weight", "layer_out.bias"],
        ]

    def forward(self, input_dict):
        """Forward pass of the Conv.

        Parameters
        ----------
        input_dict : Dict
            Conatins input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        output = {}
        x = input_dict["img"]
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        out = self.softmax(x)
        output["score"] = out
        output["loss"] = F.cross_entropy(out, input_dict["label"], reduction="mean")
        return output


class CNNCifar(nn.Module):
    """Convolutional Neural Network architecture for cifar dataset."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)

        self.weight_keys = [
            ["fc1.weight", "fc1.bias"],
            ["fc2.weight", "fc2.bias"],
            ["fc3.weight", "fc3.bias"],
            ["conv2.weight", "conv2.bias"],
            ["conv1.weight", "conv1.bias"],
        ]

    def forward(self, input_dict):
        """Forward pass of the Conv.

        Parameters
        ----------
        input_dict : Dict
            Conatins input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        output = {}
        x = input_dict["img"]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        output["score"] = out
        output["loss"] = F.cross_entropy(out, input_dict["label"], reduction="mean")
        return output


def create_model(model_config, model_rate=None, track=False, device="cpu"):
    """Create the model based on the configuration given in hydra."""
    model = None
    model_config = model_config.copy()
    model_config["track"] = track

    if model_config["model"] == "MLP":
        model = MLP()
        model.to(device)
    elif model_config["model"] == "CNNCifar":
        model = CNNCifar()
        model.to(device)
    elif model_config["model"] == "conv":
        model = conv(model_rate=model_rate, model_config=model_config, device=device)
    elif model_config["model"] == "resnet18":
        model = resnet18(
            model_rate=model_rate, model_config=model_config, device=device
        )
    return model


def _init_param(m_param):
    if isinstance(m_param, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m_param.weight.data.fill_(1)
        m_param.bias.data.zero_()
    elif isinstance(m_param, nn.Linear):
        m_param.bias.data.zero_()
    return m_param


class _Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, inp):
        """Forward of Scalar nn.Module."""
        output = inp / self.rate if self.training else inp
        return output


def get_parameters(net) -> List[np.ndarray]:
    """Return the parameters of model as numpy.NDArrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """Set the model parameters with given parameters."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(model, train_loader, label_split, settings):
    """Train a model with given settings.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    train_loader : DataLoader
        The DataLoader containing the data to train the network on.
    label_split : torch.tensor
        Tensor containing the labels of the data.
    settings: Dict
        Dictionary conatining the information about eopchs, optimizer,
        lr, momentum, weight_decay, device to train on.
    """
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        settings["optimizer"],
        model.parameters(),
        learning_rate=settings["lr"],
        momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )

    model.train()
    for _ in range(settings["epochs"]):
        for images, labels in train_loader:
            input_dict = {}
            input_dict["img"] = images.to(settings["device"])
            input_dict["label"] = labels.to(settings["device"])
            input_dict["label_split"] = label_split.type(torch.int).to(
                settings["device"]
            )
            optimizer.zero_grad()
            output = model(input_dict)
            output["loss"].backward()
            if ("clip" not in settings) or (
                "clip" in settings and settings["clip"] is True
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()


def test(model, test_loader, label_split=None, device="cpu"):
    """Evaluate the network on the test set.

    Parameters
    ----------
    model : nn.Module
        The neural network to test.
    test_loader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        model.train(False)
        for images, labels in test_loader:
            input_dict = {}
            input_dict["img"] = images.to(device)
            input_dict["label"] = labels.to(device)
            if label_split is not None:
                input_dict["label_split"] = label_split.type(torch.int).to(device)
            output = model(input_dict)
            test_loss += output["loss"].item()
            correct += (
                (output["score"].argmax(1) == input_dict["label"])
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def param_model_rate_mapping(
    model_name, parameters, clients_model_rate, global_model_rate=1
):
    """Map the model rate to subset of global parameters(as list of indices).

    Parameters
    ----------
    model_name : str
        The name of the neural network of global model.
    parameters : Dict
        state_dict of the global model.
    client_model_rate : List[float]
        List of model rates of active clients.
    global_model_rate: float
        Model rate of the global model.

    Returns
    -------
    Dict
        model rate to parameters indices relative to global model mapping.
    """
    unique_client_model_rate = list(set(clients_model_rate))
    print(unique_client_model_rate)

    if "conv" in model_name:
        idx = _mr_to_param_idx_conv(
            parameters, unique_client_model_rate, global_model_rate
        )
    elif "resnet" in model_name:
        idx = _mr_to_param_idx_resnet18(
            parameters, unique_client_model_rate, global_model_rate
        )
    else:
        raise ValueError("Not valid model name")

    # add model rate as key to the params calculated
    param_idx_model_rate_mapping = OrderedDict()
    for i, _ in enumerate(unique_client_model_rate):
        param_idx_model_rate_mapping[unique_client_model_rate[i]] = idx[i]

    return param_idx_model_rate_mapping


def _mr_to_param_idx_conv(parameters, unique_client_model_rate, global_model_rate):
    idx_i = [None for _ in range(len(unique_client_model_rate))]
    idx = [OrderedDict() for _ in range(len(unique_client_model_rate))]
    output_weight_name = [k for k in parameters.keys() if "weight" in k][-1]
    output_bias_name = [k for k in parameters.keys() if "bias" in k][-1]
    for k, val in parameters.items():
        parameter_type = k.split(".")[-1]
        for index, _ in enumerate(unique_client_model_rate):
            if "weight" in parameter_type or "bias" in parameter_type:
                scaler_rate = unique_client_model_rate[index] / global_model_rate
                _get_key_k_idx_conv(
                    idx,
                    idx_i,
                    {
                        "index": index,
                        "parameter_type": parameter_type,
                        "k": k,
                        "val": val,
                    },
                    output_names={
                        "output_weight_name": output_weight_name,
                        "output_bias_name": output_bias_name,
                    },
                    scaler_rate=scaler_rate,
                )
            else:
                pass
    return idx


def _get_key_k_idx_conv(
    idx,
    idx_i,
    param_info,
    output_names,
    scaler_rate,
):
    if param_info["parameter_type"] == "weight":
        if param_info["val"].dim() > 1:
            input_size = param_info["val"].size(1)
            output_size = param_info["val"].size(0)
            if idx_i[param_info["index"]] is None:
                idx_i[param_info["index"]] = torch.arange(
                    input_size, device=param_info["val"].device
                )
            input_idx_i_m = idx_i[param_info["index"]]
            if param_info["k"] == output_names["output_weight_name"]:
                output_idx_i_m = torch.arange(
                    output_size, device=param_info["val"].device
                )
            else:
                local_output_size = int(np.ceil(output_size * (scaler_rate)))
                output_idx_i_m = torch.arange(
                    output_size, device=param_info["val"].device
                )[:local_output_size]
            idx[param_info["index"]][param_info["k"]] = output_idx_i_m, input_idx_i_m
            idx_i[param_info["index"]] = output_idx_i_m
        else:
            input_idx_i_m = idx_i[param_info["index"]]
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m
    else:
        if param_info["k"] == output_names["output_bias_name"]:
            input_idx_i_m = idx_i[param_info["index"]]
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m
        else:
            input_idx_i_m = idx_i[param_info["index"]]
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m


def _mr_to_param_idx_resnet18(parameters, unique_client_model_rate, global_model_rate):
    idx_i = [None for _ in range(len(unique_client_model_rate))]
    idx = [OrderedDict() for _ in range(len(unique_client_model_rate))]
    for k, val in parameters.items():
        parameter_type = k.split(".")[-1]
        for index, _ in enumerate(unique_client_model_rate):
            if "weight" in parameter_type or "bias" in parameter_type:
                scaler_rate = unique_client_model_rate[index] / global_model_rate
                _get_key_k_idx_resnet18(
                    idx,
                    idx_i,
                    {
                        "index": index,
                        "parameter_type": parameter_type,
                        "k": k,
                        "val": val,
                    },
                    scaler_rate=scaler_rate,
                )
            else:
                pass
    return idx


def _get_key_k_idx_resnet18(
    idx,
    idx_i,
    param_info,
    scaler_rate,
):
    if param_info["parameter_type"] == "weight":
        if param_info["val"].dim() > 1:
            input_size = param_info["val"].size(1)
            output_size = param_info["val"].size(0)
            if "conv1" in param_info["k"] or "conv2" in param_info["k"]:
                if idx_i[param_info["index"]] is None:
                    idx_i[param_info["index"]] = torch.arange(
                        input_size, device=param_info["val"].device
                    )
                input_idx_i_m = idx_i[param_info["index"]]
                local_output_size = int(np.ceil(output_size * (scaler_rate)))
                output_idx_i_m = torch.arange(
                    output_size, device=param_info["val"].device
                )[:local_output_size]
                idx_i[param_info["index"]] = output_idx_i_m
            elif "shortcut" in param_info["k"]:
                input_idx_i_m = idx[param_info["index"]][
                    param_info["k"].replace("shortcut", "conv1")
                ][1]
                output_idx_i_m = idx_i[param_info["index"]]
            elif "linear" in param_info["k"]:
                input_idx_i_m = idx_i[param_info["index"]]
                output_idx_i_m = torch.arange(
                    output_size, device=param_info["val"].device
                )
            else:
                raise ValueError("Not valid k")
            idx[param_info["index"]][param_info["k"]] = (output_idx_i_m, input_idx_i_m)
        else:
            input_idx_i_m = idx_i[param_info["index"]]
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m
    else:
        input_size = param_info["val"].size(0)
        if "linear" in param_info["k"]:
            input_idx_i_m = torch.arange(input_size, device=param_info["val"].device)
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m
        else:
            input_idx_i_m = idx_i[param_info["index"]]
            idx[param_info["index"]][param_info["k"]] = input_idx_i_m


def param_idx_to_local_params(global_parameters, client_param_idx):
    """Get the local parameters from the list of param indices.

    Parameters
    ----------
    global_parameters : Dict
        The state_dict of global model.
    client_param_idx : List
        Local parameters indices with respect to global model.

    Returns
    -------
    Dict
        state dict of local model.
    """
    local_parameters = OrderedDict()
    for k, val in global_parameters.items():
        parameter_type = k.split(".")[-1]
        if "weight" in parameter_type or "bias" in parameter_type:
            if "weight" in parameter_type:
                if val.dim() > 1:
                    local_parameters[k] = copy.deepcopy(
                        val[torch.meshgrid(client_param_idx[k])]
                    )
                else:
                    local_parameters[k] = copy.deepcopy(val[client_param_idx[k]])
            else:
                local_parameters[k] = copy.deepcopy(val[client_param_idx[k]])
        else:
            local_parameters[k] = copy.deepcopy(val)
    return local_parameters


def get_state_dict_from_param(model, parameters):
    """Get the state dict from model & parameters as np.NDarrays.

    Parameters
    ----------
    model : nn.Module
        The neural network.
    parameters : np.NDarray
        Parameters of the model as np.NDarrays.

    Returns
    -------
    Dict
        state dict of model.
    """
    # Load the parameters into the model
    for param_tensor, param_ndarray in zip(
        model.state_dict(), parameters_to_ndarrays(parameters)
    ):
        model.state_dict()[param_tensor].copy_(torch.from_numpy(param_ndarray))
    # Step 3: Obtain the state_dict of the model
    state_dict = model.state_dict()
    return state_dict
