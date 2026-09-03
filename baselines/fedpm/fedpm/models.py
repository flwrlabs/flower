"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedpm.resnet import resnet20

activation_dict = {"relu": F.relu, "sigmoid": F.sigmoid}


class Bern(torch.autograd.Function):
    """Custom Bernouli function that supports gradients. The original Pytorch
    implementation of Bernouli function, does not support gradients.

    First-Order gradient of bernouli function with prbabilty p, is p.

    Inputs: Tensor of arbitrary shapes with bounded values in [0,1] interval
    Outputs: Randomly generated Tensor of only {0,1}, given Inputs as distributions.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        pvals = ctx.saved_tensors
        return pvals[0] * grad_output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_model(params):
    if params.name == "LeNet":
        return LeNet5()
    elif params.name == "ResNet":
        return resnet20(params)
    elif params.name == "Conv8":
        if params.mode == "mask":
            return Mask8CNN()
    elif params.name == "Conv6":
        if params.mode == "mask":
            return Mask6CNN()
        elif params.mode == "dense":
            return Dense6CNN()
    elif params.name == "Conv4":
        if params.mode == "mask":
            return Mask4CNN()
        elif params.mode == "dense":
            return Dense4CNN()


class Dense4CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense4CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding="same", bias=False, device=device
        )
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding="same", bias=False, device=device
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding="same", bias=False, device=device
        )
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding="same", bias=False, device=device
        )

        self.dense1 = nn.Linear(6272, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense6CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense6CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv5 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv6 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding="same", device=device
        )

        self.dense1 = nn.Linear(4096, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense10CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense10CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv5 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv6 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv7 = nn.Conv2d(
            256, 512, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv8 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv9 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv10 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding="same", device=device
        )

        self.dense1 = nn.Linear(1024, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x, ths=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv9(x))
        x = F.max_pool2d(F.relu(self.conv10(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding="same"
        )
        self.act1 = nn.ReLU()
        self.avg1 = nn.MaxPool2d(kernel_size=2, stride=3)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding="same"
        )
        self.act2 = nn.ReLU()
        self.avg2 = nn.MaxPool2d(kernel_size=2, stride=3)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, padding="same"
        )
        self.act3 = nn.ReLU()
        self.avg3 = nn.MaxPool2d(kernel_size=2, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.avg1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.avg2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.avg3(x)
        x = x.view(-1, 120)
        return self.classifier(x)


class MaskedLinear(nn.Linear):
    """Implementation of masked linear layer, with training strategy in https://proceedi
    ngs.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf.
    """

    def __init__(
        self, in_features, out_features, init="ME_init", device=None, **kwargs
    ):
        super(MaskedLinear, self).__init__(
            in_features, out_features, device=device, **kwargs
        )
        self.device = device
        arr_weights = None
        self.device = device
        self.init = init
        self.c = np.e * np.sqrt(1 / in_features)
        # Different weight initialization distributions
        if init == "ME_init":
            arr_weights = np.random.choice(
                [-self.c, self.c], size=(out_features, in_features)
            )
        elif init == "ME_init_sym":
            arr_weights = np.random.choice(
                [-self.c, self.c], size=(out_features, in_features)
            )
            arr_weights = np.triu(arr_weights, k=1) + np.tril(arr_weights)
        elif init == "uniform":
            arr_weights = np.random.uniform(
                -self.c, self.c, size=(out_features, in_features)
            ) * np.sqrt(3)
        elif init == "k_normal":
            arr_weights = np.random.normal(0, self.c, size=(out_features, in_features))

        self.weight = nn.Parameter(
            torch.tensor(
                arr_weights, requires_grad=False, device=self.device, dtype=torch.float
            )
        )

        arr_bias = np.random.choice([-self.c, self.c], size=out_features)
        self.bias = nn.Parameter(
            torch.tensor(
                arr_bias, requires_grad=False, device=self.device, dtype=torch.float
            )
        )

        # Weights of Mask
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.mask = nn.Parameter(
            torch.randn_like(self.weight, requires_grad=True, device=self.device)
        )
        self.bias_mask = nn.Parameter(
            torch.randn_like(self.bias, requires_grad=True, device=self.device)
        )

    def forward(self, x, ths=None):
        if ths is None:
            # Generate probability of Bernoulli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        # Compute element-wise product with mask
        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m

        # Apply the effective weight on the input data
        lin = F.linear(
            x, effective_weight.to(self.device), effective_bias.to(self.device)
        )
        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return "Mask Layer: \n FC Weights: {}, {}, MASK: {}".format(
            self.weight.sum(), torch.abs(self.weight).sum(), self.mask.sum() / prod
        )


class MaskANN(nn.Module):
    def __init__(self, init="ME_init", activation="relu", device=None):
        super(MaskANN, self).__init__()
        self.activation = activation
        self.init = init
        self.ml1 = MaskedLinear(784, 600, init=init, device=device)
        self.ml2 = MaskedLinear(600, 300, init=init, device=device)
        self.ml3 = MaskedLinear(300, 10, init=init, device=device)

    def forward(self, x):
        x = self.ml1(x)
        x = activation_dict[self.activation](x)
        x = self.ml2(x)
        x = activation_dict[self.activation](x)
        x = self.ml3(x)

        return x

    def get_layers(self):
        return [self.ml1, self.ml2, self.ml3]


class BigMaskANN(nn.Module):
    def __init__(self, init="ME_init", activation="relu", device=None):
        super(BigMaskANN, self).__init__()
        self.activation = activation
        self.init = init
        self.ml1 = MaskedLinear(784, 800, init=init, device=device)
        self.ml2 = MaskedLinear(800, 200, init=init, device=device)
        self.ml3 = MaskedLinear(200, 47, init=init, device=device)

    def forward(self, x):
        x = self.ml1(x)
        x = activation_dict[self.activation](x)
        x = self.ml2(x)
        x = activation_dict[self.activation](x)
        x = self.ml3(x)

        return x

    def get_layers(self):
        return [self.ml1, self.ml2, self.ml3]


class MaskedConv2d(nn.Conv2d):
    """Implementation of masked convolutional layer, with training strategy in https://p
    roceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        init="ME_init",
        device=None,
        **kwargs,
    ):
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, device=device, **kwargs
        )
        self.device = device
        arr_weights = None
        self.init = init
        self.c = np.e * np.sqrt(1 / (kernel_size**2 * in_channels))

        if init == "ME_init":
            arr_weights = np.random.choice(
                [-self.c, self.c],
                size=(out_channels, in_channels, kernel_size, kernel_size),
            )
        elif init == "uniform":
            arr_weights = np.random.uniform(
                -self.c,
                self.c,
                size=(out_channels, in_channels, kernel_size, kernel_size),
            ) * np.sqrt(3)
        elif init == "k_normal":
            arr_weights = np.random.normal(
                0,
                self.c**2,
                size=(out_channels, in_channels, kernel_size, kernel_size),
            )

        self.weight = nn.Parameter(
            torch.tensor(
                arr_weights, requires_grad=False, device=self.device, dtype=torch.float
            )
        )

        arr_bias = np.random.choice([-self.c, self.c], size=out_channels)
        self.bias = nn.Parameter(
            torch.tensor(
                arr_bias, requires_grad=False, device=self.device, dtype=torch.float
            )
        )

        self.mask = nn.Parameter(
            torch.randn_like(self.weight, requires_grad=True, device=self.device)
        )
        self.bias_mask = nn.Parameter(
            torch.randn_like(self.bias, requires_grad=True, device=self.device)
        )
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x, ths=None):
        if ths is None:
            # Generate probability of bernouli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m
        # Apply the effective weight on the input data
        lin = self._conv_forward(
            x, effective_weight.to(self.device), effective_bias.to(self.device)
        )

        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return "Mask Layer: \n FC Weights: {}, {}, MASK: {}".format(
            self.weight.sum(), torch.abs(self.weight).sum(), self.mask.sum() / prod
        )


class Mask4CNN(nn.Module):
    """4Conv model studied in https://proceedings.neurips.cc/paper/2019/file/1113d7a76ff
    ceca1bb350bfe145467c6-Paper.pdf for cifar10.
    """

    def __init__(self, init="ME_init", activation="relu", device=None):
        super(Mask4CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(
            1, 64, kernel_size=3, stride=1, padding="same", init=init, device=device
        )
        self.conv2 = MaskedConv2d(
            64, 64, kernel_size=3, stride=1, padding="same", init=init, device=device
        )
        self.conv3 = MaskedConv2d(
            64, 128, kernel_size=3, stride=1, padding="same", init=init, device=device
        )
        self.conv4 = MaskedConv2d(
            128, 128, kernel_size=3, stride=1, padding="same", init=init, device=device
        )

        self.dense1 = MaskedLinear(6272, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 10, init=init, device=device)

    def forward(self, x, ths=None):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv2(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv4(x, ths)),
            kernel_size=2,
            stride=2,
        )

        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(f"conv4_model_{self.init}"))


class Mask6CNN(nn.Module):
    """6Conv model studied in https://proceedings.neurips.cc/paper/2019/file/1113d7a76ff
    ceca1bb350bfe145467c6-Paper.pdf for cifar10.
    """

    def __init__(self, init="ME_init", activation="relu", device=None):
        super(Mask6CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(
            3, 64, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv2 = MaskedConv2d(
            64, 64, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv3 = MaskedConv2d(
            64, 128, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv4 = MaskedConv2d(
            128, 128, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv5 = MaskedConv2d(
            128, 256, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv6 = MaskedConv2d(
            256, 256, 3, init=init, device=device, stride=1, padding="same"
        )

        self.dense1 = MaskedLinear(4096, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 10, init=init, device=device)

    def forward(self, x, ths=None):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv2(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv4(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv5(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv6(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(f"conv6_model_{self.init}"))

    def get_layers(self):
        return [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.dense1,
            self.dense2,
            self.dense3,
        ]


class Mask8CNN(nn.Module):
    """8Conv model studied in https://arxiv.org/pdf/1911.13299.pdf, here for
    cifar100.
    """

    def __init__(self, init="ME_init", activation="relu", device=None):
        super(Mask8CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(
            3, 64, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv2 = MaskedConv2d(
            64, 64, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv3 = MaskedConv2d(
            64, 128, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv4 = MaskedConv2d(
            128, 128, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv5 = MaskedConv2d(
            128, 256, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv6 = MaskedConv2d(
            256, 256, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv7 = MaskedConv2d(
            256, 512, 3, init=init, device=device, stride=1, padding="same"
        )
        self.conv8 = MaskedConv2d(
            512, 512, 3, init=init, device=device, stride=1, padding="same"
        )

        self.dense1 = MaskedLinear(2048, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 100, init=init, device=device)

    def forward(self, x, ths=None):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv2(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv4(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv5(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv6(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = activation_dict[self.activation](self.conv7(x, ths))
        x = F.max_pool2d(
            activation_dict[self.activation](self.conv8(x, ths)),
            kernel_size=2,
            stride=2,
        )
        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(f"conv8_model_{self.init}"))

    def get_layers(self):
        return [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.dense1,
            self.dense2,
            self.dense3,
        ]


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
