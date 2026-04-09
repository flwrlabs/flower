"""..."""

import random
from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from flwr.client import NumPyClient
from hydra.utils import instantiate


class FedAvgClient(NumPyClient):
    """..."""

    def __init__(self, net, trainloader, device, num_epochs, learning_rate, momentum, weight_decay):
        """..."""
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config):
        """..."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """..."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        self.lr_decay_accumulated = config["lr_decay_accumulated"]
        self.local_update()
        return self.get_parameters({}), len(self.trainloader), {}

    def local_update(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate * self.lr_decay_accumulated, momentum=self.momentum, weight_decay=self.weight_decay
        )
        self.net.train()
        for _ in range(self.num_epochs):
            self.training_loop(criterion, optimizer)

    def training_loop(self, criterion, optimizer):
        """..."""
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.net(images), labels)
            loss.backward()
            optimizer.step()


class NaiveMixClient(NumPyClient):
    """..."""

    def __init__(self, net, trainloader, device, num_epochs, learning_rate, momentum, weight_decay):
        """..."""
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config):
        """..."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """..."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        self.mashed_data = config["mashed_data"]
        self.mixup_ratio = config["mixup_ratio"]
        self.lr_decay_accumulated = config["lr_decay_accumulated"]
        self.local_update()
        return self.get_parameters({}), len(self.trainloader), {}

    def local_update(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate * self.lr_decay_accumulated, momentum=self.momentum, weight_decay=self.weight_decay
        )
        self.net.train()
        for _ in range(self.num_epochs):
            self.training_loop(criterion, optimizer)

    def training_loop(self, criterion, optimizer):
        """..."""
        for images, labels in self.trainloader:
            mashed_image, mashed_label = random.choice(self.mashed_data)
            images, labels = images.to(self.device), labels.to(self.device)

            mashed_image, mashed_label = mashed_image[None, :].to(
                self.device
            ), mashed_label[None, :].to(self.device)

            num_classes = len(mashed_label[0])
            mashed_labels = mashed_label.expand_as(F.one_hot(labels, num_classes))

            optimizer.zero_grad()

            mixup_outputs = self.net(
                (1 - self.mixup_ratio) * images + self.mixup_ratio * mashed_image
            )
            loss = (1 - self.mixup_ratio) * criterion(
                mixup_outputs, labels
            ) + self.mixup_ratio * criterion(mixup_outputs, mashed_labels)

            loss.backward()
            optimizer.step()


class FedMixClient(NumPyClient):
    """..."""

    def __init__(self, net, trainloader, device, num_epochs, learning_rate, momentum, weight_decay):
        """..."""
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config):
        """..."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """..."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        self.mashed_data = config["mashed_data"]
        self.mixup_ratio = config["mixup_ratio"]
        self.lr_decay_accumulated = config["lr_decay_accumulated"]
        self.local_update()
        return self.get_parameters({}), len(self.trainloader), {}

    def local_update(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate * self.lr_decay_accumulated, momentum=self.momentum, weight_decay=self.weight_decay
        )
        self.net.train()
        for _ in range(self.num_epochs):
            self.training_loop(criterion, optimizer)

    def training_loop(self, criterion, optimizer):
        """..."""
        for images, labels in self.trainloader:
            batch_size = labels.size()[0]

            mashed_image, mashed_label = random.choice(self.mashed_data)
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad_()

            mashed_image = mashed_image.to(self.device)
            mashed_label = mashed_label.to(self.device)

            mashed_image = mashed_image.repeat(batch_size, 1, 1, 1)

            num_classes = len(mashed_label)
            mashed_labels = mashed_label.expand_as(F.one_hot(labels, num_classes))

            optimizer.zero_grad()

            scaled_images = (1 - self.mixup_ratio) * images

            intermediate_output = self.net(scaled_images)

            jacobian = autograd.grad(
                outputs=intermediate_output[:, labels].sum(), inputs=images, retain_graph=True)[0].view(batch_size, 1, -1)

            l1 = (1 - self.mixup_ratio) * criterion(intermediate_output, labels)
            l2 = (1 - self.mixup_ratio) * self.mixup_ratio * torch.mean(torch.bmm(jacobian, mashed_image.view(batch_size, -1, 1)))

            for i in range(num_classes):
                if mashed_labels[0, i] > 0:
                    mashed_labels_ = i * torch.ones_like(labels).to(self.device)
                    l1 = l1 + mashed_labels[0, i] * self.mixup_ratio * criterion(
                        intermediate_output, mashed_labels_)

            loss = l1 + l2
            loss.backward()
            optimizer.step()

    def evaluate(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.trainloader), {"accuracy": float(accuracy)}

    def test(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if len(self.trainloader.dataset) == 0:
            raise ValueError("Testloader can't be 0, exiting...")
        loss /= len(self.trainloader.dataset)
        accuracy = correct / total
        return loss, accuracy


def gen_client_fn(config, trainloaders, model):
    """..."""
    def client_fn(cid):
        """..."""
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        trainloader = trainloaders[int(cid)]

        client = instantiate(
            config, net=net, trainloader=trainloader, device=device)

        return client

    return client_fn
