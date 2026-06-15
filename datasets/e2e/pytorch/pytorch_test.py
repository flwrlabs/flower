import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets.utils.logging import disable_progress_bar
from parameterized import parameterized_class, parameterized
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from flwr_datasets import FederatedDataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Using parameterized testing, two different sets of parameters are specified:
# 1. CIFAR10 dataset with the simple ToTensor transform.
# 2. CIFAR10 dataset with a composed transform that first converts an image to a tensor
#    and then normalizes it.
@parameterized_class(
    [
        {"dataset_name": "cifar10", "test_split": "test", "transforms": ToTensor()},
        {"dataset_name": "cifar10", "test_split": "test", "transforms": Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )},
    ]
)
class FdsToPyTorch(unittest.TestCase):
    """Test the conversion from FDS to PyTorch Dataset and Dataloader."""

    dataset_name = ""
    test_split = ""
    transforms = None
    trainloader = None
    expected_img_shape_after_transform = [3, 32, 32]

    @classmethod
    def setUpClass(cls):
        """Disable progress bar to keep the log clean.
        """
        disable_progress_bar()

    def _create_trainloader(self, batch_size: int) -> DataLoader:
        """Create a trainloader from the federated dataset."""
        partition_id = 0
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 100})
        partition = fds.load_partition(partition_id, "train")
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        def apply_transforms(batch):
            images = batch["img"]
            if isinstance(images, list):
                batch["img"] = [self.transforms(img) for img in images]
            else:
                batch["img"] = self.transforms(images)
            return batch

        # Avoid Dataset.with_format("torch"), which imports torchvision.io.VideoReader.
        trainloader = DataLoader(
            partition_train_test["train"].with_transform(apply_transforms),
            batch_size=batch_size,
            shuffle=True,
        )
        return trainloader

    def test_create_partition_dataloader_with_transforms_shape(self) -> None:
        """Test if the DataLoader returns batches with the expected shape."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        images = batch["img"]
        self.assertEqual(tuple(images.shape),
                         (batch_size, *self.expected_img_shape_after_transform))

    def test_create_partition_dataloader_with_transforms_batch_type(self) -> None:
        """Test if the DataLoader returns batches of type dictionary."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        self.assertIsInstance(batch, dict)

    def test_create_partition_dataloader_with_transforms_data_type(self) -> None:
        """Test to verify if the data in the DataLoader batches are of type Tensor."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        images = batch["img"]
        self.assertIsInstance(images, Tensor)

    @parameterized.expand([
        ("not_nan", torch.isnan),
        ("not_inf", torch.isinf),
    ])
    def test_train_model_loss_value(self, name, condition_func):
        """Test if the model trains and if the loss is a correct number."""
        trainloader = self._create_trainloader(16)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create the model, criterion, and optimizer
        net = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Training loop for one epoch
        net.train()
        loss = None
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['img'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        self.assertFalse(condition_func(loss).item())


if __name__ == '__main__':
    unittest.main()
