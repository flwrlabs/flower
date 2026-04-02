"""vitexample: A Flower / PyTorch app with Vision Transformers."""

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def get_model(num_classes: int):
    """Return a pretrained ViT with all layers frozen except output head."""

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # We're going to federated the finetuning of this model
    # using (by default) the Oxford Flowers-102 dataset. One easy way
    # to achieve this is by re-initializing the output block of the
    # ViT so it outputs 102 clases instead of the default 1k
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, num_classes)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)

    return model


def trainer(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


fds = None


def get_dataset_partition(num_partitions: int, partition_id: int, dataset_name: str):
    """Get Oxford Flowers datasets and partition it."""
    global fds
    if fds is None:
        # Get dataset (by default Oxford Flowers-102) and create IID partitions
        partitioner = IidPartitioner(num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name, partitioners={"train": partitioner}
        )

    return fds.load_partition(partition_id)


def apply_eval_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def apply_train_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch
