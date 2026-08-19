"""Models, training and eval functions."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils
import torchvision.transforms as transforms
from flwr.common.logger import log
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import CLIPModel


def resnet50() -> torch.nn.modules:
    """Return ResNet-50 model as feature extractor."""
    resnet50_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Remove last layer and flatten outputs
    resnet50_model = torch.nn.Sequential(
        *(list(resnet50_model.children())[:-1]), torch.nn.Flatten()
    )

    # Set the hidden_dimension
    resnet50_model.hidden_dimension = 2048

    return resnet50_model


def clip_vit(name: str) -> torch.nn.modules:
    """Return CLIP-ViT as feature extractor.

    Parameters
    ----------
    name : str
        Name of the CLIP model on transformer library,
        e.g. `openai/clip-vit-base-patch32`.
    """

    class ClipVit(nn.Module):
        """Wrap outputs to return only pooled outputs."""

        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model
            self.hidden_dimension = vision_model.config.hidden_size

        def forward(self, x):
            """Return pooled output (CLS token)."""
            output = self.vision_model(x)
            return output[1]

    vision_model = CLIPModel.from_pretrained(name).vision_model

    return ClipVit(vision_model)


def transform(mean: List, std: List) -> transforms.Compose:
    """Return `transforms.Compose` function for normalizing images.

    Parameters
    ----------
    mean : List
        Sequence of means for each channel
    std : List
        Sequence of standard deviations for each channel.

    Returns
    -------
    transforms.Compose
        Transform function for normalizing images
    """
    transform_comp = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_comp


def extract_features(
    dataloader: DataLoader, feature_extractor: torch.nn.Module, device: torch.device
) -> Tuple[NDArray, NDArray]:
    """Extract features and labels from images using feature extractor.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader containing {'img': img, 'label': label}
        dicts to be extracted.
    feature_extractor : torch.nn.Module
        Model for extracting features.
    device : torch.device
        Device for loading `feature_extractor`.

    Returns
    -------
    features : NDArray
        2D array containing features extracted from `feature_extractor`.
    labels : NDArray
        2D array containing labels of `features`.
    """
    feature_extractor.to(device)

    features, labels = [], []
    for sample in dataloader:
        batch_samples = sample["img"].to(device)
        batch_label = sample["label"].to(device)
        with torch.no_grad():
            feature = feature_extractor(batch_samples)
        features.append(feature.cpu().detach().numpy())
        labels.append(batch_label.cpu().detach().numpy())

    # reshape feauturs and labels into a single numpy array
    features_np = np.concatenate(features, axis=0).astype("float64")
    labels_np = np.concatenate(labels)

    return features_np, labels_np


def test(
    classifier_head: torch.nn.Linear,
    dataloader: DataLoader,
    feature_extractor: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the `classifier_head` on the dataset.

    Parameters
    ----------
    classifier_head : torch.nn.Linear
        Classifier head model.
    dataloader : DataLoader
        Dataset used for evaluating `classifier_head` containing
        {'img': img, 'label': label} dicts.
    feature_extractor : torch.nn.Module
        Model used for extracting features from the `dataloader`.
    device : torch.device
        Device for loading `feature_extractor`.

    Returns
    -------
    loss : float
        CrossEntropy Loss of `classifier_head` on the dataset.
    accuracy : float
        Accuracy of `classifier_head` on the dataset.
    """
    classifier_head.eval()
    feature_extractor.eval()
    classifier_head.to(device)
    feature_extractor.to(device)

    correct, total, loss = 0, 0, 0
    for sample in dataloader:
        samples = sample["img"].to(device)
        labels = sample["label"].to(device)
        with torch.no_grad():
            feature = feature_extractor(samples)
            output = classifier_head(feature)
        pred = torch.max(output, 1)[1].data.squeeze()
        correct += (pred == labels).sum().item()
        total += samples.shape[0]
        running_loss = nn.CrossEntropyLoss()(output, labels)
        loss += running_loss.cpu().item()

    return loss, correct / total


# pylint: disable=too-many-locals, too-many-arguments
def train(
    classifier_head: torch.nn.Linear,
    dataloader: DataLoader,
    opt: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    feature_extractor: Optional[torch.nn.Module] = None,
    verbose: Optional[bool] = False,
) -> None:
    """Trains the `classifier_head`.

    Parameters
    ----------
    classifier_head : torch.nn.Linear
        Classifier head model.
    dataloader : DataLoader
        Dataset used for evaluating `classifier_head`
        containing {'img': img, 'label': label} dicts.
    opt : torch.optim.Optimizer
        Optimizer for the `classifier_head`.
    num_epochs: int
        Number of epochs to train the `classifier_head`.
    device : torch.device
        Device for loading `feature_extractor`.
    feature_extractor : torch.nn.Module, Optional
        Model used for extracting features from the `dataloader`, optional.
    `verbose` : bool, Optional
        Whether or not log the accuracy during the training. Defaults to False.
    """
    classifier_head.to(device)
    if feature_extractor:
        feature_extractor.eval()
        feature_extractor.to(device)

    for epoch in range(num_epochs):
        correct, total, loss = 0, 0, 0
        for _, batch in enumerate(dataloader):
            classifier_head.zero_grad()
            samples = batch["img"].to(device)
            labels = batch["label"].to(device)
            if feature_extractor:
                with torch.no_grad():
                    samples = feature_extractor(samples)
            output = classifier_head(samples)
            pred = torch.max(output, 1)[1].data.squeeze()
            correct += (pred == labels).sum().item()
            total += samples.shape[0]
            running_loss = nn.CrossEntropyLoss()(output, labels)
            loss += running_loss
            running_loss.backward()
            opt.step()
        if verbose:
            log(logging.INFO, "Epoch: %s --- Accuracy: %s", epoch + 1, correct / total)
