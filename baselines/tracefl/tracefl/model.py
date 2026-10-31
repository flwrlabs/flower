"""tracefl-baseline: Multi-Model Support for TraceFL."""

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
)


class Net(nn.Module):
    """Simple CNN model (adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self, num_classes=10, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Do forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def initialize_model(name, cfg_dataset):
    """Initialize and configure the model based on its name and dataset configuration.

    Parameters
    ----------
    name : str
        The name or identifier of the model.
    cfg_dataset : object
        Configuration object for the dataset. Must contain attribute num_classes
        and channels.

    Returns
    -------
    dict
        A dictionary containing:
            - "model": The initialized model.
            - "num_classes": Number of classes in the dataset.

    Raises
    ------
    ValueError
        If the specified model name is not supported.
    """
    model_dict = {"model": None, "num_classes": cfg_dataset.num_classes}

    # Transformer models
    if name in [
        "squeezebert/squeezebert-uncased",
        "openai-community/openai-gpt",
        "Intel/dynamic_tinybert",
        "google-bert/bert-base-cased",
        "microsoft/MiniLM-L12-H384-uncased",
        "distilbert/distilbert-base-uncased",
    ]:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name,
            num_labels=cfg_dataset.num_classes,
        )

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        # Set pad_token to eos_token or add a new pad_token if eos_token is not
        # available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        model_dict["model"] = model.cpu()
        model_dict["tokenizer"] = tokenizer

    # ResNet models
    elif name.find("resnet") != -1:
        model = None
        if "resnet18" == name:
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif "resnet34" == name:
            model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif "resnet50" == name:
            model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif "resnet101" == name:
            model = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif "resnet152" == name:
            model = torchvision.models.resnet152(weights="IMAGENET1K_V1")

        if model is None:
            raise ValueError(f"ResNet model {name} not supported")

        # Handle grayscale input (1 channel) for medical datasets
        if cfg_dataset.channels == 1:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace final layer for our number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()

    # DenseNet models
    elif name == "densenet121":
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")

        # Handle grayscale input (1 channel) for medical datasets
        if cfg_dataset.channels == 1:
            logging.info(
                "Changing the first layer of densenet model to accept 1 channel"
            )
            model.features[0] = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        # Replace final layer for our number of classes
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()

    # Simple CNN (our original model)
    elif name == "cnn":
        model_dict["model"] = Net(
            num_classes=cfg_dataset.num_classes, channels=cfg_dataset.channels
        )

    else:
        raise ValueError(f"Model {name} not supported")

    return model_dict


def _prepare_images(pixel_values: Any, device: torch.device) -> torch.Tensor:
    """Convert pixel batches into a batched tensor on the device."""
    # Handle different input types - simplified since DefaultDataCollator handles
    # batching
    if torch.is_tensor(pixel_values):
        images = pixel_values.float()
    elif isinstance(pixel_values, np.ndarray):
        images = torch.from_numpy(pixel_values).float()
    else:
        images = torch.tensor(pixel_values).float()

    # Ensure correct format: (N, C, H, W)
    if images.ndim == 3:
        # Single image: (C, H, W) -> (1, C, H, W)
        images = images.unsqueeze(0)
    elif images.ndim == 4:
        # Batch of images: should already be (N, C, H, W)
        pass
    else:
        raise ValueError(
            "Expected image tensor with 3 or 4 dimensions, got shape "
            f"{tuple(images.shape)}"
        )

    return images.to(device)


def _prepare_labels(label_values: Any, device: torch.device) -> torch.Tensor:
    """Convert heterogeneous label batches into a 1D tensor on the device."""
    if torch.is_tensor(label_values):
        labels = label_values
    elif isinstance(label_values, np.ndarray):
        labels = torch.from_numpy(label_values)
    elif isinstance(label_values, Iterable) and not isinstance(
        label_values, bytes | str
    ):
        processed: list[torch.Tensor] = []
        for item in label_values:
            if torch.is_tensor(item):
                tensor_item = item
            elif isinstance(item, np.ndarray):
                tensor_item = torch.from_numpy(item)
            else:
                tensor_item = torch.tensor(item)

            processed.append(tensor_item.reshape(-1))

        if not processed:
            raise ValueError("Received empty label batch; cannot create tensor")

        if len(processed) == 1:
            labels = processed[0]
        else:
            labels = torch.stack(processed)
            if labels.ndim > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
    else:
        labels = torch.tensor(label_values)

    # Ensure labels are 1D for cross-entropy loss
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    elif labels.ndim > 1:
        # Flatten multi-dimensional labels to 1D
        labels = labels.view(-1)

    return labels.to(device=device, dtype=torch.long)


def _infer_input_channels(model: nn.Module) -> int | None:
    """Return the first convolutional layer's input channel count if available."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module.in_channels
    return None


def train(net, trainloader, epochs, device, model_type="cnn"):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()

            if model_type == "transformer":
                # For transformer models, use different input format
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _prepare_labels(batch["labels"], device)

                outputs = net(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                # For CNN models (ResNet, DenseNet, simple CNN)
                pixel_values = batch["pixel_values"]
                images = _prepare_images(pixel_values, device)
                labels = _prepare_labels(batch["labels"], device)

                logits = net(images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device, model_type="cnn"):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    with torch.no_grad():
        for batch in testloader:
            if model_type == "transformer":
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _prepare_labels(batch["labels"], device)

                outputs = net(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                pixel_values = batch["pixel_values"]
                images = _prepare_images(pixel_values, device)
                labels = _prepare_labels(batch["labels"], device)

                logits = net(images)

            loss += criterion(logits, labels).item()
            correct += (torch.max(logits.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


# pylint: disable=too-many-statements
def test_neural_network(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate the global model on the server test data using the appropriate method.
    This function provides the same interface as original TraceFL for compatibility.

    Parameters
    ----------
    arch : str
        The architecture type ("cnn", "resnet", "densenet", or "transformer").
    global_net_dict : dict
        Dictionary containing the global model.
    server_testdata : object
        The test dataset.
    batch_size : int, optional
        Batch size for evaluation (default is 32).

    Returns
    -------
    dict
        A dictionary with evaluation loss, accuracy, and detailed prediction
        information.
    """
    model = global_net_dict["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Determine model type
    if arch in ["cnn", "resnet", "densenet"]:
        # Vision model evaluation
        testloader = DataLoader(
            server_testdata,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DefaultDataCollator(),
        )

        all_losses = []
        all_predictions = []
        all_labels = []
        correct_indices = []
        incorrect_indices = []

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                images = _prepare_images(batch["pixel_values"], device)
                labels = _prepare_labels(batch["labels"], device)

                logits = model(images)
                loss = criterion(logits, labels)

                predictions = torch.max(logits.data, 1)[1]
                correct_mask = predictions == labels

                all_losses.append(loss.item())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Track correct/incorrect indices
                batch_start_idx = batch_idx * batch_size
                for i, is_correct in enumerate(correct_mask):
                    idx = batch_start_idx + i
                    if is_correct:
                        correct_indices.append(idx)
                    else:
                        incorrect_indices.append(idx)

        # Convert to numpy arrays then to tensors (matching original TraceFL)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        correct_indices = torch.from_numpy(np.array(correct_indices))
        incorrect_indices = torch.from_numpy(np.array(incorrect_indices))

        avg_loss = np.mean(all_losses)
        accuracy = len(correct_indices) / len(all_labels)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "eval_loss": avg_loss,
            "eval_accuracy": {"accuracy": accuracy},
            "eval_correct_indices": correct_indices,
            "eval_incorrect_indices": incorrect_indices,
            "eval_actual_labels": all_labels,
            "eval_predicted_labels": all_predictions,
        }

    if arch == "transformer":
        # Transformer model evaluation
        testloader = DataLoader(
            server_testdata,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DefaultDataCollator(),
        )

        all_losses = []
        all_predictions = []
        all_labels = []
        correct_indices = []
        incorrect_indices = []

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                predictions = torch.max(logits.data, 1)[1]
                correct_mask = predictions == labels

                all_losses.append(loss.item())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Track correct/incorrect indices
                batch_start_idx = batch_idx * batch_size
                for i, is_correct in enumerate(correct_mask):
                    idx = batch_start_idx + i
                    if is_correct:
                        correct_indices.append(idx)
                    else:
                        incorrect_indices.append(idx)

        # Convert to numpy arrays then to tensors (matching original TraceFL)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        correct_indices = torch.from_numpy(np.array(correct_indices))
        incorrect_indices = torch.from_numpy(np.array(incorrect_indices))

        avg_loss = np.mean(all_losses)
        accuracy = len(correct_indices) / len(all_labels)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "eval_loss": avg_loss,
            "eval_accuracy": {"accuracy": accuracy},
            "eval_correct_indices": correct_indices,
            "eval_incorrect_indices": incorrect_indices,
            "eval_actual_labels": all_labels,
            "eval_predicted_labels": all_predictions,
        }

    raise ValueError(f"Architecture {arch} not supported")
