"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration


def get_model(device, num_classes, compile: bool = True):
    """Create model: Whisper-tiny Encoder + classification head."""
    encoder = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).get_encoder()
    encoder = encoder.to(device)
    if compile:
        encoder = torch.compile(encoder)

    # This classification head is 782K parameters
    # This is the only part of the model that is trained in federation
    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(1500, 128, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(1),
        torch.nn.Linear(128 * 384, num_classes),
    ).to(device)
    return encoder, classifier


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.n += 1

    def __call__(self):
        return self.total / self.n


def construct_balanced_sampler(trainset):
    hist, _ = np.histogram(trainset["targets"], bins=12)
    # Mask of non-zeros
    hist_mask = hist > 0
    w_per_class = len(trainset) / (
        hist + 1
    )  # avoid dividing by zeros  # doesn't have to add up to 1 (relative is what matters)
    w_per_class += 1  # needed in case trainset has very few samples
    # Apply mask so we don't attempt sampling classes that aren't present
    w_per_class *= hist_mask
    w_ss = [w_per_class[t] for t in trainset["targets"]]
    return WeightedRandomSampler(w_ss, len(w_ss))


def train_one_epoch(
    model,
    classifier,
    optimizer,
    criterion,
    dataloader,
    device,
    disable_tqdm: bool = False,
):
    """Train the classification head.

    This is a very standard looking way of training PyTorch models.
    """
    model.eval()
    classifier.train()
    classifier.to(device)
    loss_avg, acc_avg = RunningAvg(), RunningAvg()
    avg_loss, avg_acc = 0.0, 0.0
    with tqdm(total=len(dataloader.dataset), disable=disable_tqdm) as t:
        for b in dataloader:
            optimizer.zero_grad()
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            with torch.no_grad():
                res = model(data)[0]

            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(resres.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / data.shape[0]
            loss_ = loss.cpu().item()

            loss_avg.update(loss_)
            acc_avg.update(acc)

            t.update(data.shape[0])
            avg_loss, avg_acc = loss_avg(), acc_avg()
            t.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.4f}"})

    return avg_loss, avg_acc


def eval_model(model, classifier, criterion, dataloader, device):
    """Evaluate a model on a validation/test set.

    This is a very normal looking way of doing this with PyTorch.
    """
    model.eval()
    classifier.eval()
    classifier.to(device)
    correct = 0
    loss_ = 0
    total = 0
    with torch.no_grad():
        for b in dataloader:
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            res = model(data)[0]
            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            _, predicted = torch.max(resres.data, 1)
            correct += (predicted == labels).sum().item()
            total += data.shape[0]
            loss_ += loss.cpu().item()

    accuracy = correct / total
    loss = loss_ / total

    return loss, accuracy
