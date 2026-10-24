"""bcfl_edge: Google Speech Commands keyword partitions (Sec. IV-A of the paper)."""

import torch
import torch.nn.functional as F
from datasets import DatasetDict, Value
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader

KEYWORDS = ("go", "stop", "left", "right", "up", "down", "yes", "no")
SAMPLE_RATE = 16_000

FDS = None  # Cache FederatedDataset


def keep_keywords(dataset: DatasetDict) -> DatasetDict:
    """Keep the eight keywords of the paper and relabel them 0-7."""
    names = dataset["train"].features["label"].names
    new_label = {names.index(word): i for i, word in enumerate(KEYWORDS)}
    dataset = dataset.filter(
        lambda labels: [label in new_label for label in labels],
        input_columns="label",
        batched=True,
    )
    dataset = dataset.cast_column("label", Value("int64"))
    return dataset.map(
        lambda labels: {"label": [new_label[label] for label in labels]},
        input_columns="label",
        batched=True,
    )


def spectrogram(audio) -> torch.Tensor:
    """Log-magnitude spectrogram of a 1 s clip, resized to 32x32."""
    samples = audio.get_all_samples()
    waveform = samples.data.mean(dim=0)[:SAMPLE_RATE]
    waveform = F.pad(waveform, (0, SAMPLE_RATE - len(waveform)))
    stft = torch.stft(
        waveform,
        n_fft=255,
        hop_length=128,
        window=torch.hann_window(255),
        return_complex=True,
    )
    image = torch.log1p(stft.abs())[None, None]
    return F.interpolate(image, size=(32, 32), mode="bilinear")[0]


def _to_tensors(batch):
    return {
        "spectrogram": [spectrogram(audio) for audio in batch["audio"]],
        "label": batch["label"],
    }


def _federated_dataset(dataset: str, num_partitions: int) -> FederatedDataset:
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        FDS = FederatedDataset(
            dataset=dataset,
            subset="v0.01",
            preprocessor=keep_keywords,
            # Non-uniform, label-skewed partitions, as in the paper.
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=0.5,
                    min_partition_size=20,
                    seed=42,
                )
            },
        )
    return FDS


def load_partition(
    partition_id: int, num_partitions: int, batch_size: int, dataset: str
) -> DataLoader:
    """Training data of one device."""
    fds = _federated_dataset(dataset, num_partitions)
    partition = fds.load_partition(partition_id).with_transform(_to_tensors)
    return DataLoader(partition, batch_size=batch_size, shuffle=True)


def load_test(num_partitions: int, batch_size: int, dataset: str) -> DataLoader:
    """Centralized test set, used to evaluate the global model."""
    fds = _federated_dataset(dataset, num_partitions)
    test = fds.load_split("test").with_transform(_to_tensors)
    return DataLoader(test, batch_size=batch_size)
