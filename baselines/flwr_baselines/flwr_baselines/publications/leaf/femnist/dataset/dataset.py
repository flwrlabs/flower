"""FEMNIST dataset creation module."""

import pathlib
from logging import INFO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from flwr_baselines.publications.leaf.femnist.dataset.nist_preprocessor import (
    NISTPreprocessor,
)
from flwr_baselines.publications.leaf.femnist.dataset.nist_sampler import NistSampler
from flwr_baselines.publications.leaf.femnist.dataset.zip_downloader import (
    ZipDownloader,
)


class NISTLikeDataset(Dataset):
    """Dataset representing NIST or preprocessed variant of it."""

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        labels: np.ndarray,
        transform: transforms = transforms.ToTensor(),
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.ToTensor() if transform is None else transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


def create_dataset(df_info: pd.DataFrame, labels: np.ndarray) -> NISTLikeDataset:
    """Instantiate NISTLikeDataset.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains paths to images
    labels: np.ndarray
        0 till N-1 classes labels in the same order as in df_info

    Returns
    -------
    nist_like_dataset: NISTLikeDataset
        created dataset
    """
    nist_like_dataset = NISTLikeDataset(df_info["path"].values, labels)
    return nist_like_dataset


def create_partition_list(df_info: pd.DataFrame) -> List[List[int]]:
    """Create list of list with int masks identifying writers.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains writer_id information

    Returns
    -------
    division_list: List[List[int]]
        List of lists of indices to identify unique writers
    """
    writers_ids = df_info["writer_id"].values
    unique_writers = np.unique(writers_ids)
    indices = {
        writer_id: np.where(writers_ids == writer_id)[0].tolist()
        for writer_id in unique_writers
    }
    return list(indices.values())


def partition_dataset(
    dataset: Dataset, division_list: List[List[int]]
) -> List[Dataset]:
    """
    Partition dataset for niid settings - by writer id (each partition has only single writer data).
    Parameters
    ----------
    dataset: Dataset
        dataset of all images
    division_list: List[List[int]]
        list of lists of indices to identify unique writers

    Returns
    -------
    subsets: List[Dataset]
        subsets of datasets divided by writer id
    """
    subsets = []
    for sequence in division_list:
        subsets.append(Subset(dataset, sequence))
    return subsets


# pylint: disable=too-many-locals
def train_valid_test_partition(
    partitioned_dataset: List[Dataset],
    train_split: float = 0.9,
    validation_split: float = 0.0,
    test_split: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).

    Parameters
    ----------
    partitioned_dataset: List[Dataset]
        partitioned datasets
    train_split: float
        part of the data used for training
    validation_split: float
        part of the data used for validation
    test_split: float
        part of the data used for testing
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_subsets = []
    validation_subsets = []
    test_subsets = []

    for subset in partitioned_dataset:
        subset_len = len(subset)
        train_len = int(train_split * subset_len)
        # Do this checkup for full dataset use
        # Consider the case sample size == 5 and
        # train_split = 0.5 test_split = 0.5
        # if such check as below is not performed
        # one sample will be missing
        if validation_split == 0.0:
            test_len = subset_len - train_len
            val_len = 0
        else:
            test_len = int(test_split * subset_len)
            val_len = subset_len - train_len - test_len
        train_dataset, validation_dataset, test_dataset = random_split(
            subset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_subsets.append(train_dataset)
        validation_subsets.append(validation_dataset)
        test_subsets.append(test_dataset)
    return train_subsets, validation_subsets, test_subsets


def transform_datasets_into_dataloaders(
    datasets: List[Dataset], **dataloader_kwargs
) -> List[DataLoader]:
    """Transform datasets into dataloaders.

    Parameters
    ----------
    datasets: List[Dataset]
        list of datasets
    dataloader_kwargs
        arguments to DataLoader

    Returns
    -------
    dataloader: List[DataLoader]
        list of dataloaders
    """
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, **dataloader_kwargs))
    return dataloaders


# pylint: disable=too-many-arguments
def create_federated_dataloaders(
    sampling_type: str,
    dataset_fraction: float,
    batch_size: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """Create the federated dataloaders by following all the preprocessing
    steps and division.

    Parameters
    ----------
    sampling_type: str
        "niid" or "iid"
    dataset_fraction: float
        fraction of the total data that will be used for sampling
    batch_size: int
        batch size
    train_fraction, validation_fraction, test_fraction: float
        fraction of each local dataset used for training, validation, testing
    random_seed: int
        random seed for data shuffling

    Returns
    -------
    """
    if train_fraction + validation_fraction + test_fraction != 1.0:
        raise ValueError(
            "The fraction of train, validation and test should add up to 1.0."
        )
    # Download and unzip the data
    log(INFO, "NIST data downloading started")
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("by_class", "data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader(
        "by_write", "data/raw", nist_by_writer_url
    )
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
    log(INFO, "NIST data downloading done")

    # Preprocess the data
    log(INFO, "Preprocessing of the NIST data started")
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
    log(INFO, "Preprocessing of the NIST data done")

    # Create information for sampling
    log(INFO, "Creation of the sampling information started")
    df_info_path = pathlib.Path("data/processed_FeMNIST/processed_images_to_labels.csv")
    df_info = pd.read_csv(df_info_path, index_col=0)
    sampler = NistSampler(df_info)
    sampled_data_info = sampler.sample(
        sampling_type, dataset_fraction, random_seed=random_seed
    )
    sampled_data_info_path = pathlib.Path(
        f"data/processed_FeMNIST/{sampling_type}_sampled_images_to_labels.csv"
    )
    sampled_data_info.to_csv(sampled_data_info_path)
    log(INFO, "Creation of the sampling information done")

    # Create a list of DataLoaders
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets started")
    sampled_data_info = pd.read_csv(sampled_data_info_path)
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])
    full_dataset = create_dataset(sampled_data_info, labels)
    division_list = create_partition_list(sampled_data_info)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    (
        partitioned_train,
        partitioned_validation,
        partitioned_test,
    ) = train_valid_test_partition(
        partitioned_dataset,
        random_seed=random_seed,
        train_split=train_fraction,
        validation_split=validation_fraction,
        test_split=test_fraction,
    )
    trainloaders = transform_datasets_into_dataloaders(
        partitioned_train, batch_size=batch_size
    )
    valloaders = transform_datasets_into_dataloaders(
        partitioned_validation, batch_size=batch_size
    )
    testloaders = transform_datasets_into_dataloaders(
        partitioned_test, batch_size=batch_size
    )
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets done")
    return trainloaders, valloaders, testloaders
