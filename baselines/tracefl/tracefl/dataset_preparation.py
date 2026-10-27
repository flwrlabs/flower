"""Handle the dataset partitioning and (optionally) complex downloads.

This module provides dataset preparation functionality for TraceFL baseline, supporting
multiple datasets and partitioning strategies without caching.
"""

import logging
import random
from collections import Counter, defaultdict
from functools import partial

import torch
from datasets import Dataset, DatasetDict, load_dataset
from flwr_datasets import FederatedDataset

# import torchvision.transforms as transforms  # Not used in current implementation
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

# Optional imports for medical datasets
try:
    from medmnist import OrganAMNIST, PathMNIST

    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    OrganAMNIST = None
    PathMNIST = None


def train_test_transforms_factory(cfg):
    """Create and return train and test transformation functions for image datasets.

    Parameters
    ----------
    cfg : object
        A configuration object that must include:
            - dname: Name of the dataset (e.g. "cifar10", "mnist").

    Returns
    -------
    dict
        A dictionary with two keys:
            - 'train': Transformation function for training data.
            - 'test': Transformation function for test data.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.
    """
    train_transforms = None
    test_transforms = None

    if cfg.dname in ["cifar10", "cifar100"]:

        def apply_train_transform_cifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            # Convert list of images to tensor
            example["pixel_values"] = torch.stack(
                [transform(image.convert("RGB")) for image in example["img"]]
            )

            # Handle different label column names
            if "fine_label" in example:
                example["label"] = torch.tensor(example["fine_label"])
                del example["fine_label"]
            else:
                example["label"] = torch.tensor(example["label"])

            del example["img"]
            return example

        def apply_test_transform_cifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            # Convert list of images to tensor
            example["pixel_values"] = torch.stack(
                [transform(image.convert("RGB")) for image in example["img"]]
            )

            # Handle different label column names
            if "fine_label" in example:
                example["label"] = torch.tensor(example["fine_label"])
                del example["fine_label"]
            else:
                example["label"] = torch.tensor(example["label"])

            del example["img"]
            return example

        train_transforms = apply_train_transform_cifar
        test_transforms = apply_test_transform_cifar

    elif cfg.dname == "mnist":

        def apply_train_transform_mnist(example):
            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )
            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["image"]
            ]
            example["label"] = torch.tensor(example["label"])
            return example

        def apply_test_transform_mnist(example):
            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )
            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["image"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["image"]
            return example

        train_transforms = apply_train_transform_mnist
        test_transforms = apply_test_transform_mnist

    else:
        raise ValueError(f"Unknown dataset: {cfg.dname}")

    return {"train": train_transforms, "test": test_transforms}


def _initialize_medical_dataset(cfg, _dat_partitioner_func, _fetch_only_test_data):
    """Initialize and process a medical dataset (MedMNIST) by applying TraceFL's RGB-
    compatible preprocessing pipeline.

    Parameters
    ----------
    cfg : object
        Configuration object containing dataset and partitioning parameters.
    dat_partitioner_func : function
        The function that partitions the dataset into client and server splits.
    fetch_only_test_data : bool
        If True, only the test data will be fetched.

    Returns
    -------
    dict
        A dictionary containing:
            - 'client2data': Mapping of client IDs to transformed training data.
            - 'server_data': Transformed test data.
    """
    # Import the correct dataset class
    if cfg.dname == "pathmnist":
        data_class = PathMNIST
    elif cfg.dname == "organamnist":
        data_class = OrganAMNIST
    else:
        raise ValueError(f"Unknown medical dataset: {cfg.dname}")

    # Load MedMNIST datasets
    train_dataset = data_class(split="train", download=True)
    test_dataset = data_class(split="test", download=True)

    # Convert to Hugging Face format
    train_hf = Dataset.from_dict(
        {
            "img": [train_dataset[i][0] for i in range(len(train_dataset))],
            "label": [train_dataset[i][1] for i in range(len(train_dataset))],
        }
    )

    test_hf = Dataset.from_dict(
        {
            "img": [test_dataset[i][0] for i in range(len(test_dataset))],
            "label": [test_dataset[i][1] for i in range(len(test_dataset))],
        }
    )

    # Apply medical dataset transforms (convert to grayscale tensors)
    def apply_medical_transform(example):
        transform = Compose(
            [
                Resize((32, 32)),  # Match original TraceFL size
                ToTensor(),
                Normalize((0.5,), (0.5,)),  # Grayscale normalization
            ]
        )
        # Convert PIL to tensor - batched mode processing (matches original TraceFL)
        example["pixel_values"] = [
            transform(img.convert("L")) for img in example["img"]
        ]
        example["label"] = torch.tensor(example["label"])
        del example["img"]
        return example

    # For medical datasets, use simple partitioning instead of FederatedDataset
    # This avoids the DatasetDict issue with FederatedDataset

    # Apply transforms first - batched mode processing (matches original TraceFL)
    train_transformed = train_hf.map(
        apply_medical_transform, batched=True, batch_size=256, num_proc=1
    ).with_format("torch")
    test_transformed = test_hf.map(
        apply_medical_transform, batched=True, batch_size=256, num_proc=1
    ).with_format("torch")

    # Simple partitioning for medical datasets
    # Set random seed for reproducibility
    random.seed(42)

    # Partition training data among clients
    client2data = defaultdict(list)
    client2class = defaultdict(lambda: defaultdict(int))

    # Simple random partitioning
    for i, example in enumerate(train_transformed):
        client_id = str(i % cfg.num_clients)
        client2data[client_id].append(example)
        label_key = _label_to_str(example["label"])
        client2class[client_id][label_key] += 1

    # Convert lists to datasets
    client_datasets = {}
    for client_id, examples in client2data.items():
        client_datasets[client_id] = Dataset.from_list(examples)

    # Limit server test data
    if len(test_transformed) > cfg.max_server_data_size:
        server_data = test_transformed.select(range(cfg.max_server_data_size))
    else:
        server_data = test_transformed

    return {
        "client2data": client_datasets,
        "server_data": server_data,
        "client2class": {
            cid: dict(label_counts) for cid, label_counts in client2class.items()
        },
        "fds": None,  # No FederatedDataset for medical datasets
    }


def _initialize_text_dataset(cfg, _dat_partitioner_func, _fetch_only_test_data):
    """Initialize and process a text dataset by applying tokenization.

    Parameters
    ----------
    cfg : object
        Configuration object containing dataset and partitioning parameters.
    dat_partitioner_func : function
        The function that partitions the dataset into client and server splits.
    fetch_only_test_data : bool
        If True, only the test data will be fetched.

    Returns
    -------
    dict
        A dictionary containing:
            - 'client2data': Mapping of client IDs to transformed training data.
            - 'server_data': Transformed test data.
    """
    # Load text dataset
    if cfg.dname == "dbpedia_14":
        train_dataset = load_dataset("dbpedia_14", split="train")
        test_dataset = load_dataset("dbpedia_14", split="test")
    elif cfg.dname == "yahoo_answers_topics":
        train_dataset = load_dataset("yahoo_answers_topics", split="train")
        test_dataset = load_dataset("yahoo_answers_topics", split="test")
    else:
        raise ValueError(f"Unknown text dataset: {cfg.dname}")

    # For text datasets, use simple partitioning instead of FederatedDataset
    # This avoids the DatasetDict issue with FederatedDataset

    # Apply text transforms first
    def apply_text_transform(example):
        # Determine batch size from available columns
        if "label" in example:
            batch_size = len(example["label"])
            label_key = "label"
        elif "labels" in example:
            batch_size = len(example["labels"])
            label_key = "labels"
        elif "topic" in example:
            batch_size = len(example["topic"])
            label_key = "topic"
        else:
            # Fallback: use any available column to determine batch size
            available_keys = list(example.keys())
            if available_keys:
                batch_size = len(example[available_keys[0]])
                label_key = available_keys[0]  # Use first available key as label
            else:
                raise ValueError("No columns found in dataset")

        # Create placeholder tokenized inputs for each item in the batch
        example["input_ids"] = [[1, 2, 3, 4, 5] for _ in range(batch_size)]
        example["attention_mask"] = [[1, 1, 1, 1, 1] for _ in range(batch_size)]

        # Handle labels - rename to 'label' for consistency
        if label_key != "label":
            example["label"] = example[label_key]
            del example[label_key]

        # Remove text columns to save memory
        for col in ["title", "content", "question_title", "question_content", "answer"]:
            if col in example:
                del example[col]

        return example

    # Apply transforms first
    train_transformed = train_dataset.map(
        apply_text_transform, batched=True, batch_size=256, num_proc=1
    ).with_format("torch")
    test_transformed = test_dataset.map(
        apply_text_transform, batched=True, batch_size=256, num_proc=1
    ).with_format("torch")

    # Simple partitioning for text datasets
    # Set random seed for reproducibility
    random.seed(42)

    # Partition training data among clients
    client2data = defaultdict(list)
    client2class = defaultdict(lambda: defaultdict(int))

    # Simple random partitioning
    for i, example in enumerate(train_transformed):
        client_id = str(i % cfg.num_clients)
        client2data[client_id].append(example)
        label_key = _label_to_str(example["label"])
        client2class[client_id][label_key] += 1

    # Convert lists to datasets
    client_datasets = {}
    for client_id, examples in client2data.items():
        client_datasets[client_id] = Dataset.from_list(examples)

    # Limit server test data
    if len(test_transformed) > cfg.max_server_data_size:
        server_data = test_transformed.select(range(cfg.max_server_data_size))
    else:
        server_data = test_transformed

    return {
        "client2data": client_datasets,
        "server_data": server_data,
        "client2class": {
            cid: dict(label_counts) for cid, label_counts in client2class.items()
        },
        "fds": None,  # No FederatedDataset for text datasets
    }


def _initialize_image_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    """Initialize and process an image dataset by applying train/test transformations.

    Parameters
    ----------
    cfg : object
        A configuration object containing dataset and partitioning parameters.
    dat_partitioner_func : function
        The function that partitions the dataset into client and server splits.
    fetch_only_test_data : bool
        If True, only the test data will be fetched.

    Returns
    -------
    dict
        A dictionary containing:
            - 'client2data': Mapping of client IDs to transformed training data.
            - 'server_data': Transformed test data.
    """
    # Set correct label column based on dataset
    if cfg.dname == "cifar100":
        target_label_col = "fine_label"  # CIFAR-100 uses fine_label instead of label
    else:
        target_label_col = "label"

    d = dat_partitioner_func(cfg, target_label_col, fetch_only_test_data)
    transforms = train_test_transforms_factory(cfg=cfg)

    d["client2data"] = {
        k: v.map(
            transforms["train"], batched=True, batch_size=256, num_proc=8
        ).with_format("torch")
        for k, v in d["client2data"].items()
    }
    d["server_data"] = (
        d["server_data"]
        .map(transforms["test"], batched=True, batch_size=256, num_proc=8)
        .with_format("torch")
    )
    return d


def _label_to_str(label):
    """Convert label values from datasets into a normalized string representation."""
    # Handle common tensor/int formats
    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            label = label.item()
        else:
            label = label.tolist()
    if isinstance(label, list | tuple) and len(label) == 1:
        label = label[0]
    return str(label)


def get_labels_count(partition, target_label_col):
    """Count the number of occurrences for each label in a dataset partition.

    Parameters
    ----------
    partition : Dataset or list-like
        The dataset partition where each example contains the target label.
    target_label_col : str
        The key corresponding to the label in each example.

    Returns
    -------
    dict
        A dictionary mapping each label to its count.
    """
    label2count = Counter(
        _label_to_str(example[target_label_col])  # type: ignore
        for example in partition
    )  # type: ignore
    return dict(label2count)


def _fix_partition(cfg, c_partition, target_label_col):
    """Clean and truncate a client data partition based on minimum sample requirements.

    Parameters
    ----------
    cfg : object
        Configuration object with attributes:
            - max_per_client_data_size : int, maximum allowed data per client.
            - batch_size : int, the batch size for training.
    c_partition : Dataset or list-like
        The original dataset partition for a client.
    target_label_col : str
        The key corresponding to the label in each example.

    Returns
    -------
    dict
        A dictionary with keys:
            - 'partition': The cleaned and possibly truncated dataset partition.
            - 'partition_labels_count': The count of labels in the partition.
    """
    label2count = get_labels_count(c_partition, target_label_col)

    filtered_labels = {
        label: count for label, count in label2count.items() if count >= 10
    }

    indices_to_select = [
        i
        for i, example in enumerate(c_partition)
        if _label_to_str(example[target_label_col]) in filtered_labels  # type: ignore
    ]

    ds = c_partition.select(indices_to_select)

    assert (
        cfg.max_per_client_data_size > 0
    ), f"max_per_client_data_size: {cfg.max_per_client_data_size}"

    if len(ds) > cfg.max_per_client_data_size:
        ds = ds.select(range(cfg.max_per_client_data_size))

    if len(ds) % cfg.batch_size == 1:
        ds = ds.select(range(len(ds) - 1))

    partition_labels_count_raw = get_labels_count(ds, target_label_col)
    partition_labels_count = {
        str(label): int(count) for label, count in partition_labels_count_raw.items()
    }
    return {"partition": ds, "partition_labels_count": partition_labels_count}


# pylint: disable=too-many-positional-arguments
def _partition_helper(
    partitioner,
    cfg,
    target_label_col,
    _fetch_only_test_data,
    subtask,
    train_dataset=None,
    test_dataset=None,
):
    """Partition the dataset among clients and prepare the server test data.

    Parameters
    ----------
    partitioner : object
        An instance of a partitioner (e.g. DirichletPartitioner, ShardPartitioner).
    cfg : object
        A configuration object containing:
            - num_clients: Number of clients.
            - max_server_data_size: Maximum number of samples for the server.
            - max_per_client_data_size: Maximum samples allowed per client.
            - batch_size: Batch size used during training.
            - dname: Dataset name.
    target_label_col : str
        The label key used for partitioning.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        If specified, indicates a subset of data to be used.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'client2data': Mapping of client IDs to their dataset partitions.
            - 'server_data': The server test dataset.
            - 'client2class': Mapping of client IDs to label counts.
            - 'fds': The FederatedDataset instance used for partitioning.
    """
    clients_class = []
    clients_data = []
    server_data = None
    fds = None

    # Handle custom datasets or use FederatedDataset
    if train_dataset is not None and test_dataset is not None:
        # Use provided datasets
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
        fds = FederatedDataset(
            dataset=dataset_dict, partitioners={"train": partitioner}
        )
    else:
        # Use FederatedDataset with dataset name
        if subtask is not None:
            fds = FederatedDataset(
                dataset=cfg.dname, partitioners={"train": partitioner}, subset=subtask
            )
        else:
            fds = FederatedDataset(
                dataset=cfg.dname, partitioners={"train": partitioner}
            )

    if len(fds.load_split("test")) < cfg.max_server_data_size:
        server_data = fds.load_split("test")
    else:
        server_data = fds.load_split("test").select(range(cfg.max_server_data_size))

    logging.info("Partition helper: Keys in the dataset are: %s", server_data[0].keys())

    for cid in range(cfg.num_clients):
        client_partition = fds.load_partition(cid)
        temp_dict = {}

        if cfg.max_per_client_data_size > 0:
            logging.info(" Fixing partition for client %s", cid)
            temp_dict = _fix_partition(cfg, client_partition, target_label_col)
        else:
            logging.info(" No data partition fix required for client %s", cid)
            temp_dict = {
                "partition": client_partition,
                "partition_labels_count": get_labels_count(
                    client_partition, target_label_col
                ),
            }

        if len(temp_dict["partition"]) >= cfg.batch_size:
            clients_data.append(temp_dict["partition"])
            clients_class.append(temp_dict["partition_labels_count"])

    logging.info(" -- fix partition is done --")
    client2data = {f"{client_id}": v for client_id, v in enumerate(clients_data)}
    client2class = {f"{client_id}": v for client_id, v in enumerate(clients_class)}
    return {
        "client2data": client2data,
        "server_data": server_data,
        "client2class": client2class,
        "fds": fds,
    }


# pylint: disable=too-many-positional-arguments
def _dirichlet_data_distribution(
    cfg,
    target_label_col,
    fetch_only_test_data,
    train_dataset=None,
    test_dataset=None,
    subtask=None,
):
    """Partition the dataset among clients using a Dirichlet distribution.

    Parameters
    ----------
    cfg : object
        Configuration object containing:
            - num_clients: Number of clients.
            - dirichlet_alpha: Alpha parameter for the Dirichlet distribution.
    target_label_col : str
        Key used for partitioning by label.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        A subset indicator for partitioning if needed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data', 'server_data', 'client2class', and 'fds'.
    """
    partitioner = DirichletPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        alpha=cfg.dirichlet_alpha,
        min_partition_size=0,
        self_balancing=True,
        shuffle=True,
    )

    return _partition_helper(
        partitioner,
        cfg,
        target_label_col,
        fetch_only_test_data,
        subtask,
        train_dataset,
        test_dataset,
    )


def _sharded_data_distribution(
    num_classes_per_partition, cfg, target_label_col, fetch_only_test_data, subtask=None
):
    """Partition the dataset among clients using a sharded non-IID distribution.

    Parameters
    ----------
    num_classes_per_partition : int
        Number of classes that each client should receive.
    cfg : object
        Configuration object containing:
            - num_clients: Number of clients.
    target_label_col : str
        The key used for partitioning by label.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        A subset indicator for partitioning if needed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data', 'server_data', 'client2class', and 'fds'.
    """
    partitioner = ShardPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        shard_size=2000,
        num_shards_per_partition=num_classes_per_partition,
        shuffle=True,
    )
    return _partition_helper(
        partitioner, cfg, target_label_col, fetch_only_test_data, subtask
    )


def _pathological_partitioner(
    num_classes_per_partition, cfg, target_label_col, fetch_only_test_data, subtask=None
):
    """Partition the dataset among clients using a pathological (highly non-IID)
    strategy.

    Parameters
    ----------
    num_classes_per_partition : int
        Number of classes assigned per client.
    cfg : object
        Configuration object containing:
            - num_clients: Number of clients.
    target_label_col : str
        The key used for partitioning by label.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        A subset indicator for partitioning if needed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data', 'server_data', 'client2class', and 'fds'.
    """
    partitioner = PathologicalPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        num_classes_per_partition=num_classes_per_partition,
        shuffle=True,
        class_assignment_mode="deterministic",
    )
    return _partition_helper(
        partitioner, cfg, target_label_col, fetch_only_test_data, subtask
    )


def _load_dist_based_clients_server_datasets(
    cfg, dat_partitioner_func, _fetch_only_test_data=False
):
    """Load and partition the dataset into client and server splits based on a
    distribution strategy.

    Parameters
    ----------
    cfg : object
        A configuration object containing dataset information (e.g. dname,
        architecture).
    dat_partitioner_func : function
        The partitioner function to use for splitting the data.
    fetch_only_test_data : bool, optional
        If True, only the test data is loaded (default is False).

    Returns
    -------
    dict
        A dictionary containing:
            - 'client2data': Client-specific training data.
            - 'server_data': Server test data.
            - 'client2class': Class counts per client.
            - 'fds': FederatedDataset object used for partitioning.

    Raises
    ------
    ValueError
        If the dataset name or architecture is unknown.
    """
    print(f"Dataset name: {cfg}")

    if cfg.dname in ["cifar10", "cifar100", "mnist"]:
        return _initialize_image_dataset(
            cfg, dat_partitioner_func, _fetch_only_test_data
        )
    if cfg.dname in ["pathmnist", "organamnist"]:
        return _initialize_medical_dataset(
            cfg, dat_partitioner_func, _fetch_only_test_data
        )
    if cfg.dname in ["dbpedia_14", "yahoo_answers_topics"]:
        return _initialize_text_dataset(
            cfg, dat_partitioner_func, _fetch_only_test_data
        )
    raise ValueError(f"Unknown dataset: {cfg.dname}")


class ClientsAndServerDatasets:
    """Prepare and manage the datasets for clients and the server in a federated
    setting.

    This class initializes the dataset partitioning according to the configuration,
    sets up the client and server datasets, and provides access to the processed data.

    Attributes
    ----------
    cfg : object
        Configuration object with parameters for data distribution and model settings.
    data_dist_partitioner_func : function
        The partitioner function selected based on the distribution type.
    client2data : dict
        Mapping of client IDs to their training datasets.
    server_testdata : Dataset
        The server's test dataset.
    client2class : dict
        Mapping of client IDs to label counts.
    fds : FederatedDataset
        The FederatedDataset object used for partitioning.
    """

    def __init__(self, cfg):
        """Initialize the ClientsAndServerDatasets instance.

        Parameters
        ----------
        cfg : object
            Configuration object with necessary parameters for dataset partitioning.
        """
        self.cfg = cfg
        self.data_dist_partitioner_func = None
        self._set_distribution_partitioner()
        self._setup()

    def _set_distribution_partitioner(self):
        """Set the data distribution partitioner function based on the configuration.

        The method selects the partitioner function to use (e.g. Dirichlet, sharded, or
        pathological) based on cfg.data_dist.dist_type.
        """
        if self.cfg.data_dist.dist_type == "non_iid_dirichlet":
            self.data_dist_partitioner_func = _dirichlet_data_distribution
        elif self.cfg.data_dist.dist_type == "sharded-non-iid-1":
            self.data_dist_partitioner_func = partial(
                _sharded_data_distribution, 1
            )  # passing num_classes_per_partition
        elif self.cfg.data_dist.dist_type == "sharded-non-iid-2":
            self.data_dist_partitioner_func = partial(_sharded_data_distribution, 2)
        elif self.cfg.data_dist.dist_type == "sharded-non-iid-3":
            self.data_dist_partitioner_func = partial(_sharded_data_distribution, 3)
        elif self.cfg.data_dist.dist_type == "PathologicalPartitioner-1":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 1)
        elif self.cfg.data_dist.dist_type == "PathologicalPartitioner-2":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 2)
        elif self.cfg.data_dist.dist_type == "PathologicalPartitioner-3":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 3)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.cfg.data_dist.dist_type}"
            )

    def _setup_hugging_dataset(self):
        """Set up the Hugging Face dataset by partitioning it among clients and the
        server.

        This method loads the dataset, applies tokenization or transformation (depending
        on the type), and stores the client and server data along with class
        distributions.
        """
        # pylint: disable=attribute-defined-outside-init
        # Attributes intentionally set here for lazy loading in property
        d = _load_dist_based_clients_server_datasets(
            self.cfg.data_dist, self.data_dist_partitioner_func
        )
        self.client2data = d["client2data"]
        self.server_testdata = d["server_data"]
        self.client2class = d["client2class"]
        self.fds = d["fds"]
        logging.info("client2class: %s", self.client2class)

        data_per_client = [len(dl) for dl in self.client2data.values()]
        logging.info("Data per client in experiment %s", data_per_client)
        min_data = min(len(dl) for dl in self.client2data.values())
        logging.info("Min data on a client: %s", min_data)

    def _setup(self):
        """Set up method to initialize the Hugging Face dataset."""
        self._setup_hugging_dataset()

    def get_data(self):
        """Retrieve the prepared client and server datasets for federated simulation.

        Returns
        -------
        dict
            A dictionary containing:
                - 'server_testdata': The server's test dataset.
                - 'client2class': Label count per client.
                - 'client2data': Client training datasets.
                - 'fds': The FederatedDataset object used for partitioning.
        """
        return {
            "server_testdata": self.server_testdata,
            "client2class": self.client2class,
            "client2data": self.client2data,
            "fds": self.fds,
        }
