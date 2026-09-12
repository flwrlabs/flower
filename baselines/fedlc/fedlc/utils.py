"""fedrs: A Flower Baseline."""

from typing import Tuple


def get_ds_info(dataset: str) -> Tuple[int, str]:
    """Accepts name of dataset and returns info of number of classes and name of label
    column to partition by."""
    if dataset == "cifar10":
        return 10, "label"
    if dataset == "cifar100":
        return 100, "fine_label"
    raise ValueError(
        f"Dataset {dataset} unsupported! Only 'cifar10' and 'cifar100' are supported currently"
    )
