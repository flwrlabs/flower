"""Shared configuration enums and helpers for the GHBM baseline."""

from enum import StrEnum
from typing import TypeVar


class AlgorithmName(StrEnum):
    """Supported federated optimization algorithms."""

    FEDAVG = "fedavg"
    GHBM = "ghbm"
    LOCAL_GHBM = "localghbm"
    FED_HBM = "fedhbm"


class DatasetName(StrEnum):
    """Supported datasets."""

    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class ModelName(StrEnum):
    """Supported model families."""

    LENET = "lenet"
    RESNET = "resnet"


class NormLayer(StrEnum):
    """Supported normalization layers for ResNet."""

    GROUP = "group"
    BATCH = "batch"


EnumT = TypeVar("EnumT", bound=StrEnum)


def parse_enum(value: object, enum_cls: type[EnumT]) -> EnumT:
    """Parse a run-config value into the requested enum type."""
    return enum_cls(str(value).lower())
