# Flower Datasets

[![GitHub license](https://img.shields.io/github/license/flwrlabs/flower)](https://github.com/flwrlabs/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/flwrlabs/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/flwrlabs/flower/actions/workflows/framework.yml/badge.svg)
![Downloads](https://pepy.tech/badge/flwr-datasets)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.ai/join-slack)

Flower Datasets (`flwr-datasets`) is a library to quickly and easily create datasets for federated learning, federated evaluation, and federated analytics. It was created by the `Flower Labs` team that also created Flower: A Friendly Federated AI Framework.


> [!TIP]
> For complete documentation that includes API docs, how-to guides and tutorials, please visit the [Flower Datasets Documentation](https://flower.ai/docs/datasets/) and for full FL example see the [Flower Examples page](https://github.com/flwrlabs/flower/tree/main/hub/apps).

## Installation

For a complete installation guide visit the [Flower Datasets Documentation](https://flower.ai/docs/datasets/)

```bash
pip install flwr-datasets[vision]
```

## Overview

Flower Datasets library supports:
* **downloading datasets** - choose the dataset from Hugging Face's `datasets`,
* **partitioning datasets** - customize the partitioning scheme,
* **creating centralized datasets** - leave parts of the dataset unpartitioned (e.g. for centralized evaluation).

Thanks to using Hugging Face's `datasets` used under the hood, Flower Datasets integrates with the following popular formats/frameworks:
* Hugging Face,
* PyTorch,
* TensorFlow,
* Numpy,
* Pandas,
* Jax,
* Arrow.

Create **custom partitioning schemes** or choose from the **implemented [partitioning schemes](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html#module-flwr_datasets.partitioner)**:

* Partitioner (the abstract base class) `Partitioner`
* IID partitioning `IidPartitioner(num_partitions)`
* Dirichlet partitioning `DirichletPartitioner(num_partitions, partition_by, alpha)`
* Distribution partitioning `DistributionPartitioner(distribution_array, num_partitions, num_unique_labels_per_partition, partition_by, preassigned_num_samples_per_label, rescale)`
* InnerDirichlet partitioning `InnerDirichletPartitioner(partition_sizes, partition_by, alpha)`
* Pathological partitioning `PathologicalPartitioner(num_partitions, partition_by, num_classes_per_partition, class_assignment_mode)`
* Natural ID partitioning `NaturalIdPartitioner(partition_by)`
* Size based partitioning (the abstract base class for the partitioners dictating the division based the number of samples) `SizePartitioner`
* Linear partitioning `LinearPartitioner(num_partitions)`
* Square partitioning `SquarePartitioner(num_partitions)`
* Exponential partitioning `ExponentialPartitioner(num_partitions)`
* more to come in the future releases (contributions are welcome).
<p align="center">
  <img src="./docs/source/_static/readme/comparison_of_partitioning_schemes.png" alt="Comparison of partitioning schemes."/>
  <br>
  <em>Comparison of Partitioning Schemes on CIFAR10</em>
</p>

PS: This plot was generated using a library function (see [flwr_datasets.visualization](https://flower.ai/docs/datasets/ref-api/flwr_datasets.visualization.html) package for more).


## Usage

Flower Datasets exposes the `FederatedDataset` abstraction to represent the dataset needed for federated learning/evaluation/analytics. It has two powerful methods that let you handle the dataset preprocessing: `load_partition(partition_id, split)` and `load_split(split)`.

Here's a basic quickstart example of how to partition the MNIST dataset:

```
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioners import IidPartitioner

# The train split of the MNIST dataset will be partitioned into 100 partitions
partitioner = IidPartitioner(num_partitions=100)
fds = FederatedDataset("ylecun/mnist", partitioners={"train": partitioner})

partition = fds.load_partition(0)

centralized_data = fds.load_split("test")
```

For more details, please refer to the specific how-to guides or tutorials. They showcase customization and more advanced features.
