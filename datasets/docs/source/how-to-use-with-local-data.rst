Use with Local Data
===================

You can partition your local files and Python objects in
``Flower Datasets`` library using any available ``Partitioner``.

This guide details how to create a `Hugging Face <https://huggingface.co/>`_ `Dataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset>`_ which is the required type of input for Partitioners.
We will cover:

* local files: CSV, JSON, image, audio,
* in-memory data: dictionary, list, pd.DataFrame, np.ndarray.


General Overview
----------------
An all-in-one dataset preparation (downloading, preprocessing, partitioning) happens
using `FederatedDataset <ref-api/flwr_datasets.FederatedDataset.html>`_. However, we
will use only the `Partitioner` here since we use locally accessible data.

The rest of this guide will explain how to create a
`Dataset <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset>`_
from local files and existing (in memory) Python objects.

Local Files
-----------
CSV
^^^
.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Single file
  data_files = "path-to-my-file.csv"

  # Multiple Files
  data_files = [ "path-to-my-file-1.csv", "path-to-my-file-2.csv", ...]
  dataset = load_dataset("csv", data_files=data_files)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

JSON
^^^^

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Single file
  data_files = "path-to-my-file.json"

  # Multiple Files
  data_files = [ "path-to-my-file-1.json", "path-to-my-file-2.json", ...]
  dataset = load_dataset("json", data_files=data_files)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)


Image
^^^^^
You can create an image dataset in two ways:

1) give a path the directory

The directory needs to be structured in the following way: dataset-name/split/class/name. For example:

.. code-block::

  mnist/train/1/unique_name.png
  mnist/train/1/unique_name.png
  mnist/train/2/unique_name.png
  ...
  mnist/test/1/unique_name.png
  mnist/test/1/unique_name.png
  mnist/test/2/unique_name.png

Then, the path you can give is `./mnist`.

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Directly from a directory
  dataset_dict = load_dataset("imagefolder", data_dir="/path/to/folder")
  # Note that what we just loaded is a DatasetDict, we need to choose a single split
  # and assign it to the partitioner.dataset
  # e.g. "train" split but that depends on the structure of your directory
  dataset = dataset_dict["train"]

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

2) create a dataset from a CSV/JSON file and cast the path column to Image.

.. code-block:: python

  from datasets import Image, load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset = load_dataset(...)
  dataset = dataset.cast_column("path", Image())

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)


Audio
^^^^^
Analogously to the image datasets, there are two methods here:

1) give a path to the directory

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset_dict = load_dataset("audiofolder", data_dir="/path/to/folder")
  # Note that what we just loaded is a DatasetDict, we need to choose a single split
  # and assign it to the partitioner.dataset
  # e.g. "train" split but that depends on the structure of your directory
  dataset = dataset_dict["train"]

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

2) create a dataset from a CSV/JSON file and cast the path column to Audio.

.. code-block:: python

  from datasets import Audio, load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset = load_dataset(...)
  dataset = dataset.cast_column("path", Audio())

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

In-Memory
---------

From dictionary
^^^^^^^^^^^^^^^
.. code-block:: python

  from datasets import Dataset
  from flwr_datasets.partitioner import ChosenPartitioner
  data = {"features": [1, 2, 3], "labels": [0, 0, 1]}
  dataset = Dataset.from_dict(data)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From list
^^^^^^^^^
.. code-block:: python

  from datasets import Dataset
  from flwr_datasets.partitioner import ChosenPartitioner
  
  my_list = [
    {"features": 1, "labels": 0},
    {"features": 2, "labels": 0},
    {"features": 3, "labels": 1}
  ]
  dataset = Dataset.from_list(my_list)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From pd.DataFrame
^^^^^^^^^^^^^^^^^
.. code-block:: python

  from datasets import Dataset
  from flwr_datasets.partitioner import ChosenPartitioner
  
  data = {"features": [1, 2, 3], "labels": [0, 0, 1]}
  df = pd.DataFrame(data)
  dataset = Dataset.from_pandas(df)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From np.ndarray
^^^^^^^^^^^^^^^
The np.ndarray will be first transformed to pd.DataFrame

.. code-block:: python

  from datasets import Dataset
  from flwr_datasets.partitioner import ChosenPartitioner
  
  data = np.array([[1, 2, 3], [0, 0, 1]]).T
  # You can add the column names by passing columns=["features", "labels"]
  df = pd.DataFrame(data)
  dataset = Dataset.from_pandas(df)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

Partitioner Details
-------------------
Partitioning is triggered automatically during the first ``load_partition`` call.
You do not need to call any “do_partitioning” method.

Partitioner abstraction is designed to allow for a single dataset assignment.

.. code-block:: python

  partitioner.dataset = your_dataset # (your_dataset must be of type dataset.Dataset)

If you need to do the same partitioning on a different dataset, create a new Partitioner
for that, e.g.:

.. code-block:: python

  from flwr_datasets.partitioner import IidPartitioner

  iid_partitioner_for_mnist = IidPartitioner(num_partitions=10)
  iid_partitioner_for_mnist.dataset = mnist_dataset

  iid_partitioner_for_cifar = IidPartitioner(num_partitions=10)
  iid_partitioner_for_cifar.dataset = cifar_dataset

If your local dataset is already sorted (for example by label or timestamp), use
``shuffle=True`` so the IID shards are created from a randomized order:

.. code-block:: python

  shuffled_iid_partitioner = IidPartitioner(
      num_partitions=10,
      shuffle=True,
      seed=42,
  )
  shuffled_iid_partitioner.dataset = your_dataset


More Resources
--------------
If you are looking for more details or you have not found the format you are looking for, please visit the `HuggingFace Datasets docs <https://huggingface.co/docs/datasets/index>`_.
This guide is based on the following ones:

* `General Information <https://huggingface.co/docs/datasets/en/loading>`_
* `Tabular Data <https://huggingface.co/docs/datasets/en/tabular_load>`_
* `Image Data <https://huggingface.co/docs/datasets/en/image_load>`_
* `Audio Data <https://huggingface.co/docs/datasets/en/audio_load>`_
