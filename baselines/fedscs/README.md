---

title: Robust Federated Learning via Stable Cosine Similarity
url: https://doi.org/10.1109/ICCST63435.2025.11295691
labels: [federated learning, robust aggregation, non-iid, cosine similarity, security, cifar-10]
dataset: [CIFAR-10]
-------------------

# Robust Federated Learning via Stable Cosine Similarity

FedSCS is a robust federated learning aggregation method based on **Stable Cosine Similarity (SCS)**. The method evaluates client updates using their geometric alignment and stability to identify unreliable or potentially malicious updates without requiring trusted server-side validation data.

> [!NOTE]
> If you use this baseline in your work, please cite the original FedSCS paper as well as the Flower paper.

**Paper:** [Robust Federated Learning via Stable Cosine Similarity](https://doi.org/10.1109/ICCST63435.2025.11295691)

**Authors:** Rakib Ul Haque and Panagiotis Markopoulos

**Publication:** 2025 IEEE International Carnahan Conference on Security Technology (ICCST), pp. 1–6. The paper received the Distinguished Conference Paper Award.

**Abstract:** Federated Learning (FL) remains vulnerable to performance degradation in the presence of non-IID data distributions, noisy updates, and adversarial clients. Existing robust aggregation methods often depend on heuristic thresholds, computationally expensive pairwise similarity matrices, or trusted data held by the server, limiting their scalability and practicality. FedSCS introduces a novel aggregation framework built around a client trust score called Stable Cosine Similarity (SCS). The metric captures both geometric alignment and temporal stability, enabling dynamic and interpretable trust assessment. FedSCS is fully unsupervised, hyperparameter-free, and computationally efficient, avoiding pairwise operations and external validation data. The method is evaluated on CIFAR-10 classification and DOTA-v1 object detection under clean, non-IID, and noisy-client settings. Across the evaluated scenarios, FedSCS demonstrates improved robustness compared with strong federated learning baselines.

## About this baseline

**What's implemented:** This baseline provides a Flower implementation of the FedSCS federated learning experiment for CIFAR-10 image classification. The implementation uses a convolutional neural network trained across multiple clients with non-IID data generated using a fixed-size Dirichlet partition. The current implementation supports the Flower ServerApp/ClientApp execution model and includes the FedSCS aggregation strategy.

The CIFAR-10 experiment uses 10 simulated clients, with each client receiving the same number of training samples while the class distribution varies according to a Dirichlet distribution with concentration parameter α=0.3.

**Datasets:** CIFAR-10. The training set contains 50,000 images and the test set contains 10,000 images. The training data are partitioned among 10 clients using a fixed-size Dirichlet distribution.

**Hardware Setup:** The experiments can be executed on a CUDA-capable NVIDIA GPU. The development system used for this implementation contains two NVIDIA RTX A6000 GPUs. A single CUDA-capable GPU is sufficient to run the CIFAR-10 experiment, although execution time depends on the number of clients, communication rounds, local epochs, batch size, and available GPU resources.

**Contributors:** Rakib Ul Haque, University of Texas at San Antonio.

## Experimental Setup

**Task:** Image classification on CIFAR-10 using federated learning.

**Model:** The baseline uses a custom convolutional neural network implemented in `fedscs/model.py`.

| Component       | Configuration                 |
| --------------- | ----------------------------- |
| Input           | CIFAR-10 RGB image, 32×32     |
| Conv layer 1    | 3 → 32 channels, 3×3 kernel   |
| Activation      | ReLU                          |
| Pooling         | 2×2 MaxPool                   |
| Conv layer 2    | 32 → 64 channels, 3×3 kernel  |
| Activation      | ReLU                          |
| Pooling         | 2×2 MaxPool                   |
| Conv layer 3    | 64 → 128 channels, 3×3 kernel |
| Activation      | ReLU                          |
| Pooling         | 2×2 MaxPool                   |
| Fully connected | 128×4×4 → 256                 |
| Activation      | ReLU                          |
| Output layer    | 256 → 10 classes              |

The model is defined in `fedscs/model.py`.

**Dataset:**

| Dataset  | Training Samples | Test Samples | Clients | Partitioning         |   α |
| -------- | ---------------: | -----------: | ------: | -------------------- | --: |
| CIFAR-10 |           50,000 |       10,000 |      10 | Fixed-size Dirichlet | 0.3 |

The training dataset is first grouped by class. A Dirichlet distribution is sampled independently for each class to determine how that class is distributed across clients. The implementation then ensures that clients receive the same total number of training examples.

With 50,000 training samples and 10 clients, each client receives approximately 5,000 training samples.

The partitioning implementation is located in `fedscs/dataset.py`.

The CIFAR-10 test set is used by the client evaluation process.

**Training Hyperparameters:**

| Hyperparameter           |           Default |
| ------------------------ | ----------------: |
| Number of clients        |                10 |
| Number of server rounds  |                 3 |
| Client training fraction |               0.5 |
| Local epochs             |                 1 |
| Batch size               |                32 |
| Learning rate            |              0.01 |
| Optimizer                |               SGD |
| Loss function            |     Cross-Entropy |
| Dataset                  |          CIFAR-10 |
| Dirichlet α              |               0.3 |
| Random seed              |                42 |
| GPU                      | CUDA if available |

The learning rate and local training configuration can be modified through the Flower run configuration.

## Code Structure

```text
fedscs/
├── fedscs/
│   ├── __init__.py
│   ├── client_app.py
│   ├── dataset.py
│   ├── model.py
│   ├── server_app.py
│   ├── strategy.py
│   └── utils.py
├── pyproject.toml
└── README.md
```

### `fedscs/model.py`

Contains the CIFAR-10 neural network and local training/evaluation functions.

The main components are:

```text
Net
train()
test()
```

### `fedscs/dataset.py`

Downloads CIFAR-10 and provides the client-specific training and testing data loaders.

The current implementation uses the fixed-size Dirichlet partition generated for the 10-client experiment.

### `fedscs/strategy.py`

Contains the FedSCS aggregation strategy.

The strategy is responsible for evaluating client updates and performing robust aggregation based on Stable Cosine Similarity.

### `fedscs/server_app.py`

Defines the Flower `ServerApp`.

The server:

1. Initializes the global model.
2. Creates the FedSCS strategy.
3. Starts federated training.
4. Runs the configured number of communication rounds.
5. Saves the final global model.

### `fedscs/client_app.py`

Defines the Flower `ClientApp`.

Each client:

1. Identifies its partition.
2. Loads its local CIFAR-10 data.
3. Receives the global model.
4. Performs local training.
5. Returns the updated model to the server.
6. Evaluates the received global model when requested.

### `fedscs/utils.py`

Contains utility functions used by the baseline, including reproducibility utilities.

## Environment Setup

The baseline uses Python 3.12 for the standard Flower baseline environment. The current development environment uses Python 3.11 and has also been tested with the required PyTorch/CUDA configuration.

### Create the environment

```bash
pyenv virtualenv 3.12.12 fedscs
pyenv activate fedscs
```

### Install the baseline

From the directory containing `pyproject.toml`:

```bash
pip install -e .
```

The required dependencies are specified in `pyproject.toml`.

For a CUDA-enabled installation, make sure the installed PyTorch build is compatible with the NVIDIA driver on the system.

### Verify PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A CUDA-enabled installation should report:

```text
True
```

### Verify the model

```bash
python -c "from fedscs.model import Net; print('Model import successful')"
```

### Verify the dataset

```bash
python -c "from fedscs.dataset import load_data; tr, te = load_data(0, 10); print(len(tr.dataset), len(te.dataset))"
```

For the current CIFAR-10 implementation, the expected output is:

```text
5000 10000
```

### Verify utilities

```bash
python -c "from fedscs.utils import set_seed; set_seed(42); print('Utils import successful')"
```

### Verify the package

```bash
python -c "import fedscs; print('Package import successful')"
```

## Running the Experiments

Before running the experiment, verify that the application configuration in `pyproject.toml` points to the correct ServerApp and ClientApp:

```toml
[tool.flwr.app.components]
serverapp = "fedscs.server_app:app"
clientapp = "fedscs.client_app:app"
```

The default experiment can be started with:

```bash
flwr run .
```

To stream the logs:

```bash
flwr run . --stream
```

The default configuration is specified in:

```toml
[tool.flwr.app.config]
num-server-rounds = 3
fraction-train = 0.5
local-epochs = 1
batch-size = 32
learning-rate = 0.01
```

The exact configuration should be kept synchronized with the implementation in `client_app.py`, `server_app.py`, and `strategy.py`.

### Changing experiment parameters

For example:

```bash
flwr run . --run-config num-server-rounds=10,local-epochs=5,learning-rate=0.01
```

A larger experiment can therefore be executed without modifying the Python source code.

## Reproducibility

The implementation uses a fixed random seed of 42 for reproducibility.

The seed is applied to:

* Python's random number generator
* NumPy
* PyTorch
* CUDA

The corresponding utility is implemented in:

```text
fedscs/utils.py
```

## Expected Experiment

The primary experiment evaluates federated CIFAR-10 classification under non-IID client data.

The experiment consists of:

1. Downloading CIFAR-10.
2. Partitioning the 50,000 training examples among 10 clients.
3. Using a fixed-size Dirichlet distribution with α=0.3.
4. Initializing a common global CNN.
5. Sampling a fraction of clients for each training round.
6. Performing local SGD training.
7. Aggregating client updates using FedSCS.
8. Evaluating the resulting global model.
9. Saving the final model.

The implementation is designed so that the aggregation method can be compared with standard federated learning aggregation methods within the same Flower framework.

## Results

Results should be added here after the Flower implementation has been experimentally validated against the original paper.

For example:

| Method      | CIFAR-10 Accuracy |
| ----------- | ----------------: |
| FedAvg      |               TBD |
| FedProx     |               TBD |
| FedSCAFFOLD |               TBD |
| FedSCS      |               TBD |

The values in this table should only be populated after running the corresponding experiments.

## Citation

If you use this baseline, please cite the original paper:

```bibtex
@inproceedings{haque2025fedsсs,
  author={Haque, Rakib Ul and Markopoulos, Panagiotis},
  title={Robust Federated Learning via Stable Cosine Similarity},
  booktitle={2025 IEEE International Carnahan Conference on Security Technology (ICCST)},
  year={2025},
  pages={1--6},
  doi={10.1109/ICCST63435.2025.11295691}
}
```

The paper was published in the 2025 IEEE International Carnahan Conference on Security Technology and received the Distinguished Conference Paper Award.

## Acknowledgements

This work was developed at the University of Texas at San Antonio.

This material is based upon work supported by the National Science Foundation under Grant No. 2332744.

The authors acknowledge the Flower project for providing the federated learning framework used to implement and execute this baseline.
