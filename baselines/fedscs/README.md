---
title: Robust Federated Learning via Stable Cosine Similarity
url: https://doi.org/10.1109/ICCST63435.2025.11295691
labels: [federated learning, robust aggregation, non-iid, cosine similarity, security, cifar-10]
dataset: [CIFAR-10]
---

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

| Hyperparameter           | Default |
| ------------------------ | ------: |
| Number of clients        |      10 |
| Number of server rounds  |       2 |
| Client training fraction |     1.0 |
| Client evaluation fraction | 1.0 |
| Local epochs             |       2 |
| Batch size               |      32 |
| Learning rate            |    0.01 |
| Optimizer               |     SGD |
| Loss function            | Cross-Entropy |
| Dataset                  | CIFAR-10 |
| Dirichlet α              |     0.3 |
| Random seed              |      42 |
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
