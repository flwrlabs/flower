# FedAvg on MNIST Baseline

This baseline implements the Federated Averaging (FedAvg) algorithm on the MNIST dataset using PyTorch and the Flower framework.

## Overview

- **Model:** A simple CNN with two convolutional layers, two fully connected layers.
- **Dataset:** MNIST (60K training images, 10K test images).
- **Federation:** The training set is partitioned IID among 10 clients (configurable).
- **Training Hyperparameters:** 5 local epochs per round, learning rate = 0.1, running on CPU.
- **Strategy:** FedAvg (all clients participate each round). A centralized evaluation is performed on the MNIST test set after each round.

## Running the Baseline

1. **Install Dependencies:**  
   Navigate to this directory and install with:
   ```bash
   pip install -e .
