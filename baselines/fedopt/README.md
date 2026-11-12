---
title: Adaptive Federated Optimization

url: https://arxiv.org/abs/2003.00295
labels: [label1, label2] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. system heterogeneity, image classification, asynchronous, weight sharing, cross-silo). Do not use ""
dataset: [CIFAR-10]
---

# Adaptive Federated Optimization


> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2003.00295](https://arxiv.org/abs/2003.00295)

**Authors:** Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan

**Abstract:** Federated learning is a distributed machine learning paradigm in which a large number of clients coordinate with a central server to learn a model without sharing their own training data. Standard federated optimization methods such as Federated Averaging (FedAvg) are often difficult to tune and exhibit unfavorable convergence behavior. In non-federated settings, adaptive optimization methods have had notable success in combating such issues. In this work, we propose federated versions of adaptive optimizers, including Adagrad, Adam, and Yogi, and analyze their convergence in the presence of heterogeneous data for general non-convex settings. Our results highlight the interplay between client heterogeneity and communication efficiency. We also perform extensive experiments on these methods and show that the use of adaptive optimizers can significantly improve the performance of federated learning.

## About this baseline

**What’s implemented:** Figure 1 results for CIFAR-10

**Datasets:** CIFAR-10

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

**Task:** Image Classification

**Model:** A ResNet-18 model without any modifications to it's original architecture with the exception of replacing BatchNormalization layers with [GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) layers.

**Dataset:** We follow the pre-processing of the dataset CIFAR-10 as indicated in Appendix C.1, resulting in `3x24x24` input images. Then, the entire dataset is split into the 500 clients, resulting in 100 trianing examples per client. The test set is used for centralised evaluation. The dataset can be split either IID or non-IID. The latter, following LDA with user-defined alpha (defaults to 0.1)

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

To construct the Python environment, simply run:

```bash
# Set directory to use python 3.10 (install with `pyenv install <version>` if you don't have it)
pyenv local 3.10.12

# Tell poetry to use python3.10
poetry env use 3.10.12

# Install
poetry install
```

## Running the Experiments

Assuming you have activated your environment via `poetry shell`, then:

```bash  
# By default runs with FedAvg for 4000 rounds
python -m fedopt.main

# You might want to reduce the number of rounds
python -m fedopt.main num_rounds=200

# You can change the strategy via the `strategy` input argument
python -m fedopt.main strategy=fedadam # choose any in {fedavg, fedadam, fedyogi, fedadagrad}
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
