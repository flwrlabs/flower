---
title: Preservation of the Global Knowledge by Not-True Distillation in Federated Learning
url: https://openreview.net/forum?id=qw3MZb1Juo
labels: [deep learning, federated learning, continual learning, knowledge distillation]
dataset: [mnist]
---

# FedNTD: Preservation of the Global Knowledge by Not-True Distillation in Federated Learning

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** <https://openreview.net/forum?id=qw3MZb1Juo>

**Authors:** Gihun Lee, Minchan Jeong, Yongjin Shin, Sangmin Bae, Se-Young Yun

**Abstract:** In federated learning, a strong global model is collaboratively learned by aggregating clients' locally trained models. Although this precludes the need to access clients' data directly, the global model's convergence often suffers from data heterogeneity. This study starts from an analogy to continual learning and suggests that forgetting could be the bottleneck of federated learning. We observe that the global model forgets the knowledge from previous rounds, and the local training induces forgetting the knowledge outside of the local distribution. Based on our findings, we hypothesize that tackling down forgetting will relieve the data heterogeneity problem. To this end, we propose a novel and effective algorithm, Federated Not-True Distillation (FedNTD), which preserves the global perspective on locally available data only for the not-true classes. In the experiments, FedNTD shows state-of-the-art performance on various setups without compromising data privacy or incurring additional communication costs.

## About this baseline

**What’s implemented:** The code in this directory replicates the experiments in Preservation of the Global Knowledge by Not-True Distillation in Federated Learning (Lee et al., 2022) for MNIST, which proposed the FedNTD algorithm.

**Datasets:** MNIST from PyTorch's Torchvision

**Hardware Setup:** These experiments were run on a laptop machine with 16 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: we install PyTorch with GPU support but by default, the entire experiment runs on CPU-only mode.

**Contributors:** [Sarang S](https://github.com/eigengravy)

## Experimental Setup

**Task:** :warning: **what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them**

**Model:** :warning: **provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed.**

**Dataset:** :warning: **Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table.**

**Training Hyperparameters:** :warning: **Include a table with all the main hyperparameters in your baseline. Please show them with their default value.**

## Environment Setup

:warning: *The Python environment for all baselines should follow these guidelines in the `EXTENDED_README`. Specify the steps to create and activate your environment. If there are any external system-wide requirements, please include instructions for them too. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step).*

## Running the Experiments

To run this FedNTD with MNIST baseline, first ensure you have activated your Poetry environment (execute poetry shell from this directory), then:

```bash  
poetry run python -m fedntd.main
```

## Expected Results

:warning: *Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments.*

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```

## Acknowledgements

Heavily inspired from the official implementation at [Lee-Gihun/FedNTD](https://github.com/Lee-Gihun/FedNTD).
