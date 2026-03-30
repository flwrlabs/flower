---
tags: [flowertune, llm, finetuning, lora, code]
dataset: [code-alpaca-20k]
framework: [torch, transformers, peft]
---

# FlowerTune LLM on Code Dataset

This directory conducts federated instruction tuning with a pretrained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model on a [Code dataset](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of Code challenge.

## Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/flowertune-llm-code
```

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

> **Tip:** Learn how to configure your `pyproject.toml` file for Flower apps in [this guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Experimental setup

The dataset is divided into 10 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.2) of the total nodes to participate in each round, for a total of `200` rounds. All the Flower App settings are defined in `pyproject.toml`.

Before proceeding you need to create a new SuperLink connection and define 10 virtual SuperNodes. To do this, let's first locate the Flower Configuration file and then edit it.

Locate the Flower Configuration file:

```shell
flwr config list
```

```console
# Example output:
Flower Config file: /path/to/your/.flwr/config.toml
SuperLink connections:
 supergrid
 local (default)
```

Add a new connection named `flowertune` and make it the default.

```TOML
[superlink.flowertune]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 6
options.backend.client-resources.num-gpus = 1.0
```

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard). Additionally, the number of supernodes (i.e. `options.num-supernodes`) must be 10.

## Running the challenge

First make sure that you have got the access to [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model with your Hugging-Face account. You can request access directly from the Hugging-Face website.
Then, follow the instruction [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account. Note you only need to complete this stage once in your development machine:

```bash
hf auth login
```

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## VRAM consumption

We use Mistral-7B model with 4-bit quantization as default. The estimated VRAM consumption per client for each challenge is shown below:

| Challenges | GeneralNLP |  Finance  |  Medical  |   Code    |
| :--------: | :--------: | :-------: | :-------: | :-------: |
|    VRAM    | ~25.50 GB  | ~17.30 GB | ~22.80 GB | ~17.40 GB |

You can adjust the CPU/GPU resources you assign to each of the clients based on your device, which are specified with `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` in your `flowertune` connection in your `config.toml`.

## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
