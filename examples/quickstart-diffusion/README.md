---
tags: [quickstart, diffusion, vision, federated-learning]
dataset: [Oxford-Flowers]
framework: [diffusers, peft, flower]
---


# Federated Diffusion Model Training with Flower (Quickstart Example)

This example demonstrates how to train a **Diffusion Model** (based on [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)) in a **Federated Learning (FL)** environment using [Flower](https://flower.ai/).  
The training uses **Low-Rank Adaptation (LoRA)** to enable lightweight fine-tuning of a diffusion model in a distributed setup, even with limited compute resources.
The example uses the Oxford Flowers dataset, a collection of RGB flower images commonly used for training image-generation models.

Prompt:
A realistic image of a blooming yellow daffodil in natural sunlight

The model was trained using 400 images for training, 80 images for evaluation, and fine-tuned for 5 federated rounds.
Below is a sample output image generated using the final LoRA-adapted model.

![](_static/federated_diffusion_sample.png)
---
## Overview

In this example:

- Each client fine-tunes **only the LoRA parameters** of the Stable Diffusion UNet.
- The **Oxford-Flowers** dataset is partitioned among multiple clients using Flower Datasets.
- The server performs **FedAvg aggregation** on the LoRA weights after each round.
- After all rounds, the aggregated LoRA adapter is saved into `final_lora_model/` and can be merged with the base model for image generation.

This provides a clean example of how **federated diffusion fine-tuning** can be performed using Diffusers + PEFT + Flower.

---
## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-diffusion . \
		&& rm -rf _tmp && cd quickstart-diffusion
```

After cloning, your directory will look like this:

```shell
quickstart-diffusion
├── diffusion_example
│   ├── __init__.py
│   ├── client_app.py        # Defines your ClientApp logic
│   ├── server_app.py        # Defines the ServerApp and strategy
│   └── task.py              # Model setup, data loading, and training functions
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file


```
### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `diffusion_example` package.

```bash
pip install -e .
```
## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

Run the project in the `local-simulation-gpu` federation that allocates both CPU and GPU resources to each `ClientApp`.  
Since diffusion models are memory-intensive, we recommend running **one client at a time** or limiting parallelism to **1–2 clients per GPU** (each client may use 3–6 GB of VRAM depending on image size and model configuration).  
You can modify the level of parallelism or memory allocation in the `client-resources` section of your `pyproject.toml` file to fit your system’s GPU capacity.


```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config "num-server-rounds=5 fraction-train=0.1"
```

### Result output

Example of training step results for each client and corresponding server logs:


```bash
Loading project configuration... 
Success
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 19.94it/s]
Trainable parameters: 797184

 Starting federated diffusion training for 5 rounds...
 Using base model: runwayml/stable-diffusion-v1-5
 Training LoRA parameters only (256 layers)
INFO :      Starting FedAvg strategy:
INFO :          ├── Number of rounds: 5
INFO :          ├── ArrayRecord (3.07 MB)
INFO :          ├── ConfigRecord (train): (empty!)
INFO :          ├── ConfigRecord (evaluate): (empty!)
INFO :          ├──> Sampling:
INFO :          │       ├──Fraction: train (1.00) | evaluate ( 1.00)
INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
INFO :          │       └──Minimum available nodes: 2
INFO :          └──> Keys in records:
INFO :                  ├── Weighted by: 'num-examples'
INFO :                  ├── ArrayRecord key: 'arrays'
INFO :                  └── ConfigRecord key: 'config'
INFO :      
INFO :      
INFO :      [ROUND 1/5]
INFO :      configure_train: Sampled 2 nodes (out of 2)
(ClientAppActor pid=73271) /opt/homebrew/Caskroom/miniconda/base/envs/diffusion-env/lib/python3.9/site-packages/flwr_datasets/utils.py:109: UserWarning: The currently tested dataset are ['mnist', 'ylecun/mnist', 'cifar10', 'uoft-cs/cifar10', 'fashion_mnist', 'zalando-datasets/fashion_mnist', 'sasha/dog-food', 'zh-plus/tiny-imagenet', 'scikit-learn/adult-census-income', 'cifar100', 'uoft-cs/cifar100', 'svhn', 'ufldl-stanford/svhn', 'sentiment140', 'stanfordnlp/sentiment140', 'speech_commands', 'LIUM/tedlium', 'flwrlabs/femnist', 'flwrlabs/ucf101', 'flwrlabs/ambient-acoustic-context', 'jlh/uci-mushrooms', 'Mike0307/MNIST-M', 'flwrlabs/usps', 'scikit-learn/iris', 'flwrlabs/pacs', 'flwrlabs/cinic10', 'flwrlabs/caltech101', 'flwrlabs/office-home', 'flwrlabs/fed-isic2019']. Given: nkirschi/oxford-flowers.
(ClientAppActor pid=73271) Partition 0: 400 training samples, 80 test samples
Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]
Loading pipeline components...:  50%|█████     | 3/6 [00:00<00:00, 26.32it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:00<00:00, 31.15it/s]
(ClientAppActor pid=73271) Trainable parameters: 797184
Loading pipeline components...: 100%|██████████| 6/6 [00:00<00:00, 35.03it/s]
(ClientAppActor pid=73270) Partition 1: 400 training samples, 80 test samples
(ClientAppActor pid=73270) Trainable parameters: 797184

INFO :      aggregate_train: Received 2 results and 0 failures
INFO :          └──> Aggregated MetricRecord: {'loss': 0.267623713389039}
INFO :      configure_evaluate: Sampled 2 nodes (out of 2)
(ClientAppActor pid=73271) Partition 1: 400 training samples, 80 test samples
Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  33%|███▎      | 2/6 [00:00<00:00, 17.51it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:00<00:00, 23.65it/s]
(ClientAppActor pid=73271) Trainable parameters: 797184
(ClientAppActor pid=73270) Partition 0: 400 training samples, 80 test samples

INFO :      aggregate_evaluate: Received 2 results and 0 failures
INFO :          └──> Aggregated MetricRecord: {'loss': 0.3734127514064312}
INFO :      
INFO :      [ROUND 2/5]
.
.
INFO :      [ROUND 3/5]
.
.
INFO :      [ROUND 4/5]
.
.
INFO :      [ROUND 5/5]
.
.
INFO :      
INFO :      Strategy execution finished in 35154.13s
INFO :      
INFO :      Final results:
INFO :      
INFO :          Global Arrays:
INFO :                  ArrayRecord (3.072 MB)
INFO :          
INFO :          Aggregated ClientApp-side Train Metrics:
INFO :          { 1: {'loss': '2.6762e-01'},
INFO :            2: {'loss': '2.4918e-01'},
INFO :            3: {'loss': '2.5092e-01'},
INFO :            4: {'loss': '2.5719e-01'},
INFO :            5: {'loss': '2.4323e-01'}}
INFO :          
INFO :          Aggregated ClientApp-side Evaluate Metrics:
INFO :          { 1: {'loss': '3.7341e-01'},
INFO :            2: {'loss': '3.2973e-01'},
INFO :            3: {'loss': '3.5631e-01'},
INFO :            4: {'loss': '3.6203e-01'},
INFO :            5: {'loss': '3.3863e-01'}}
INFO :          
INFO :          ServerApp-side Evaluate Metrics:
INFO :          {}
INFO :      
Saving final federated LoRA adapter...
Saved final LoRA model at: final_lora_model


