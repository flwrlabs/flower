---
tags: [quickstart, diffusion, vision, federated-learning]
dataset: [Oxford-Flowers]
framework: [diffusers, peft, flower]
---


# Federated Diffusion Model Training with Flower using Secure Aggregation (Quickstart Example)

This example demonstrates how to train a **Diffusion Model** (based on [Segmind Tiny-SD](https://huggingface.co/segmind/tiny-sd)) in a **Federated Learning (FL)** environment using [Flower](https://flower.ai/).  
The training uses **Low-Rank Adaptation (LoRA)** to enable lightweight fine-tuning of a diffusion model in a distributed setup, even with limited compute resources.
The example uses the Oxford Flowers dataset, a collection of RGB flower images commonly used for training image-generation models.

In this example, the diffusion model is fine-tuned using a Secure Aggregation technique within a federated learning setup to enhance data privacy. Secure Aggregation ensures that individual client updates (model parameters or gradients) are encrypted before being sent to the central server. As a result, the server can only access the aggregated model updates and cannot inspect any single client’s contribution. This prevents sensitive training information from being exposed, even to the server coordinating the learning process. By combining diffusion model fine-tuning with Secure Aggregation, the system enables collaborative model improvement across multiple clients while maintaining strong privacy guarantees for local datasets.

```aiignore
                 ┌───────────────────────────┐
                 │        Central Server      │
                 │  (Aggregation Only)        │
                 └─────────────┬─────────────┘
                               │
                     Encrypted Model Updates
                               │
        ───────────────────────┼───────────────────────
                               │
        ▼                                              ▼
┌──────────────────┐                          ┌──────────────────┐
│     Client 1     │                          │     Client 2     │
│ Oxford Flowers   │                          │ Oxford Flowers   │
│ Local Dataset    │                          │ Local Dataset    │
│                  │                          │                  │
│ Tiny-SD + LoRA   │                          │ Tiny-SD + LoRA   │
│ Fine-Tuning      │                          │ Fine-Tuning      │
│                  │                          │                  │
│ 🔒 Secure Masking │                          │ 🔒 Secure Masking │
└─────────┬────────┘                          └─────────┬────────┘
          │                                             │
          └────────── Encrypted Gradients/Weights ──────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Secure Aggregation Module │
                 │  (No Raw Client Access)   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 Updated Global Diffusion Model

```


### Using the prompt 'A wild garden flower with soft natural lighting', images were generated under all training configurations to visually compare model performance.

The model was fine-tuned on the Oxford Flowers dataset over 10 federated learning rounds. Below is a sample image generated using the final LoRA-adapted model.

![](_static/without_fine_tun.png)
---
#### Image generated using the original model without fine-tuning.

![](_static/normal_fine_tun.png)
---
#### Image generated after fine-tuning the model.


![](_static/secAgg_fine_tun.png)
---
#### Sample image produced by the model after privacy-preserving fine-tuning with Secure Aggregation.


## Overview
In this example:

- Each client fine-tunes **only the LoRA parameters** of the Stable Diffusion UNet locally on its private data.
- The **Oxford-Flowers** dataset is partitioned among multiple clients using Flower Datasets.
- Before transmission, client model updates are **masked/encrypted using Secure Aggregation**, ensuring the server cannot access any individual client’s weights.
- The server performs **privacy-preserving FedAvg aggregation** on the secured LoRA weights after each round, operating only on aggregated updates.
- After all rounds, the aggregated LoRA adapter is saved into `final_lora_model/` and can be merged with the base model for image generation.

This demonstrates how federated diffusion fine-tuning with Secure Aggregation can be achieved using Diffusers + PEFT + Flower while protecting client-level training information.


---
## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-diffusion-SecAggregation . \
		&& rm -rf _tmp && cd quickstart-diffusion-SecAggregation
```

After cloning, your directory will look like this:

```shell
quickstart-diffusion-SecAggregation
├── diffusionSecAgg
│   ├── __init__.py
│   ├── client_app.py        # Defines your ClientApp logic
│   ├── server_app.py        # Defines the ServerApp and strategy
│   └── task.py              # Model setup, data loading, and training functions
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file


```
### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `diffusionSecAgg` package.

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


```
Loading project configuration... 
Success
Loading pipeline components...: 100%|█████████████| 5/5 [00:02<00:00,  1.72it/s]
Trainable parameters: 648192
INFO :       Starting federated diffusion training for 3 rounds...
INFO :       Using base model: segmind/tiny-sd
INFO :       Training LoRA parameters only (108 layers)
INFO :      [INIT]
INFO :      Using initial global parameters provided by strategy
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :      
INFO :      [ROUND 1]
INFO :      Secure aggregation commencing.
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]
Loading pipeline components...:  40%|████      | 2/5 [00:00<00:01,  2.90it/s]
Loading pipeline components...:  60%|██████    | 3/5 [00:01<00:00,  2.95it/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.37it/s]
Map:  84%|████████▎ | 3000/3584 [00:14<00:02, 209.40 examples/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.43it/s]
(ClientAppActor pid=550) Trainable parameters: 648192
(ClientAppActor pid=550) Average loss: 0.265599
(ClientAppActor pid=553) Partition 1: 2867 training samples, 717 test samples
(ClientAppActor pid=553) Trainable parameters: 648192
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      Secure aggregation completed.
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
(ClientAppActor pid=550) Partition 0: 2868 training samples, 717 test samples
Map: 100%|██████████| 3584/3584 [00:17<00:00, 203.85 examples/s] [repeated 2x across cluster]
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  60%|██████    | 3/5 [00:01<00:01,  1.62it/s] [repeated 3x across cluster]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.10it/s]
(ClientAppActor pid=553) Trainable parameters: 648192
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :      
INFO :      [ROUND 2]
INFO :      Secure aggregation commencing.
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
(ClientAppActor pid=553) Average loss: 0.270134
(ClientAppActor pid=550) Partition 0: 2868 training samples, 717 test samples [repeated 2x across cluster]
(ClientAppActor pid=550) Trainable parameters: 648192
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  60%|██████    | 3/5 [00:02<00:01,  1.42it/s] [repeated 5x across cluster]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  1.99it/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:01<00:00,  2.93it/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.10it/s]
(ClientAppActor pid=553) Trainable parameters: 648192
(ClientAppActor pid=553) Average loss: 0.274635
(ClientAppActor pid=553) Partition 1: 2867 training samples, 717 test samples
(ClientAppActor pid=550) Trainable parameters: 648192
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      Secure aggregation completed.
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
(ClientAppActor pid=550) Partition 1: 2867 training samples, 717 test samples
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  40%|████      | 2/5 [00:01<00:02,  1.48it/s] [repeated 4x across cluster]
Loading pipeline components...: 100%|██████████| 5/5 [00:01<00:00,  2.83it/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.04it/s]
(ClientAppActor pid=553) Trainable parameters: 648192
(ClientAppActor pid=550) Average loss: 0.286401
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :      
INFO :      [ROUND 3]
INFO :      Secure aggregation commencing.
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
(ClientAppActor pid=550) Partition 0: 2868 training samples, 717 test samples [repeated 2x across cluster]
(ClientAppActor pid=550) Trainable parameters: 648192
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  60%|██████    | 3/5 [00:01<00:01,  1.98it/s] [repeated 5x across cluster]
Loading pipeline components...: 100%|██████████| 5/5 [00:01<00:00,  2.61it/s]
Loading pipeline components...: 100%|██████████| 5/5 [00:02<00:00,  2.29it/s]
(ClientAppActor pid=553) Trainable parameters: 648192
(ClientAppActor pid=553) Average loss: 0.261073
(ClientAppActor pid=553) Partition 1: 2867 training samples, 717 test samples
(ClientAppActor pid=550) Trainable parameters: 648192
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      Secure aggregation completed.
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
(ClientAppActor pid=550) Partition 1: 2867 training samples, 717 test samples
Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s] [repeated 2x across cluster]
Loading pipeline components...:  60%|██████    | 3/5 [00:01<00:01,  1.81it/s] [repeated 5x across cluster]
Loading pipeline components...: 100%|██████████| 5/5 [00:01<00:00,  2.77it/s]
(ClientAppActor pid=553) Trainable parameters: 648192
(ClientAppActor pid=550) Average loss: 0.272471
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 1606.31s
INFO :      History (loss, distributed):
INFO :      		round 1: 0.4985333716289865
INFO :      		round 2: 0.5122772523926364
INFO :      		round 3: 0.5009203565617403
INFO :      	History (metrics, distributed, fit):
INFO :      	{'loss': [(1, 0.2678664769740928),
INFO :      	          (2, 0.2805175630913494),
INFO :      	          (3, 0.2667721105652447)]}
INFO :      	History (metrics, distributed, evaluate):
INFO :      	{'psnr': [(1, 3.1460328712231584),
INFO :      	          (2, 3.0280961100425987),
INFO :      	          (3, 3.11378640178591)]}
INFO :      
Loading pipeline components...: 100%|█████████████| 5/5 [00:01<00:00,  3.22it/s]
100%|███████████████████████████████████████████| 30/30 [00:10<00:00,  2.90it/s]
Loading pipeline components...: 100%|█████████████| 5/5 [00:01<00:00,  3.26it/s]
100%|███████████████████████████████████████████| 30/30 [00:10<00:00,  2.89it/s]
```
