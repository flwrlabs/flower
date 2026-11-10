# Flower Simulation of Training Weldon on Camelyon

This introductory example uses the simulation capabilities of Flower to 
simulate federated Weldon model training on Camelyon dataset using a single machine.

## Running the example

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/weldon-on-camelyon . && rm -rf flower && cd weldon-on-camelyon
```

This will create a new directory called `weldon-on-camelyon` containing the following files:

```
-- README.md           <- Your're reading this right now
-- sim.py              <- Simulation code
-- utils.py            <- Auxiliary functions for this example
-- model.py            <- Model achitecture
-- data_managers.py    <- Data loader
-- dataset_manager.py  <- Data pre-processing code
-- requirements.txt    <- Example dependencies
```

### Environment Setup

#### Python version
python = ">=3.8,<3.11"


#### Installing dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `requirements.txt`. We recommend [pip](https://pip.pypa.io/en/latest/development/) to install those dependencies and manage your virtual environment, 
but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

### Run Federated Learning Example

```bash
# After activating your environment
# and then run the example
python sim.py --data-path=./data --pool-size=2 --num-rounds=10 --num-gpus=0.5 --num-cpus=2
```
This command will run an FL simulation with 2 clients for 10 rounds. 
Note that this will assign 2xCPUs and 50% of the GPU's VRAM to each client.
This means that you can have 2 concurrent clients on each GPU.

### Expected Results

```shell
INFO flwr 2024-02-20 21:20:07,210 | app.py:228 | app_fit: metrics_distributed {'auc': [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0)], 'accuracy': [(1, 0.75), (2, 0.5), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0)]}

INFO flwr 2024-02-20 21:20:07,210 | app.py:230 | app_fit: metrics_centralized {'auc': [(0, 0.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0)], 'accuracy': [(0, 0.0), (1, 0.75), (2, 0.5), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0)]}
```

After finishing the training, the first line shows the **AUC** and **accuracy** values for distributed (on-client) evaluation,
while the second line is for centralised evaluation.

Don't worry about the values of the results. We just use sample dataset, so the model is easy to over-fit. 
