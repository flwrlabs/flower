# Flower Baselines


> [!NOTE] 
> We are changing the way we structure the Flower baselines. While we complete the transition to the new format, you can still find the existing baselines in the `flwr_baselines` directory. Currently, you can make use of baselines for [FedAvg](https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist), [FedOpt](https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization), and [LEAF-FEMNIST](https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist).


## Structure

Each baseline in this directory is fully self-contained in terms of source code in its own directory. In addition, each baseline uses its very own Python environment as designed by the contributors of such baseline in order to replicate the experiments in the paper. Each baseline directory contains the following structure:

```bash
baselines/<baseline-name>/
                ├── README.md
                ├── pyproject.toml
                └── <baseline-name>
                            └── *.py # several .py files
```

## Running the baselines

> [!NOTE]
> We are in the process of migrating all baselines to use `flwr run`. Those baselines that remain using the previous system (i.e. using [Poetry](https://python-poetry.org/), [Hydra](https://hydra.cc/) and [start_simulation](https://flower.ai/docs/framework/ref-api/flwr.simulation.start_simulation.html)) might require you to first setup `Poetry` and `pyenv` already on your machine, please take a look at the [Documentation](https://flower.ai/docs/baselines/how-to-use-baselines.html#setting-up-your-machine) for a guide on how to do so.

Each baseline is self-contained in its own directory. To run a baseline:

1. Cloning the flower repository

    ```bash
    git clone https://github.com/flwrlabs/flower.git && cd flower
    ```

2. Navigate inside the directory of the baseline you'd like to run.
3. Follow the `[Environment Setup]` instructions in the `README.md`.
4. Run the baseline as indicated in the `[Running the Experiments]` section in the `README.md` or in the `[Expected Results]` section to reproduce the experiments in the paper.


## Contributing a new baseline

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add an experiment to an existing baseline paper? Great, we really appreciate your contribution !!

> [!TIP]
> A more verbose version of these steps can be found in the [Flower Baselines documentation](https://flower.ai/docs/baselines/how-to-contribute-baselines.html).

The steps to follow are:

1. Create a new Python 3.12 environment and install Flower (`pip install flwr`)
1. Fork the Flower repo and clone it into your machine.
2. Navigate to the `baselines/` directory, from there and with your environment activated, run:

    ```shell
    # This will create a Flower App named `baseline`
    flwr new @flwrlabs/baseline
    ```
3. Rename the app from `baseline` to the name of the algorithm/paper you are implementing (e.g. `FedAwesome`).
4. Then, go inside your baseline directory and continue with the steps detailed in the `README.md`.
5. Once your code is ready, check that you have completed all the sections in the `README.md` and that, if a new environment is created, your baseline still runs (i.e. play the role of a person running the baseline you want to contribute).
6. Create a Pull Request (PR). Then, the process to merge your baseline into the Flower repo will begin!


Further resources:
* [GitHub docs: About forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
* [GitHub docs: Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
* [GitHub docs: Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

