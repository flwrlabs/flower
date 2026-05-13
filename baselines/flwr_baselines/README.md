# Flower Baselines

## Project Setup

Start by cloning the Flower Baselines project. We prepared a single-line command that you can copy into your shell which will clone the project for you:

```shell
git clone --depth=1 https://github.com/flwrlabs/flower.git && cd baselines
```

Project dependencies (such as `flwr` and `torch`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.
Once inside your Python virtual environment containing `poetry`, go ahead and run:

```shell
poetry install
```

Poetry will install all your dependencies in a newly created virtual environment if you haven't create and/or activated one. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr_baselines"
```
