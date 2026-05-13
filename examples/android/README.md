---
tags: [mobile, vision, sdk]
dataset: [CIFAR-10]
framework: [Android, Java, TensorFlowLite]
---

# Flower Android Example (TensorFlowLite)

This example demonstrates a federated learning setup with Android clients in a background thread. The training on Android is done on a CIFAR10 dataset using TensorFlow Lite. The setup is as follows:

- The CIFAR10 dataset is randomly split across 10 clients. Each Android client holds a local dataset of 5000 training examples and 1000 test examples.
- The FL server runs in Python but all the clients run on Android.
- We use a strategy called FedAvgAndroid for this example.
- The strategy is vanilla FedAvg with a custom serialization and deserialization to handle the Bytebuffers sent from Android clients to Python server.

The background thread is established via the `WorkManager` library of Android, thus, it will run comfortably on Android Versions from 8 to 13.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/flwrlabs/flower.git && mv flower/examples/android . && rm -rf flower && cd android
```

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning on Android Devices

The included `run.sh` will start the Flower server (using `server.py`). You can simply start it in a terminal as follows:

```shell
poetry run ./run.sh
```

Download and install the `flwr_android_client.apk` on each Android device/emulator. The server currently expects a minimum of 4 Android clients, but it can be changed in the `server.py`.

When the Android app runs, add the client ID (between 1-10), the IP and port of your server, and press `Start`. This will load the local CIFAR10 dataset in memory, establish connection with the server, and start the federated training. To abort the federated learning process, press `Stop`. You can clear and refresh the log messages by pressing `Clear` and `Refresh` buttons respectively.
