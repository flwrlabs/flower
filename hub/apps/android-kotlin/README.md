---
tags: [mobile, vision, sdk]
dataset: [CIFAR-10]
framework: [Android, Kotlin, TensorFlowLite]
---

# Flower Android Client Example with Kotlin and TensorFlow Lite 2022

This example is similar to the Flower Android Example in Java:

> This example demonstrates a federated learning setup with Android Clients. The training on Android is done on a CIFAR10 dataset using TensorFlow Lite. The setup is as follows:
>
> - The CIFAR10 dataset is randomly split across 10 clients. Each Android client holds a local dataset of 5000 training examples and 1000 test examples.
> - The FL server runs in Python but all the clients run on Android.
> - We use a strategy called FedAvgAndroid for this example.
> - The strategy is vanilla FedAvg with a custom serialization and deserialization to handle the Bytebuffers sent from Android clients to Python server.

## Set up

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```sh
git clone --depth=1 https://github.com/flwrlabs/flower.git && mv flower/examples/android-kotlin . && rm -rf flower && cd android-kotlin
```

Download the training and testing data from <https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1> and extract them to `client/app/src/main/assets/data`.

Download the TFLite model from <https://github.com/FedCampus/dyn_flower_android_drf/files/11858642/cifar10.zip> to `client/app/src/main/assets/model/cifar10.tflite`.
Alternatively, see `gen_tflite/README.md` for information on how to convert the CIFAR10 models to a `.tflite` file.

### Install dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```sh
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```sh
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

<details>
<summary>Alternatively, with Pip.</summary>

```sh
python3 -m pip install -r requirements.txt
```

</details>

## Run the demo

Start the Flower server at `./`:

```sh
python3 server.py
```

<details>
<summary>Or without the "3" on windows.</summary>

```sh
python server.py
```

</details>

Install the app on *physical* Android devices and launch it.

<!-- TODO: APK. -->

*Note*: the highest tested JDK version the app supports is 16; it fails to build using JDK 19 on macOS.

In the user interface, fill in:

- Device number: a unique number among 1 ~ 10.
  This number is used to choose the partition of the training dataset.
- Server IP: an IPv4 address of the computer your backend server is running on. You can probably find it in your system network settings.
- Server port: 8080.

Push the first button and load the dataset. This may take a minute.

Push the second button and start the training.
