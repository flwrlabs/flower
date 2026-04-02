---
tags: [mobile, vision, sdk]
dataset: [MNIST]
framework: [Swift]
---

# FLiOS - A Flower SDK for iOS Devices with Example

FLiOS is a sample application for testing and benchmarking the Swift implementation of Flower. The default scenario uses the MNIST dataset and the associated digit recognition model. The app includes the Swift package in `./src/swift` and allows extension for other benchmarking scenarios. The app guides the user through the steps of the machine learning process that would be executed in a normal production environment as a background task of the application. The app is therefore aimed at researchers and research institutions to test their hypotheses and perform performance analyses.

## Project Setup

### Installing Dependencies

Project dependencies (`flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Learning on iOS Clients

To start the server, write the following command in the terminal in the ios folder (with the conda environment created above):

```shell
python3 server.py
```

Open the FLiOS.xcodeproj with XCode, wait until the dependencies are fetched, then click build and run with iPhone 13 Pro Max as target, or you can deploy it in your own iOS device by connecting your Mac, where you run your XCode, and your iPhone.

When the iOS app runs, load both the training and test dataset first. Then enter the hostname and port of your server in the TextField provided. Finally press `Start` which will start the federated training.

## Adding further Scenarios

If you want to add more scenarios beyond MNIST, do the following:

- Open the _scenarios.ipynb_ notebook and adapt it to your needs based on the existing structure
- Open Xcode and add the dataset(s) and model to the sources of your project
- Add the dataset(s) to _Copy Bundle Resources_ in the Build Phases settings of the project
- Navigate to the _Constants.swift_ file and add your scenario so that it fits into the given structure
