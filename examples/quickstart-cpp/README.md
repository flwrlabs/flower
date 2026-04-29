---
tags: [quickstart, linear regression, tabular]
dataset: [Synthetic]
framework: [C++]
---

# Flower Clients in C++

In this example you will train a linear model on synthetic data using C++ clients.

## Acknowledgements

Many thanks to the original contributors to this code:

- Lekang Jiang (original author and main contributor)
- Francisco José Solís (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

## Option 1: Run with Docker (recommended)

The easiest way to test the C++ client is with Docker Compose, which builds the
client and runs the full federated learning setup automatically.

```bash
cd examples/quickstart-cpp
docker compose up --build
```

This starts the SuperLink, two C++ SuperNode clients, and the Python ServerApp.

To clean up:

```bash
docker compose down
```

## Option 2: Run locally

### Install requirements

You'll need CMake, a C++17 compiler, and Python with `flwr` and `numpy` installed.

### Building the example

```bash
cmake -S . -B build -DUSE_LOCAL_FLWR=ON
cmake --build build
```

### Run the SuperLink, two clients, and the ServerApp in separate terminals

```bash
flwr-superlink --insecure
```

```bash
build/flower-supernode 0 127.0.0.1:9092
```

```bash
build/flower-supernode 1 127.0.0.1:9092
```

```bash
flower-server-app server:app --insecure
```
