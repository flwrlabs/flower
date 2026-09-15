---
tags: [quickstart, linear regression, tabular]
dataset: [Synthetic]
framework: [C++]
---

# Flower Clients in C++

In this example you will train a linear model on synthetic data using external
C++ clients and a Python `ServerApp`.

This quickstart uses the Flower `grpc-rere` Fleet API used by recent Flower
versions. The C++ clients connect to a running SuperLink, receive train and
evaluate messages, serialize replies with Flower `RecordDict` payloads, and
push the resulting objects back through the SuperLink object store.

## Acknowledgements

Many thanks to the contributors to this code:

For This Version:

- Jiaxiang Geng (Duke Kunshan University, main contributor)
- Yiyi Lu (Duke Kunshan University, main contributor)
- Lunyu Zhao (Duke Kunshan University, main contributor)
- Bing Luo (Duke Kunshan University, director)

Edge-Intelligence-Lab Fork: https://github.com/Edge-Intelligence-Lab/flower-C--SDK

For Previous Version:
- Lekang Jiang (original author and main contributor)
- Francisco Jose Solis (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

## Install requirements

You'll need Python with `flwr>=1.31.0`, CMake, a C++17 compiler, gRPC C++,
protobuf, `protoc`, `grpc_cpp_plugin`, and OpenSSL.

Install the Python dependencies from this directory:

```bash
python -m pip install -e .
```

## Building the example

This example provides a `CMakeLists.txt` file to configure and build the C++
client.

From `examples/quickstart-cpp` inside a Flower checkout:

```bash
cmake -S . -B build
cmake --build build -j
```

If this directory is built outside the Flower repository, pass the Flower source
tree explicitly:

```bash
cmake -S . -B build -DFLWR_SOURCE_ROOT=/path/to/flower
cmake --build build -j
```

If gRPC/protobuf are installed in a custom prefix:

```bash
export CMAKE_PREFIX_PATH=/path/to/grpc-prefix
export PATH=/path/to/grpc-prefix/bin:$PATH
cmake -S . -B build \
  -DFLWR_SOURCE_ROOT=/path/to/flower \
  -DGRPC_CPP_PLUGIN_EXECUTABLE=/path/to/grpc_cpp_plugin
cmake --build build -j
```

## Run the `Flower SuperLink`, the two clients, and the `Flower ServerApp` in separate terminals

```bash
flower-superlink --insecure
```

```bash
build/flwr_client 0 127.0.0.1:9092
```

```bash
build/flwr_client 1 127.0.0.1:9092
```

```bash
flwr run . --stream
```

The `client.py` file only provides a placeholder `ClientApp` entry point for
Flower App metadata. The actual training clients are the external C++
`build/flwr_client` processes.
