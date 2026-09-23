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

You'll need Python 3.11 or newer, Flower **1.36.0**, CMake, a C++17 compiler, gRPC C++,
protobuf, `protoc`, `grpc_cpp_plugin`, and OpenSSL.

Install the Python dependencies from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
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

## Flower version compatibility

The Python dependency is pinned to **Flower 1.36.0**, the target validated for
this quickstart. The real training run uses two external C++ clients and a Python
ServerApp for three rounds. This does not imply compatibility with every older
or future Flower release.

On 1.36.0, the C++ clients use the `grpc-rere` Fleet API on `127.0.0.1:9092`,
and `flwr run` uses the gRPC Control API on `127.0.0.1:9093`.

The first `flwr run` migrates `[tool.flwr.federations]` from `pyproject.toml`
into the Flower config (normally `~/.flwr/config.toml`) and comments the old
block out. This migration also happens on **1.36.0**. The address supplied by
this example is already correct for that version. If an earlier run left a
different address in the Flower config, check the named connection there:

```toml
# ~/.flwr/config.toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

Do not re-add the legacy block after migration: that triggers another migration
and may overwrite the named connection. For an isolated configuration, set
`FLWR_HOME` to a separate directory in every terminal running Flower commands.

The protobuf sources are generated from `${FLWR_SOURCE_ROOT}/framework/proto`
at build time. The C++ transport uses the legacy no-session-ID object-upload
path accepted by Flower 1.36.0. This example is not a full production C++ SDK.

## Run the regression tests

After installing the Python dependencies and building, run:

```bash
ctest --test-dir build --output-on-failure
```

CTest runs the native tensor-order test, the Python serialization tests, and
cross-language tests that launch the real C++ transport against a loopback gRPC
Fleet fixture. The latter verifies that Python receives 1, 2, 9, 10, 11, 12, 21,
and 100 tensors in their original order, and that fatal registration/polling
errors cause a nonzero client exit status. The fixture uses ephemeral ports and
does not contact an existing SuperLink. It is separate from the real training
run below.

CMake uses the active Python environment. If necessary, select it explicitly
with `-DPython3_EXECUTABLE=/path/to/venv/bin/python`. To build only the example
client without the test targets, configure with `-DBUILD_TESTING=OFF`.

## Run the `Flower SuperLink`, the two clients, and the `Flower ServerApp` in separate terminals

Activate the same Python environment in each terminal running a Python command.

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
