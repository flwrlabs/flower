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

You'll need Python 3.11 or newer, Flower **\<=1.37.0**, CMake, a C++17
compiler, gRPC C++, protobuf, `protoc`, `grpc_cpp_plugin`, and OpenSSL.

Install the Python dependencies from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . "flwr==1.37.0"
```

To use Flower 1.36.0 instead, replace `flwr==1.37.0` with `flwr==1.36.0`.
The Flower dependency has only an upper bound, `flwr<=1.37.0`; no minimum
Flower version is imposed by the package metadata. The explicit version
selects one of the releases validated by this example. Accepting a version
during dependency resolution does not establish runtime compatibility.

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

This quickstart is tested with **Flower 1.36.0 and 1.37.0**. Each version has
passed a real training run with two external C++ clients and a Python ServerApp
for three rounds, as well as the CTest regressions. This does not imply
compatibility with every older or future Flower release.

The C++ clients use the `grpc-rere` Fleet API on `127.0.0.1:9092` in both
versions. The CLI's Control API connection differs:

| Flower version | Control API used by `flwr run` | Default local address |
| -------------- | ------------------------------ | --------------------- |
| 1.36.0         | gRPC                           | `127.0.0.1:9093`      |
| 1.37.0         | HTTP                           | `127.0.0.1:8000`      |

Add the following named connections to the
[Flower config](https://flower.ai/docs/framework/ref-flower-configuration.html)
(normally `~/.flwr/config.toml`). Merge them with any existing configuration
rather than replacing other connections. Create the file if it does not exist.

```toml
# ~/.flwr/config.toml
[superlink.local-136]
address = "127.0.0.1:9093"
insecure = true

[superlink.local-137]
address = "127.0.0.1:8000"
insecure = true
```

Use the connection matching the installed Flower version, as shown below.
For an isolated configuration, set `FLWR_HOME` to the same separate directory
in every terminal running Flower commands and put `config.toml` there.

This example no longer embeds the legacy `[tool.flwr.federations]` block in
`pyproject.toml`: both versions migrate that block to the Flower config, but
one Control API address is not suitable for both versions. If an earlier
checkout migrated a `local-deployment` connection, select the new versioned
connection explicitly. Do not re-add the legacy block, which can overwrite
connection settings during migration.

The protobuf sources are generated from `${FLWR_SOURCE_ROOT}/framework/proto`
at build time. The C++ transport uses the legacy no-session-ID object-upload
path accepted by Flower 1.36.0 and 1.37.0. This example is not a full production
C++ SDK.

## Run the regression tests

After installing the Python dependencies and building, run:

```bash
ctest --test-dir build --output-on-failure
```

CTest runs native model-training and tensor-order tests, Python serialization
tests, and cross-language tests that launch the real C++ transport against a loopback gRPC
Fleet fixture. The latter verifies that Python receives 1, 2, 9, 10, 11, 12, 21,
and 100 tensors in their original order, and that fatal registration/polling
errors cause a nonzero client exit status. The fixture uses ephemeral ports and
does not contact an existing SuperLink. It is separate from the real training
run below.

The training regression checks that the sampling indices contain each row only
once, small datasets use a correctly sized batch, empty datasets are rejected,
and a high-leverage first sample does not cause training to diverge.

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

For Flower 1.37.0:

```bash
flwr run . local-137 --stream
```

For Flower 1.36.0:

```bash
flwr run . local-136 --stream
```

The `client.py` file only provides a placeholder `ClientApp` entry point for
Flower App metadata. The actual training clients are the external C++
`build/flwr_client` processes.
