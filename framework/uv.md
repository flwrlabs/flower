# uv

## Install

To reproduce a Poetry env with Python 3.10.19, all extras (`simulation`, `rest`) and all dependency groups (`dev`):

```
uv sync --python=3.10.19 --locked --all-extras --all-groups
```

`--locked` installs from `uv.lock` and fails if the lockfile is out-of-date. Without `--locked`, uv may re-resolve/update lock data during the operation.

## Compile Protos

```
uv run --no-sync --python=3.10.19 ./dev/protoc.sh
```

## Format

```
uv run --no-sync --python=3.10.19 ./dev/format.sh
```

## Test

```
uv run --no-sync --python=3.10.19 ./dev/test.sh
```

## Build

```
./dev/build-uv.sh
```

For comparison, build using Poetry:

```
./dev/build.sh
```
