# uv

## Install

To reproduce a development env with Python 3.12.13, all extras (`simulation`) and all dependency groups (`dev`):

```
uv sync --python=3.12.13 --locked --all-extras --all-groups
```

`--locked` installs from `uv.lock` and fails if the lockfile is out-of-date. Without `--locked`, uv may re-resolve/update lock data during the operation.

## Compile Protos

```
uv run --no-sync --python=3.12.13 ./dev/protoc.sh
```

## Format

```
uv run --no-sync --python=3.12.13 ./dev/format.sh
```

## Test

```
uv run --no-sync --python=3.12.13 ./dev/test.sh
```

## Build

```
uv run --no-sync --python=3.12.13 ./dev/build.sh
```
