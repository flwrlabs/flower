# Dev Tooling (`devtool`) with `uv`

This directory provides developer tooling used across the repository (for example
formatting, docs helpers, and repository checks).

`devtool` uses `uv` for dependency management and command execution.
The Flower `framework` project remains Poetry-based.

## Prerequisites

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Use Python 3.10+ (as defined in `pyproject.toml`)

## Quick Start

From the repository root:

```bash
cd dev
uv sync --frozen
```

This creates/updates `dev/.venv` from `uv.lock`.

## Run `devtool` Commands

Run modules through `uv run`:

```bash
cd dev
uv run python -m devtool.check_pr_title "feat(framework): Add test"
uv run python -m devtool.init_py_check ../framework/py/flwr
uv run python -m devtool.init_py_fix ../framework/py/flwr
uv run python -m devtool.check_copyright ../framework/py/flwr
uv run python -m devtool.fix_copyright ../framework/py/flwr
uv run python -m devtool.update_html_themes
uv run python -m devtool.build_example_docs
uv run python -m devtool.protoc --project-dir ../framework
```

## Run Existing Dev Scripts with `uv`

Most scripts in this directory call Python tools directly (`python -m ...`), so run
them via `uv run` to ensure they use the `devtool` environment:

```bash
cd dev
uv run ./test-devtool.sh
uv run ./format-devtool.sh
uv run ./test.sh
uv run ./format.sh
```

The `test-devtool.sh` and `format-devtool.sh` scripts target the `devtool`
package, including its unit tests. The existing `test.sh` and `format.sh` scripts
still target examples and benchmarks.

## Updating Dependencies

1. Edit `pyproject.toml`.
2. Re-lock dependencies:

```bash
cd dev
uv lock
```

3. Validate with:

```bash
uv sync --frozen
```

4. Commit both:
- `pyproject.toml`
- `uv.lock`

## CI Notes

- CI jobs that use `devtool` should run `uv sync --frozen` and execute commands with
  `uv run`.
- CI jobs that install `framework` still use Poetry.
