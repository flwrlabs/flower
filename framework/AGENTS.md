# AGENTS.md

These instructions apply to the `framework/` subtree. Also follow the repository
root `AGENTS.md`.

## Project shape

- `py/flwr/` is the Python package. It is a typed package (`py.typed`) and the
  main source of truth for framework behavior.
- Unit tests live next to the code as `*_test.py` under `py/flwr/`. Follow nearby
  test style; many tests use `pytest` parametrization, some older tests use
  `unittest`. Prefer `pytest` for new tests.
- `proto/flwr/proto/` contains protobuf sources. Generated Python protobuf files
  live under `py/flwr/proto/`.
- `py/flwr/supercore/state/schema/` contains SQLAlchemy Core table metadata.
  Alembic revisions live in `py/flwr/supercore/state/alembic/versions/`.
- `docs/source/` contains Sphinx docs. `docs/build/` is generated output.
- `e2e/` contains standalone Flower apps and shell scripts used by CI.
- `docker/`, `swift/`, `kotlin/`, and `cc/` contain non-Python packaging/client
  surfaces. Keep Python framework changes separate from those unless the behavior
  actually crosses the boundary.
- Avoid editing or committing generated/cache outputs such as `.venv/`,
  `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `docs/build/`, and `dist/`.

## Environment and commands

Run framework commands from `framework/` unless a command is explicitly shown from
the repository root. Use Python `3.10.19` for local agent commands; CI also checks
newer Python versions, but project tooling targets Python 3.10 syntax.

Preferred local agent form:

```bash
cd framework
uv run --no-sync --python=3.10.19 <command>
```

If the environment is missing or dependencies changed, synchronize intentionally:

```bash
cd framework
uv sync --locked --python=3.10.19 --all-extras --all-groups
```

CI installs with Poetry, so when reproducing CI setup exactly use:

```bash
cd framework
python -m poetry install --all-extras
```

Useful checks:

```bash
# Fast package-only quality gate, skips e2e/docs/copyright extras in dev/test.sh
uv run --no-sync --python=3.10.19 ./dev/test.sh false

# Full framework quality gate used by pre-commit/CI
uv run --no-sync --python=3.10.19 ./dev/test.sh

# Narrow tests
uv run --no-sync --python=3.10.19 python -m pytest py/flwr/path/to_test.py
uv run --no-sync --python=3.10.19 python -m pytest py/flwr -k "name"

# Targeted type/lint checks
uv run --no-sync --python=3.10.19 python -m mypy py
uv run --no-sync --python=3.10.19 python -m ruff check py/flwr --no-respect-gitignore
uv run --no-sync --python=3.10.19 python -m pylint --ignore=py/flwr/proto py/flwr
```

`dev/test.sh` sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` for pytest because Ray's uv
runtime-env hook can stall under `uv run`. Use the same environment variable when
debugging Ray/simulation tests directly.

Format broadly only when appropriate:

```bash
uv run --no-sync --python=3.10.19 ./dev/format.sh
```

For narrow edits, prefer targeted `isort`, `black`, `docformatter`, and `ruff`
commands on the touched files. Keep `py/flwr/proto/` excluded from Python
formatters because it is generated.

## Python conventions

- Keep core `flwr` framework code ML-framework-agnostic. Do not add PyTorch,
  TensorFlow, JAX, sklearn, or similar dependencies to the core package for a
  narrow feature; e2e apps and examples carry framework-specific dependencies.
- Use explicit, typed interfaces compatible with strict mypy. Avoid untyped
  helpers unless nearby code already establishes the pattern.
- Use NumPy-style docstrings for public classes/functions. `ruff` enforces
  pydocstyle with the NumPy convention, and `docsig` checks signatures.
- Keep imports consistent with `isort` and `black` line length 88. The project
  commonly uses `from __future__ import annotations` in newer modules.
- New source files should include the Flower Apache license header matching
  nearby files.
- Prefer existing helpers for logging, exit handling, serialization, config
  parsing, and CLI errors. Do not introduce parallel utility layers without a
  concrete need.

## Public API rules

Flower's Python public API is defined by recursively following `__all__` from
`flwr/__init__.py`. See `docs/source/contributor-explanation-public-and-private-apis.rst`.

- Adding a public symbol usually requires importing it in the relevant
  `__init__.py`, adding it to `__all__`, adding/updating API docs in
  `docs/source/reference.rst`, and adding tests for the public import path.
- Do not expose implementation modules accidentally. Prefer
  `from .module import Name as Name` plus `__all__ = ["Name"]`, as nearby code
  does.
- Treat existing public behavior as compatibility-sensitive. If changing or
  removing public APIs, add deprecation handling/tests instead of hard breaks
  unless the task explicitly calls for a breaking change.

## Protobufs

- Edit `.proto` files in `proto/flwr/proto/`, not generated files in
  `py/flwr/proto/`.
- Regenerate Python protobuf outputs with:

```bash
uv run --no-sync --python=3.10.19 ./dev/protoc.sh
```

- CI runs `./framework/dev/check-protos.sh` from the repository root. That script
  reruns generation and fails if generated files differ from `HEAD`, so in a
  local dirty worktree it can report expected uncommitted generated changes.
  Use it as a clean-tree/CI parity check.
- Wire-format changes need serialization/deserialization tests, usually near
  `py/flwr/common/serde_test.py` or the module-specific test.

## Database schema and migrations

Use the Alembic generator for schema diffs. Do not hand-write a new migration for
a normal SQLAlchemy metadata change.

```bash
cd framework
uv run --no-sync --python=3.10.19 python -m dev.generate_migration "Describe schema change"
```

After generation:

- Confirm the new revision's `down_revision` and branch target are correct.
- Confirm generated operations match the SQLAlchemy metadata change.
- Review SQLite compatibility, especially `batch_alter_table` blocks.
- Update `py/flwr/supercore/state/schema/README.md` when table metadata changes.
  `dev/format.sh` regenerates this schema documentation through `paracelsus`.
- Run the migration check when schema work is involved:

```bash
uv run --no-sync --python=3.10.19 ./dev/check-migrations.sh
```

## Tests and e2e

- Put focused unit coverage next to the changed module. Prefer a narrow pytest
  command first, then broader checks.
- For CLI changes, add tests around Typer command parsing and removed/deprecated
  flags. Existing tests under `py/flwr/cli/`, `py/flwr/supernode/cli/`, and
  `py/flwr/supercore/cli/` show the expected patterns.
- For SuperLink/SuperNode/state changes, consider targeted tests under
  `py/flwr/server/superlink/`, `py/flwr/supernode/`, and `py/flwr/supercore/`.
- E2E scripts often mutate app-local `pyproject.toml`, generate certs, create
  sqlite DB files, and start background processes. Do not run them in an e2e app
  directory with unrelated local edits.
- Common e2e commands, from an app directory such as `framework/e2e/e2e-bare/`:

```bash
python simulation.py
./../test_superlink.sh e2e-bare
./../test_superlink.sh e2e-bare rest
./../test_superlink.sh e2e-bare sqlite
```

## Docs, packaging, and locks

- Build docs with:

```bash
uv run --no-sync --python=3.10.19 ./dev/build-docs.sh
```

Docs builds require system `pandoc`. Do not commit `docs/build/` output.

- Build and check release artifacts with:

```bash
uv run --no-sync --python=3.10.19 ./dev/build.sh
uv run --no-sync --python=3.10.19 ./dev/test-wheel.sh
```

Do not commit `dist/` artifacts.

- If dependency constraints change, update `pyproject.toml` and `uv.lock`
  intentionally. CI checks `uv.lock` freshness, while framework CI still
  installs with Poetry.
