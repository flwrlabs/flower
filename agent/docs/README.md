# Flower Agent documentation

This directory contains the standalone documentation site for Flower Agent.
It is compiled separately from the Flower framework documentation in
`framework/docs/source`.

The Agent documentation covers task-oriented guides, concepts, examples, and
operational guidance. The generated Python API reference remains in the
framework documentation because `flwr.agentapp` is part of the `flwr` package.

## Build locally

From the repository root, install the dedicated documentation environment:

```bash
uv sync --project agent --locked --python=3.11.14
```

Then format-check and build the documentation:

```bash
uv run --project agent --locked --python=3.11.14 \
    mdformat --check agent/docs/source
uv run --project agent --locked --python=3.11.14 \
    sphinx-build -E -a -W --keep-going -b html \
    agent/docs/source agent/docs/build/html
```

Open `agent/docs/build/html/index.html` in a browser to view the result.

After the initial environment setup, the Makefile provides the same shortcuts as
the other standalone Flower documentation projects:

```bash
make -C agent/docs docs
make -C agent/docs serve
```

The second command rebuilds the site and serves it at
<http://localhost:8000>.
