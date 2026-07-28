# Flower Agent documentation

This directory contains the standalone documentation site for Flower Agent.
It is compiled separately from the Flower framework documentation in
`framework/docs/source`.

The Agent documentation covers task-oriented guides, concepts, examples, and
operational guidance. The generated Python API reference remains in the
framework documentation because `flwr.agentapp` is part of the `flwr` package.

## Build locally

From the `framework` directory, run:

```bash
uv run --no-sync --python=3.11.14 \
  sphinx-build -W --keep-going -b html \
  docs/agent/source docs/agent/build/html
```

Open `docs/agent/build/html/index.html` in a browser to view the result.
