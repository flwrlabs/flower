# FastAPI

## Install

```bash
uv sync --all-extras
```

## Run SuperLink

Start the SuperLink FastAPI server using uvicorn:

```bash
uv run uvicorn flwr.superlink.main:app
```

## Run SuperNode

Start the SuperNode FastAPI server using uvicorn:

```bash
uv run uvicorn flwr.supernode.main:app
```

## Docs

Docs are available once the SuperLink or SuperNode FastAPI server is running:

```text
http://127.0.0.1:8000/docs
```
