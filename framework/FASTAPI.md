# FastAPI

## Install

To run FastAPI, install `flwr` with all extras:

```bash
uv sync --locked --all-extras
```

## Run SuperLink

`flower-superlink` starts the Runtime API over HTTP. During the migration of the
remaining APIs, Fleet and Control continue to use gRPC.

```bash
uv run flower-superlink --insecure
```

## Run SuperLink in Experimental Mode

Start the SuperLink's FastAPI server using uvicorn:

```bash
uv run uvicorn flwr.superlink.main:app
```

## Run SuperNode in Experimental Mode

Start the SuperNode's FastAPI server using uvicorn:

```bash
uv run uvicorn flwr.supernode.main:app
```

## Docs

Docs are available once the SuperLink or SuperNode FastAPI server is running:

```text
http://127.0.0.1:8000/docs
```
