# FastAPI

## Install

To run FastAPI, install `flwr` with all extras:

```bash
uv sync --locked --all-extras
```

## SuperLink Run Modes

With the new HTTP API, there are now four different options to start and run the SuperLink:

1. **Legacy Mode** `flower-superlink` (without `--enable-http-api`): This starts the SuperLink in "legacy mode" with only gRPC APIs, but no HTTP API.

   ```bash
   uv run flower-superlink --insecure
   ```

1. **Compatibility Mode** `flower-superlink --enable-http-api`: This starts the SuperLink in Compatibility Mode with both the HTTP API and the legacy gRPC APIs. **This is what we're running in prod until the gRPC-to-HTTP conversion is complete.** Note that in Compatibility Mode, FastAPI is limited to only 1 worker, which is a serious limitation during this transition.

   ```bash
   uv run flower-superlink --insecure --enable-http-api
   ```

1. **Next Mode** `flower-superlink --enable-http-api --disable-grpc-api`: This starts the SuperLink in "HTTP mode" with only the HTTP API, but not the legacy gRPC APIs.

   ```bash
   uv run flower-superlink --insecure --enable-http-api --disable-grpc-api
   ```

1. **Experimental Mode** `uvicorn flwr.superlink.main:app`: This starts the SuperLink in "experimental mode" via uvicorn, skipping the `flower-superlink` argument parsing. This mode is experimental because it needs to reach parity with `flower-superlink --enable-http-api --disable-grpc-api`.

   ```bash
   uv run uvicorn flwr.superlink.main:app
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
