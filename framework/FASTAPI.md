# FastAPI

## SuperLink Operating Modes

With the new HTTP API, there are now four different options to start and run the SuperLink:

1. **Legacy Mode** `flower-superlink` (without `--enable-http-api`): This starts the SuperLink in "legacy mode" with only gRPC APIs, but no HTTP API.
2. **Compatibility Mode** `flower-superlink --enable-http-api`: Tthis starts the SuperLink in Compatibility Mode with both the HTTP API and the legacy gRPC APIs. **This is what we're running in prod until the gRPC-to-HTTP conversion is complete.** Note that in Compatibility Mode, FastAPI is limited to only 1 worker, which is a serious limitation during this transition.

    ```
    uv run flower-superlink --enable-http-api --insecure
    ```

3. **Next Mode** `flower-superlink --enable-http-api --disable-grpc-api` with `--enable-http-api` and `--disable-grpc-api`: This starts the SuperLink in "HTTP mode" with only the HTTP API, but not the legacy gRPC APIs
4. **Experimental Mode** `uvicorn flwr.superlink.main:app`: This starts the SuperLink in "experimental mode" via uvicorn, skipping the `flower-superlink` argument parsing. This mode is experimental because it needs to reach parity with `flower-superlink --enable-http-api --disable-grpc-api`.


## Install

To run FastAPI, install `flwr` with all extras to ensure the `rest` extra is included:

```
uv sync --locked --all-extras
```

## Run

Start the SuperLink's FastAPI server using uvicorn:

```
uv run uvicorn flwr.superlink.main:app
```

Start the SuperNode's FastAPI server using uvicorn:

```
uv run uvicorn flwr.supernode.main:app
```

## Docs

Docs are available once the SuperLink/SuperNode FastAPI server is running:

```
http://127.0.0.1:8000/docs
```
