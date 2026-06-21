# FastAPI

## Install

~~~
uv sync --locked --all-extras
~~~

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
