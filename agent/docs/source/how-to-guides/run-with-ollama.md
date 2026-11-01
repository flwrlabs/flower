# Run an AgentApp with a local SuperLink and Ollama

Run a Flower AgentApp entirely on your machine with an
[Ollama](https://ollama.com/) model and a local SuperLink. This setup does not
require a Flower account or model API key.

This guide builds on [Run an AgentApp with a local
SuperLink](run-with-local-superlink.md). It uses `qwen3.5:4b`, but you can use
another model that supports the required Ollama endpoint.

```{warning}
This guide disables TLS and is intended for local development only. Do not
expose the Ollama or SuperLink ports to an untrusted network.
```

## Install the prerequisites

Install:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Ollama 0.13.3 or newer

Pull the example model:

```console
$ ollama pull qwen3.5:4b
```

Start Ollama if it is not already running through the desktop app:

```console
$ ollama serve
```

Ollama listens on `127.0.0.1:11434` by default. Leave this terminal open.

## Prepare the AgentApp

Follow [Write your first AgentApp](../tutorials/write-your-first-agentapp.md)
through **Validate the bundle** to create and build a project. In its
`agent/agent_app.py`, replace the `MODEL` value with `"qwen3.5:4b"` or another
model available in your local Ollama instance. Run `ollama list` to see the
models you have pulled.

## Connect SuperLink to Ollama

Start an insecure local SuperLink from the AgentApp project directory and point
its model provider at Ollama's OpenAI-compatible Responses endpoint:

```console
$ export FLWR_MODEL_API_ENDPOINT="http://127.0.0.1:11434/v1/responses"
$ uv run flower-superlink --insecure
```

You do not need to set `FLWR_MODEL_API_KEY` because the local Ollama endpoint
does not require authentication. Leave this terminal open.

## Chat with the app

Add the local connection as shown in {ref}`add-a-local-superlink-connection`.
Then, from the AgentApp project directory, select it and start Flower Chat:

```console
$ export FLWR_CHAT_SUPERLINK=local-agent
$ uv run flwr chat
```

```{tip}
To load the app and start chatting, see {ref}`load-an-agentapp-in-flower-chat`.
```

## Troubleshoot Ollama runs

- **`/v1/responses` returns `404`:** upgrade Ollama to version 0.13.3 or newer.
- **Port `11434` refuses connections:** start Ollama with `ollama serve`, or
  confirm that the desktop app is running.
- **Ollama reports that the model is missing:** run `ollama pull <model>` and
  retry the chat message.
- **Port `8000` refuses connections:** confirm that SuperLink is running and
  that `local-agent` points to `127.0.0.1:8000`.
- **The run fails without enough detail:** inspect it with `uv run flwr list --run-id <run-id> local-agent` and `uv run flwr log <run-id> local-agent --show`.

Press {kbd}`Ctrl+C` in the SuperLink and `ollama serve` terminals when you are
finished.
