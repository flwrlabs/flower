# Run an AgentApp with a local SuperLink

Run an AgentApp locally with Flower's Community Edition runtime. This setup is
useful while developing an app: it runs SuperLink and the AgentApp processes on
your machine, without a SuperGrid account or Enterprise services.

Run the CLI and SuperLink from the same project environment to avoid protocol
mismatches.

If you haven't created an AgentApp yet, start with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md).

```{warning}
This guide disables TLS and is intended for local development only. Do not
expose the insecure SuperLink ports to an untrusted network.
```

(configure-local-model-provider)=

## Configure a model provider

The local runtime needs an Open Responses-compatible model endpoint. Set the
provider configuration in the terminal where you'll start SuperLink.

The deployment determines where upstream model access is configured:

| Deployment                                          | Required model configuration                                  |
| --------------------------------------------------- | ------------------------------------------------------------- |
| SuperGrid                                           | None in the AgentApp; Flower configures upstream model access |
| Self-hosted with Flower's default endpoint          | `FLWR_MODEL_API_KEY`                                          |
| Self-hosted with an authenticated custom endpoint   | `FLWR_MODEL_API_ENDPOINT` and `FLWR_MODEL_API_KEY`            |
| Self-hosted with an unauthenticated custom endpoint | `FLWR_MODEL_API_ENDPOINT` only                                |

`FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY` are injected into the
AgentApp in all of these cases. They authenticate the AgentApp to Flower and do
not configure the upstream provider used by SuperLink.

To use Flower's default model endpoint, provide a Flower API key:

```console
$ export FLWR_MODEL_API_KEY="<your-api-key>"
```

To use another Open Responses-compatible endpoint, set its full `/responses`
URL instead:

```console
$ export FLWR_MODEL_API_ENDPOINT="http://127.0.0.1:8080/v1/responses"
$ export FLWR_MODEL_API_KEY="<your-provider-api-key>"
```

You can omit `FLWR_MODEL_API_KEY` when the custom endpoint does not require
authentication. The model and AgentApp subprocesses inherit these variables
from SuperLink.

Without either variable, model requests through a self-hosted SuperLink fail
with:

```text
HTTP 502 {"message":"Model API key is not set (FLWR_MODEL_API_KEY)."}
```

### Use a fully on-premises model

A local inference server is a drop-in model provider only when it implements the
Open Responses protocol at a URL ending in `/responses`. An endpoint that only
implements the OpenAI-compatible `/chat/completions` API is not directly
compatible with Flower's model plane.

For a fully local supported example, follow [Run an AgentApp with a local
SuperLink and Ollama](run-with-ollama.md). For another inference server, point
`FLWR_MODEL_API_ENDPOINT` at its Open Responses endpoint. If the server only
supports Chat Completions, you must either provide your own protocol adapter or
connect to it directly from the AgentApp. Flower does not currently provide a
Chat Completions-to-Open Responses adapter.

Account connectors configured in SuperGrid are not available to this local
unauthenticated runtime. Built-in connectors depend on the local runtime and
provider configuration.

## Start SuperLink

From your AgentApp project directory, start an insecure local SuperLink:

```console
$ uv run flower-superlink --insecure
```

SuperLink starts its HTTP API on `127.0.0.1:8000` and launches the local
processes needed to execute the AgentApp and its model requests. You don't need
to start a SuperNode for an AgentApp run.

Leave this terminal open while you run the app.

## Add a local connection

In another terminal, add a connection for the local Control API to
`~/.flwr/config.toml`:

```toml
[superlink.local-agent]
address = "127.0.0.1:8000"
insecure = true
```

The connection name is `local-agent`. It doesn't require `flwr login` because
the local SuperLink has no account authentication configured.

## Run the AgentApp

From the AgentApp project directory, submit the app through the new connection:

```console
$ uv run flwr run . local-agent --stream
```

Flower builds the app, submits it to the local SuperLink, and streams the
AgentApp process logs. Override the configured prompt in the same way as a
SuperGrid run:

```console
$ uv run flwr run . local-agent \
    --run-config 'agent.input="Explain Flower Agent in one sentence."' \
    --stream
```

## Inspect and stop the run

Use the same CLI commands as you would for a remote run:

```console
$ uv run flwr list local-agent
$ uv run flwr list --run-id <run-id> local-agent
$ uv run flwr log <run-id> local-agent --show
$ uv run flwr stop <run-id> local-agent
```

Press {kbd}`Ctrl+C` in the SuperLink terminal when you're finished.

This guide starts a foreground, user-managed SuperLink. `flwr stop` stops a run;
it does not stop the SuperLink process. A SuperLink started automatically for a
`:local:` connection is a different, background-managed process. There is
currently no dedicated `flwr` command or PID file for stopping that process; see
[Run Flower locally with a managed
SuperLink](https://flower.ai/docs/framework/how-to-run-flower-locally.html#stop-the-background-local-superlink)
for the documented process-inspection procedure.

## Troubleshoot the local runtime

If a run fails, start with its details and logs:

```console
$ uv run flwr list --run-id <run-id> local-agent
$ uv run flwr log <run-id> local-agent --show
```

Common problems include:

- **Model API key is not set:** export `FLWR_MODEL_API_KEY` in the SuperLink
  terminal, then restart SuperLink
- **Invalid model endpoint:** `FLWR_MODEL_API_ENDPOINT` must include the full
  `/responses` path
- **Connection refused:** confirm that SuperLink is still running and that the
  configured address is `127.0.0.1:8000`
- **Version mismatch:** start SuperLink and run the CLI from the same project
  environment so they use compatible Flower versions
