# Custom AgentApp FAB example

This app demonstrates how to define a local `AgentApp`, bundle it into a FAB,
submit it with `flwr run`, and inspect streamed run events through the Control API.

## Run

Start a SuperLink:

```bash
export FLWR_MODEL_API_KEY="YOUR API KEY"
uv run --no-sync --python=3.11.14 flower-superlink \
    --insecure \
    --control-api-address 127.0.0.1:39093 \
    --serverappio-api-address 127.0.0.1:39094 \
    --database /tmp/flwr-agent-run.db \
    --log-file /tmp/flwr-agent-superlink.log
```

Start a ServerApp SuperExec in another terminal:

```bash
export FLWR_MODEL_API_KEY="YOUR API KEY"
export BRAVE_API_KEY="YOUR BRAVE API KEY"
uv run --no-sync --python=3.11.14 flower-superexec \
    --insecure \
    --appio-api-address 127.0.0.1:39094 \
    --plugin-type serverapp
```

For web search, set one of `BRAVE_API_KEY`, `TAVILY_API_KEY`, or `EXA_API_KEY`
in the SuperExec terminal.

Submit the local AgentApp. `flwr run` builds the FAB from this directory and
submits it to the SuperLink:

```bash
uv run --no-sync --python=3.11.14 flwr run e2e/e2e-agentapp \
    --run-config 'agent.input="What is the Flower federated learning framework? Answer in one sentence."'
```

Disable web search for a run with:

```bash
uv run --no-sync --python=3.11.14 flwr run e2e/e2e-agentapp \
    --run-config 'agent.input="Say hello in one short sentence." agent.web-search=false'
```

You can also write the FAB file explicitly:

```bash
uv run --no-sync --python=3.11.14 flwr build --app e2e/e2e-agentapp
```

After `flwr run` prints the run ID, stream task events:

```bash
grpcurl -plaintext \
    -import-path proto \
    -proto flwr/proto/control.proto \
    -d '{"run_id": 1}' \
    127.0.0.1:39093 \
    flwr.proto.Control/StreamRunEvents
```

Replace `1` with the run ID returned by `flwr run`.
