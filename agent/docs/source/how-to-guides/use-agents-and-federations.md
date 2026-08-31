# Use agents and federations

Flower uses a few related objects to describe AgentApp execution. This guide
defines them, then shows how to select an agent and execution federation in the
current browser and CLI interfaces.

## Learn the mental model

**AgentApp**
: Python control flow registered with `AgentApp.main`. It decides which model
and connector calls to make and how to handle their results.

**Agent**
: An AgentApp made available for selection. An app spec such as
`@flwrlabs/flwr-agent` identifies a published app. SuperGrid may resolve the
selection to a specific Flower App Bundle (FAB) hash.

**Run**
: One execution of one app in one federation. Every message submitted through
a chat interface starts a run.

**Run series or conversation**
: A group of related runs with shared Flower `Context`. The browser presents a
run series as a conversation. Sharing a series does not itself make history
visible to a model; the AgentApp must replay stored messages.

**Federation**
: A SuperGrid workspace that owns runs, run series, members, and execution
resources. Your account has a personal workspace and can be a member of
multiple collaborative federations.

**Connector**
: A runtime-provided tool. Built-in tools need no external account. Account
connectors use authorization granted by the signed-in user and are currently
restricted to personal-workspace runs.

## List your federations

Authenticate first, then ask SuperGrid for the federations visible to your
account:

```console
$ uvx --from flwr==1.35.0 flwr login supergrid
$ uvx --from flwr==1.35.0 flwr federation list supergrid
```

Use the full federation ID shown by this command, including its leading `@`, in
later commands.

## Choose a federation for `flwr run`

The following `uv run` examples use the Flower version installed in your local
project environment. To stay consistent with the rest of this guide, ensure
that environment has `flwr==1.35.0` installed.

Flower selects the federation in this order:

1. The value passed with `--federation`.
1. The `federation` configured for the selected SuperLink connection.
1. The account's default federation when neither is set.

With no command-line option, this command uses the configured federation or
falls back to the account default:

```console
$ uv run flwr run . supergrid
```

To choose explicitly:

```console
$ uv run flwr run . supergrid \
    --federation @account/federation-name
```

The app can also be a published app spec:

```console
$ uvx --from flwr==1.35.0 flwr run @publisher/agent supergrid \
    --federation @account/federation-name \
    --run-config 'agent.input="Summarize this federation task."'
```

The account must be a member of the federation and entitled to start the
AgentApp there.

## Choose the `flwr chat` federation

`flwr chat` starts in your `@account/personal` federation. Enter
`/federation` to open the completion menu, then select a federation visible to
your account. You can also type its full name:

```text
/federation @account/federation-name
```

Switching federations clears the transcript, resets the selected agent to
Flower Agent, and starts a new conversation. Use `flwr run --federation` when
you want to select the federation in a non-interactive command.

## Select an agent in Flower Chat

At an empty `flwr chat` prompt, type `@`. The completion menu lists agents
returned for the active federation. Choose one and add your request:

```text
@publisher/agent Compare the two proposed approaches.
```

The selected label above the prompt changes immediately. Flower keeps using
that agent for later messages. Selecting a different agent clears the current
series ID, so the request starts a new run series. Use `/new` to start a new
series without changing agents.

## Select an agent in SuperGrid

Open [Flower Agent](https://flower.ai/app). The sidebar pins your personal
federation and lists the other federations visible to your account. Select a
federation, open **New chat**, and choose one of its assigned agents above the
prompt.

The selected federation owns the run and its conversation. Recent AgentApp
conversations appear below that federation in the sidebar, where you can open
them again. Flower excludes non-AgentApp run series from this chat history.

## Know when a new series starts

| Action                                       | Result                                |
| -------------------------------------------- | ------------------------------------- |
| Send another message with the selected agent | Reuse the current run series          |
| Enter `/new` in Flower Chat                  | The next message starts a new series  |
| Select a different leading `@agent`          | Start a new series with that agent    |
| Enter `/history` in Flower Chat              | Select and continue an older series   |
| Switch federations in Flower Chat            | Start a new series in that federation |
| Select **New chat** in SuperGrid             | Start a new browser conversation      |
| Select a browser conversation                | Continue its existing run series      |
| Run `flwr run` without a series ID           | Start an independent run or series    |

## Use connectors in the right workspace

Built-in tools such as `web_search` and `web_fetch` are chosen in AgentApp code.
Slack, Notion, GitHub, and Attio are account connectors selected for a browser
run. Flower 1.35.0 rejects account-connector references for a collaborative
federation; run that task in your personal workspace instead.

See [Connect accounts](connect-accounts.md) for setup and the exact read-only
capability matrix.
