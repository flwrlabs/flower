# Use agents and federations

Choose an execution federation or an agent already assigned to it in Flower
Chat or the browser. For the underlying concepts, see [Understand the AgentApp
runtime](../explanations/agentapp-runtime.md).

```{tip}
To load a local or downloaded AgentApp, see {ref}`load-an-agentapp-in-flower-chat`.
```

## List your federations

After [logging in to SuperGrid](run-on-supergrid.md), list the federations
visible to your account:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr federation list supergrid
```

Use the full federation ID shown by this command, including its leading `@`, in
later commands.

## Choose the `flwr chat` federation

`flwr chat` starts in your `@<account>/personal` federation. Enter
`/federation` to open the completion menu, then select a federation visible to
your account. You can also type its full name:

```text
/federation @<account>/<federation-name>
```

Switching federations clears the transcript, resets the selected agent to
Flower Agent, and starts a new conversation. Your account must be a member of
the federation and entitled to start AgentApps there.

## Select an agent in Flower Chat

At an empty `flwr chat` prompt, type `@`. The completion menu lists agents
returned for the active federation. Choose one and add your request:

```text
@<publisher>/<agent> Compare the two proposed approaches.
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

For conversation and run-series behavior, see [Understand the AgentApp
runtime](../explanations/agentapp-runtime.md). For account connector access by
federation, see [Connect accounts](connect-accounts.md).
