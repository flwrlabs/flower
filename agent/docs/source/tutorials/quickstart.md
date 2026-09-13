# Chat in your browser

Use the browser to get your first Flower Agent response. You don't need to
install Flower or provide a model API key.

```{note}
The browser interface is experimental and may change between releases.
```

## Before you start

Confirm that you have:

- a Flower account with Flower Agent access
- access to the email or identity provider used to sign in
- a current desktop browser
- a reliable network connection

If you plan to use Slack, Notion, GitHub, or Attio later, make sure you can
authorize the relevant account. Account connectors work only in your personal
Flower workspace, and you won't need one for this quickstart.

## Open Flower Agent

Go to [flower.ai/app](https://flower.ai/app), select **Sign in**, and complete
the authentication flow. When you return, you should see **New chat**, a prompt
field, an agent selector, and your federations in the sidebar.

Flower starts in your `@account/personal` federation. The federation shown in
the page header or breadcrumb is where this conversation's runs will execute.

```{figure} ../_static/screenshots/browser-new-chat.png
:alt: Flower Agent new-chat screen with an empty prompt and Flower Agent selected.

A new browser chat starts with **Flower Agent** selected. The connector control
is beside the agent selector.
```

## Choose an agent

Keep **Flower Agent** selected for the first run. If the selector is empty or
still loading after a refresh, see [Troubleshoot AgentApp
runs](../how-to-guides/troubleshoot-agent-runs.md).

```{figure} ../_static/screenshots/agent-selection-menu.png
:alt: Flower Agent selector showing Flower Agent and GPT 5.6 Chat.
:width: 210px

The agent selector lists the agents available for the current execution
workspace.
```

Leave account connectors unselected for now. You can add them after the first
run works.

## Send a deterministic prompt

Enter:

```text
Reply with exactly: Flower Agent is ready.
```

Submit the prompt and wait for the run to finish.

```{admonition} Success checkpoint
:class: tip

You should see your prompt and a completed response in the same conversation.
The exact wording may differ, but the run should start, show output, and finish
without an error.
```

## Check conversation context

In the same conversation, ask:

```text
What exact phrase did I ask you to return?
```

The default Flower Agent includes earlier messages from the run series, so it
should remember the phrase. Custom AgentApps need to replay stored messages
themselves.

```{figure} ../_static/screenshots/browser-conversation-context.png
:alt: Flower Agent conversation showing the initial response, a context-dependent follow-up, and a completed run.

The follow-up uses the earlier turn, and the conversation shows the run as
completed.
```

Select **New chat** before starting an unrelated task.

The conversation appears below the personal federation in the sidebar. Select
it there when you want to return to it. You can also select another federation
in the sidebar and start a chat with one of the agents assigned to it.

## What happened

Each submitted message starts one **run** of the selected **AgentApp**. Flower
groups related runs in a **run series**, which the browser presents as a
conversation. The run series belongs to a **federation**. The runtime gives the
AgentApp an OpenAI-compatible model endpoint, an `AgentSession` for connectors
and frontend-visible events, and a `Context` for run configuration and state.

See [Use agents and
federations](../how-to-guides/use-agents-and-federations.md) to learn how these
pieces fit together.

## If the checkpoint fails

If it fails, try these steps in order:

1. Confirm that you are signed in and have Flower Agent access
1. Refresh the browser
1. Start a new chat and retry the deterministic prompt once
1. Record the visible error, time, app spec, and federation name
1. Follow [Troubleshoot AgentApp
   runs](../how-to-guides/troubleshoot-agent-runs.md)

Never include access tokens, OAuth callback URLs, connector credentials, or
private source content in a support report.

## Next steps

- Prefer a terminal workflow? [Chat in your
  terminal](get-started-with-flower-agent.md) introduces `flwr chat`
- [Write your first AgentApp](write-your-first-agentapp.md)
- [Build a collaborative research agent](build-a-collaborative-agent.md)
- [Connect accounts](../how-to-guides/connect-accounts.md) only if your project
  needs account data
