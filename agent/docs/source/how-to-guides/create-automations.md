# Create automations

An automation asks Flower to start future runs in the same run series. It can
run once or recur at a fixed interval.

```{important}
Create an automation only when the user explicitly requests future or recurring
execution. A general request such as “summarize my project” is not permission to
schedule later work.
```

## Use the default Flower Agent

The current default Flower Agent exposes the `start_automation` tool. In a
browser conversation, include both the work and its timing:

```text
At 09:00 Europe/London tomorrow, search the public web for new Flower releases
and summarize them in this conversation. Run once.
```

For a recurring request, specify an end condition:

```text
Starting at 09:00 Europe/London tomorrow, summarize new public Flower releases
every 24 hours, for three runs total.
```

Review the agent's confirmation. It should identify the next execution time and
whether the schedule repeats.

## Expose automation from a custom AgentApp

Request the runtime tool schema and include it in a model request:

```python
tools = agent.connectors.tools(["start_automation"])
```

When the model returns its `function_call`, pass it to:

```python
output = agent.connectors.call(tool_call)
```

The model-facing arguments are:

| Argument         | Required | Meaning                                                          |
| ---------------- | -------- | ---------------------------------------------------------------- |
| `input`          | Yes      | `agent.input` for every scheduled run                            |
| `start_at`       | Yes      | First run time as an ISO 8601/RFC 3339 timestamp with a timezone |
| `fixed_interval` | No       | Seconds between recurring runs; omit for one execution           |
| `max_runs`       | No       | Maximum executions; valid only with `fixed_interval`             |

A one-off function call can look like:

```json
{
  "input": "Summarize new public Flower releases.",
  "start_at": "2026-08-26T09:00:00+01:00"
}
```

A bounded recurring call can look like:

```json
{
  "input": "Summarize new public Flower releases.",
  "start_at": "2026-08-26T09:00:00+01:00",
  "fixed_interval": 86400,
  "max_runs": 3
}
```

Do not use a timezone-free value such as `2026-08-26T09:00:00`; the runtime
rejects it. Avoid an unbounded recurrence unless the user clearly requested one
and understands how to stop it.

## Understand automation scope

The 1.34.0 runtime builds scheduled runs from the current run request. It keeps
the automation's runs in the current run series and federation and replaces
`agent.input` with the scheduled `input`.

Production acceptance of federation and account-connector scoping is still
required for each deployment. Confirm the displayed federation and run series
before relying on a schedule. Account connectors remain limited to the personal
workspace.

## Inspect an automation

In SuperGrid:

1. open **Settings** > **Automations**;
1. use **Active** for upcoming schedules;
1. check the run series, next run time, remaining runs, fixed interval, and
   status; and
1. use **History** for completed, stopped, or failed schedules.

The settings page uses the Agent execution federation. If a newly created
automation does not appear, first confirm that the chat and settings page show
the same federation, then refresh once.

```{figure} ../_static/screenshots/automations-settings.png
:alt: Flower Automations settings with Active and History tabs and schedule columns.

The **Active** tab shows upcoming schedules. Use **History** for completed,
stopped, or failed automations.
```

## Stop an automation

On **Settings** > **Automations** > **Active**, select **Stop** on the relevant
row. Wait for its status to update before leaving the page.

Stopping prevents future scheduled runs. It does not stop a run that has
already started; stop that run separately from its run details or with
`flwr stop <run-id> supergrid`.

There is no public CLI command for listing or stopping automations in Flower
1.34.0. Use the SuperGrid settings page.

## Recover from a failed schedule

1. Open the run series and inspect the latest run details.
1. Check whether the model, built-in tool, or selected account connector failed.
1. Fix the underlying access problem before creating a replacement schedule.
1. Stop the old active automation if it can still retry.
1. Create a new bounded automation and verify its displayed next-run time.

Do not repeatedly create schedules while the UI is still loading; that can
produce duplicate future work.
