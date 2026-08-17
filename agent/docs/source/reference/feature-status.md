# Feature status

Flower Agent is experimental. Use this page to check whether a capability is
available in the interface and workspace where you intend to use it.

Status meanings:

- **Available:** implemented end to end in the stated interface.
- **Experimental:** available, but its interface or behavior may change.
- **Limited:** implemented with an important scope or acceptance limitation.
- **Planned:** not a public product capability yet.

```{important}
This table was last source-verified on 2026-08-17 against Flower 1.34.0 source.
The browser quickstart and conversation context were also live-verified against
flower.ai on that date. Flower 1.34.0 was not yet published on PyPI, so the
packaged CLI path and the provider-specific paths listed below still require
release acceptance.
```

| Capability                                            | Interface                                  | Scope                                        | Requirement                          | Status       |
| ----------------------------------------------------- | ------------------------------------------ | -------------------------------------------- | ------------------------------------ | ------------ |
| Chat with the default Flower Agent                    | SuperGrid browser                          | Agent execution workspace                    | Flower Agent account access          | Experimental |
| Chat with the default Flower Agent                    | `flwr chat`                                | Federation configured for `supergrid`        | Flower 1.34.0 and login              | Experimental |
| Select an agent                                       | Browser or leading `@agent` in `flwr chat` | Agents returned for the current federation   | Available agent catalog              | Experimental |
| Run a local AgentApp                                  | `flwr run`                                 | Default or explicit federation               | Valid AgentApp project               | Available    |
| Continue a browser conversation                       | SuperGrid browser                          | One run series                               | AgentApp must replay stored messages | Available    |
| Restore an old conversation interactively             | `flwr chat`                                | None                                         | —                                    | Planned      |
| `web_search`                                          | AgentApp                                   | Built-in                                     | Exposed by AgentApp code             | Available    |
| `web_fetch`                                           | AgentApp                                   | Built-in public URLs                         | Exposed by AgentApp code             | Available    |
| `browser_use`                                         | AgentApp                                   | Built-in                                     | Deployment availability not accepted | Limited      |
| `start_automation`                                    | AgentApp                                   | Run series and federation of the current run | Explicit future or recurring request | Experimental |
| Slack, Notion, GitHub, and Attio                      | SuperGrid browser runs                     | Personal workspace only                      | Connected account and run selection  | Limited      |
| Inspect and stop automations                          | SuperGrid Settings                         | Agent execution federation                   | Flower Agent account access          | Experimental |
| Persistently add or remove federation agents          | SuperGrid                                  | Collaborative federation                     | —                                    | Planned      |
| Publish or filter first-class AgentApps in Flower Hub | Hub                                        | —                                            | —                                    | Planned      |
| Public `/v1/runtime/responses` endpoint               | HTTP                                       | —                                            | —                                    | Planned      |

## Important boundaries

### Run series and model context

A run series stores related runs and their Flower `Context`. It does not force
an AgentApp to send that history to a model. The current default Flower Agent
replays stored user and assistant messages; a custom AgentApp must implement
that behavior itself.

### Connectors and federations

Built-in tools are selected by AgentApp code. Account connectors are selected
for a browser run and use the signed-in user's connection. Flower 1.34.0
restricts runs with account connectors to the user's personal workspace; do
not expect Slack, Notion, GitHub, or Attio access in a collaborative
federation.

### Source presence and deployment acceptance

The following paths need fresh acceptance in the target deployment before they
can be treated as generally available:

- connecting a non-Flower Slack workspace;
- connecting an external Notion workspace;
- Attio authorization without administrator friction;
- `browser_use` availability for users;
- connector recovery after a timeout or missing heartbeat; and
- the released PyPI and browser flows used by the deployment.

When a path has not passed that acceptance, this documentation describes its
current limitation instead of promising success from source code alone.
