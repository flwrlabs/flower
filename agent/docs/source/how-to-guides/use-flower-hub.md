# Explore and publish AgentApps on Flower Hub

Flower Hub lets you discover AgentApps that you can run on SuperGrid and share
your own AgentApps with other Flower users. Published apps use an app spec such
as `@publisher/agent-name`.

## Explore AgentApps

Open the [Flower Hub app catalog](https://flower.ai/apps) and select
**AgentApps** to browse agents. Open an AgentApp to review its description,
source code, available versions, and app spec before running it.

### Add an AgentApp to a federation

Sign in, then select the **Add app to federation** button on an AgentApp card.
Choose one of your active federations in the dialog and select **Confirm**.

Open that federation from the Flower Agent sidebar and select **New chat**.
The AgentApp is now available in the agent selector above the prompt. Review
its description and source before confirming that you want to make it
available in the federation.

### Run an AgentApp from the terminal

Use the publisher and project name from the app spec:

```console
$ uvx --from flwr==1.35.0 flwr run @publisher/agent-name supergrid \
    --run-config 'agent.input="What can you help me with?"' \
    --stream
```

See [Run an AgentApp on SuperGrid](run-on-supergrid.md) to choose a federation,
inspect logs, and stop a run.

## Publish your AgentApp

Start with a working project from [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md). This guide targets Flower
1.35.0.

### Prepare the project

Check the public metadata in `pyproject.toml`:

```toml
[project]
name = "hello-agent"
version = "0.1.0"
description = "Answer questions with a Flower AgentApp"
license = "Apache-2.0"

[tool.flwr.app]
publisher = "your-username"
display-name = "Hello Agent"
flwr-version-target = "1.35.0"

[tool.flwr.app.components]
agentapp = "hello_agent.agent_app:app"
```

The `publisher` must match your Flower account username. Flower identifies the
project as an AgentApp from the `agentapp` component, so you do not need to add
a tag or another app-type setting.

Before publishing:

- choose a final project name because it becomes part of the app spec
- write a short description that explains what the AgentApp does
- include a `README.md` with setup, configuration, and usage instructions
- remove credentials, local data, and private connector content
- update `.gitignore` so local-only files are excluded

### Validate the AgentApp

Create the environment and build the app locally:

```console
$ uv sync
$ uv run flwr build
```

Fix configuration, dependency, and component-reference errors before
publishing. A successful build reports the path of the generated `.fab` file.
The publish command uploads the project sources rather than this local bundle,
and Flower Hub builds the FAB again on the server.

### Log in to SuperGrid

```console
$ uv run flwr login supergrid
```

Complete authentication in the browser window. The signed-in account must
match the `publisher` value in `pyproject.toml`.

### Review the files and publish

Run the command from the project directory:

```console
$ uv run flwr app publish .
```

Flower applies its publish filters and `.gitignore`, prints skipped and
attached files, validates the upload, and then sends it to Flower Hub. Review
the printed file list carefully because the uploaded sources are public.

After a successful upload, open:

```text
https://flower.ai/apps/<publisher>/<project-name>/
```

For the complete file-type, size, license, and FAB-format rules, see [Publish an
App on Flower
Hub](https://flower.ai/docs/hub/how-to-publish-app-on-hub.html).

### Publish a new version

Keep the same project name and publisher, update `[project].version`, and
publish again:

```toml
[project]
version = "0.1.1"
```

```console
$ uv run flwr build
$ uv run flwr app publish .
```

An app ID cannot change between an AgentApp and a federated app. Use a new
project name if you need to publish a different app type.

### Troubleshoot publishing

- **Please log in before publishing app**: run `uv run flwr login supergrid`
- **Publisher mismatch**: set `publisher` to the username of the signed-in
  Flower account
- **Missing or invalid app description**: add a non-empty `description` under
  `[project]`
- **Required file was skipped**: review `.gitignore` and the publish include and
  exclude rules
- **Component cannot be loaded**: check the module and object named by
  `[tool.flwr.app.components].agentapp`
