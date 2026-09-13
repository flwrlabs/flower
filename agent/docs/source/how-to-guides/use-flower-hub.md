# Explore and publish AgentApps on Flower Hub

Flower Hub lets you discover AgentApps that you can run on SuperGrid and share
your own AgentApps with other Flower users. Published apps use an app spec such
as `@publisher/agent-name`.

## Explore AgentApps

Open the [Flower Hub app catalog](https://flower.ai/apps) and select type
**Agent** to browse agents. Open an AgentApp to review its description,
source code, available versions, and app spec before running it.

### Add an AgentApp to a federation

Sign in and open the AgentApp to review its description and source. If you want
to make it available in a federation, select the **Add app to federation** plus
button, choose one of your active federations, and select **Confirm**.

Open that federation from the Flower Agent sidebar and select **New chat**.
The AgentApp is now available in the agent selector above the prompt.

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
license = { file = "LICENSE" }
dependencies = ["flwr>=1.35.0,<2.0"]

[tool.flwr.app]
publisher = "your-username"
display-name = "Hello Agent"
flwr-version-target = "1.35.0"

[tool.flwr.app.components]
agentapp = "hello_agent.agent_app:app"
```

Add a top-level `LICENSE` file containing the license text. FAB format v1
requires this declared license file and a `flwr` dependency with an inclusive
lower bound. The target version must satisfy that dependency constraint.

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

### Review the files, then publish

Before running the publish command, inspect the project for source,
configuration, documentation, and data files that must not become public. The
publish filter considers supported files throughout the project, even when
they are not tracked by Git. Remove any sensitive file or add it to the
project's `.gitignore`, then review the project again.

Do not use the `Attach:` output from the publish command as a review step. The
command uploads the attached files immediately after printing their names and
does not ask for confirmation.

After you have reviewed the eligible files, publish from the project
directory:

```console
$ uv run flwr app publish .
```

Flower applies its publish filters and the project's `.gitignore`, validates
the files, and sends them to Flower Hub. The uploaded sources are public.

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

An app ID cannot change between an Agent and a Federated app. Use a new
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
