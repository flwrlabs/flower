# Connector providers

Create a standard OAuth connector from `framework/`:

```bash
uv run --no-sync --python=3.11.14 \
  python -m dev.generate_connector <provider_ref> --display-name "Provider Name"
```

The command creates one package with:

- `definition.py`: provider metadata, OAuth configuration, and API settings;
- `actions.py`: model-facing action schemas and required OAuth scopes;
- `executors.py`: provider API calls using `ConnectorContext`;
- `__init__.py`: package marker.

It also regenerates `registry_generated.py`. Do not edit that file or the central
connector registry manually.

Provider definitions are imported at startup. Executor modules are imported only
when one of their actions runs. The shared runtime validates action/executor
correspondence, constructs OAuth flows, supplies credentials, configures the HTTP
client, maps provider errors, and records usage.

Every action must declare `ActionAccess.READ` or `ActionAccess.WRITE`. This
classification is provider-independent; `required_scopes` remains the source of
provider-native OAuth permissions.

After implementing a provider, run:

```bash
uv run --no-sync --python=3.11.14 \
  python -m dev.generate_connector --check
uv run --no-sync --python=3.11.14 \
  python -m pytest py/flwr/supercore/task_process/connector
```
