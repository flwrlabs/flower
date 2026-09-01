#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Default value (true)
RUN_FULL_FORMAT=${1:-true}
echo "RUN_FULL_FORMAT: $RUN_FULL_FORMAT"

taplo fmt

# Python
python -m devtool.check_copyright py/flwr
python -m devtool.init_py_fix py/flwr
python -m isort --skip py/flwr/proto py
python -m black -q --exclude py/flwr/proto py
python -m ruff check --fix py/flwr

# Protos
find proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format -i

if $RUN_FULL_FORMAT; then
  # E2E
  python -m isort e2e
  python -m black -q e2e
fi

if $RUN_FULL_FORMAT; then
  # Markdown
  python -m mdformat --number docs/source

  # RST
  docstrfmt docs/source
fi

# Helm chart READMEs (if available in the synced internal repository)
if [ -f helm/flower-client/README.md ] && [ -f helm/flower-client/values.yaml ] \
  && [ -f helm/flower-server/README.md ] && [ -f helm/flower-server/values.yaml ]; then
  npx --yes --package=@bitnami/readme-generator-for-helm@2.7.2 readme-generator \
    --readme=helm/flower-client/README.md \
    --values=helm/flower-client/values.yaml
  npx --yes --package=@bitnami/readme-generator-for-helm@2.7.2 readme-generator \
    --readme=helm/flower-server/README.md \
    --values=helm/flower-server/values.yaml
fi

# Core SQLAlchemy schema
paracelsus inject py/flwr/supercore/state/schema/README.md dev.get_schema_base:Base \
  --import-module "flwr.supercore.state.schema.linkstate_tables:*" \
  --import-module "flwr.supercore.state.schema.corestate_tables:*" \
  --import-module "flwr.supercore.state.schema.objectstore_tables:*" \
  --layout elk

# EE SQLAlchemy schema (if available)
if python -c "import flwr.ee.state.alembic.tables" 2>/dev/null; then
  paracelsus inject py/flwr/ee/state/schema/README.md dev.get_schema_base:EEBase \
    --import-module "flwr.ee.state.alembic.tables:*" \
    --layout elk
fi
