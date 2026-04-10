#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

taplo fmt

# Python
python -m devtool.check_copyright py/flwr
python -m devtool.init_py_fix py/flwr
python -m isort --skip py/flwr/proto py
python -m black -q --exclude py/flwr/proto py
python -m docformatter -i -r py/flwr -e py/flwr/proto
python -m ruff check --fix py/flwr

# Protos
find proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format -i

# E2E
python -m isort e2e
python -m black -q e2e
python -m docformatter -i -r e2e

# Markdown
python -m mdformat --number docs/source

# RST
docstrfmt docs/source

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
