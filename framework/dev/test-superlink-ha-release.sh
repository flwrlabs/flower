#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

WITH_POSTGRES=false
RUN_RUFF=true
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:15-alpine}"
POSTGRES_PORT="${POSTGRES_PORT:-55432}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-flwr-ha-test-postgres-$$}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-flwr}"
POSTGRES_DB="${POSTGRES_DB:-flwr}"
POSTGRES_STARTED=false

usage() {
  cat <<EOF
Usage: dev/test-superlink-ha-release.sh [--with-postgres] [--skip-ruff]

Runs the focused SuperLink HA release regression gate.

Options:
  --with-postgres  Start a disposable local PostgreSQL container and run the
                   real PostgreSQL migration smoke test.
  --skip-ruff      Skip ruff on the touched test files.
  -h, --help       Show this help text.

Environment:
  PYTHON_VERSION       Python version passed to uv (default: 3.11)
  POSTGRES_IMAGE       Docker image for --with-postgres (default: postgres:15-alpine)
  POSTGRES_PORT        Local PostgreSQL port for --with-postgres (default: 55432)
  POSTGRES_CONTAINER   Temporary container name
EOF
}

cleanup() {
  if [ "$POSTGRES_STARTED" = true ]; then
    docker stop "$POSTGRES_CONTAINER" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-postgres)
      WITH_POSTGRES=true
      ;;
    --skip-ruff)
      RUN_RUFF=false
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

PYTEST_TARGETS=(
  "py/flwr/server/superlink/linkstate/linkstate_test.py"
  "py/flwr/server/superlink/serverappio/serverappio_servicer_test.py"
  "py/flwr/supercore/object_store/object_store_test.py"
  "py/flwr/supercore/state/alembic/utils_test.py"
)

RUFF_TARGETS=(
  "py/flwr/server/superlink/linkstate/linkstate_test.py"
  "py/flwr/supercore/state/alembic/utils_test.py"
)

POSTGRES_TEST_TARGET=(
  "py/flwr/supercore/state/alembic/utils_test.py::TestAlembicRun::test_run_migrations_on_real_disposable_postgresql"
)

echo "=== SuperLink HA release regression gate ==="
echo "Python: $PYTHON_VERSION"
echo "PostgreSQL smoke test: $WITH_POSTGRES"

echo
echo "- Running focused HA pytest targets"
uv run --python="$PYTHON_VERSION" python -m pytest -q "${PYTEST_TARGETS[@]}"

if [ "$RUN_RUFF" = true ]; then
  echo
  echo "- Running ruff on HA test changes"
  uv run --python="$PYTHON_VERSION" python -m ruff check "${RUFF_TARGETS[@]}"
fi

if [ "$WITH_POSTGRES" = true ]; then
  echo
  echo "- Starting disposable PostgreSQL container: $POSTGRES_CONTAINER"
  docker run --rm -d \
    --name "$POSTGRES_CONTAINER" \
    -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    -e POSTGRES_DB="$POSTGRES_DB" \
    -p "127.0.0.1:$POSTGRES_PORT:5432" \
    "$POSTGRES_IMAGE" >/dev/null
  POSTGRES_STARTED=true

  for _ in $(seq 1 30); do
    if docker exec "$POSTGRES_CONTAINER" pg_isready \
      -U postgres -d "$POSTGRES_DB" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  docker exec "$POSTGRES_CONTAINER" pg_isready -U postgres -d "$POSTGRES_DB"

  echo
  echo "- Running real PostgreSQL migration smoke test"
  FLWR_TEST_POSTGRES_DISPOSABLE_URL="postgresql+psycopg://postgres:$POSTGRES_PASSWORD@127.0.0.1:$POSTGRES_PORT/$POSTGRES_DB" \
    uv run --with "psycopg[binary]" --python="$PYTHON_VERSION" \
    python -m pytest -q "${POSTGRES_TEST_TARGET[@]}"
fi

echo
echo "SuperLink HA release regression gate passed."
