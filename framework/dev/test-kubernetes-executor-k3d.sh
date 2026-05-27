#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-kubernetes-executor-k3d.sh ==="

skip() {
    echo "SKIP: $1"
    exit 0
}

command -v docker >/dev/null 2>&1 || skip "docker is required for the optional k3d smoke harness."
docker info >/dev/null 2>&1 || skip "docker is installed, but the Docker daemon is not reachable."
command -v k3d >/dev/null 2>&1 || skip "k3d is required for the optional Kubernetes executor smoke harness."
command -v kubectl >/dev/null 2>&1 || skip "kubectl is required for the optional Kubernetes executor smoke harness."

if command -v uv >/dev/null 2>&1; then
    PYTHONPATH=py uv run --no-sync --with kubernetes python dev/kubernetes_executor_k3d_smoke.py "$@"
else
    echo "uv not found; using the current Python environment."
    echo "If this skips because the Kubernetes Python client is missing, run with uv or install the optional 'kubernetes' package locally."
    PYTHONPATH=py python dev/kubernetes_executor_k3d_smoke.py "$@"
fi
