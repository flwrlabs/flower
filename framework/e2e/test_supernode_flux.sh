#!/usr/bin/env bash

set -Eeuo pipefail

e2e_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="${e2e_dir}/e2e-bare"
launcher="${e2e_dir}/../docs/source/_static/flux/supernode-subprocess.sh"
flux_bin="${FLWR_E2E_FLUX_BIN:-${e2e_dir}/fixtures/flux}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/flower-flux-e2e.XXXXXX")"
pyproject_backup="${tmp_dir}/pyproject.toml"

superlink_pid=""
supernode_pids=()
app_flwr_existed=false
if [[ -e "${app_dir}/.flwr" ]]; then
    app_flwr_existed=true
fi

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    for pid in "${supernode_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${superlink_pid}" ]]; then
        kill "${superlink_pid}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true

    if [[ -f "${pyproject_backup}" ]]; then
        cp "${pyproject_backup}" "${app_dir}/pyproject.toml"
    fi
    if [[ "${app_flwr_existed}" == false ]]; then
        rm -rf "${app_dir}/.flwr"
    fi
    rm -rf "${tmp_dir}"
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

read -r serverappio_port fleet_port control_port clientappio_port_1 clientappio_port_2 < <(
    python -c '
import socket

sockets = [socket.socket() for _ in range(5)]
for sock in sockets:
    sock.bind(("127.0.0.1", 0))
print(*(sock.getsockname()[1] for sock in sockets))
for sock in sockets:
    sock.close()
'
)

supernode_help="$(flower-supernode --help)"

if [[ ! -x "${flux_bin}" ]]; then
    echo "Flux executable is not available: ${flux_bin}" >&2
    echo "Set FLWR_E2E_FLUX_BIN to a real Flux executable or use the bundled fixture." >&2
    exit 2
fi

cp "${app_dir}/pyproject.toml" "${pyproject_backup}"
sed -i.bak '/^\[tool\.flwr\.federations\.e2e\]/,/^$/d' "${app_dir}/pyproject.toml"
rm -f "${app_dir}/pyproject.toml.bak"
cat >>"${app_dir}/pyproject.toml" <<EOF

[tool.flwr.federations.e2e]
address = "127.0.0.1:${control_port}"
insecure = true
EOF

FLWR_HOME="${tmp_dir}/superlink" flower-superlink \
    --insecure \
    --database :flwr-in-memory: \
    --serverappio-api-address "127.0.0.1:${serverappio_port}" \
    --fleet-api-address "127.0.0.1:${fleet_port}" \
    --control-api-address "127.0.0.1:${control_port}" \
    >"${tmp_dir}/superlink.log" 2>&1 &
superlink_pid=$!
sleep 3

for index in 1 2; do
    if [[ "${index}" == 1 ]]; then
        clientappio_port="${clientappio_port_1}"
    else
        clientappio_port="${clientappio_port_2}"
    fi

    if grep -q -- '--clientappio-api-address' <<<"${supernode_help}"; then
        runtime_args=(
            --clientappio-api-address "127.0.0.1:${clientappio_port}"
        )
    else
        runtime_args=(
            --host 127.0.0.1
            --port "${clientappio_port}"
        )
    fi

    FLWR_SUPERLINK_ADDRESS="127.0.0.1:${fleet_port}" \
    FLWR_INSECURE=true \
    FLUX_JOB_TMPDIR="${tmp_dir}/supernode-${index}" \
        "${flux_bin}" run \
        -N1 \
        -n1 \
        -c2 \
        --job-name="flower-supernode-${index}" \
        --time-limit=5m \
        --output="${tmp_dir}/supernode-${index}.out" \
        --error="${tmp_dir}/supernode-${index}.err" \
        "${launcher}" \
        "${runtime_args[@]}" \
        --max-retries 0 &
    supernode_pids+=("$!")
done
sleep 5

(
    cd "${app_dir}"
    FLWR_HOME="${tmp_dir}/cli" flwr run . e2e
)

deadline=$((SECONDS + 240))
while ((SECONDS < deadline)); do
    status="$({ cd "${app_dir}" && FLWR_HOME="${tmp_dir}/cli" flwr ls . e2e --format=json; } \
        | jq -r '.runs[0].status')"
    echo "Current status: ${status}"

    if [[ "${status}" == "finished:completed" ]]; then
        echo "Flux-scheduled SuperNodes completed the Flower run."
        exit 0
    fi
    if [[ "${status}" == finished:* ]]; then
        echo "Flower run ended with status ${status}." >&2
        exit 1
    fi
    sleep 2
done

echo "Flower run did not complete within 240 seconds." >&2
exit 1
