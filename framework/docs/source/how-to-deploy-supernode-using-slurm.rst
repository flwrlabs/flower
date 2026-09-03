:og:description: Deploy Flower SuperNodes as Slurm jobs with subprocess isolation, process isolation, or GPU resources.
.. meta::
    :description: Deploy Flower SuperNodes as Slurm jobs with subprocess isolation, process isolation, or GPU resources.

###############################
 Deploy SuperNodes using Slurm
###############################

This guide shows how to deploy a SuperNode as a Slurm job with one of four deployment
patterns:

1. Default ``subprocess`` isolation
2. ``process`` isolation with a separate SuperExec
3. ``process`` isolation with a GPU-enabled SuperExec
4. ``process`` isolation with scheduler-selected nodes

Each example connects a SuperNode to SuperGrid. Before you continue, register a separate
key pair for every SuperNode that you plan to deploy. See :doc:`Connect SuperNodes to
SuperGrid <how-to-connect-supernodes-to-supergrid>` for instructions.

You will need:

- Access to a Slurm cluster and a compute partition
- :doc:`Flower installed <how-to-install-flower>` on the compute nodes
- A registered SuperNode private key that is readable from the compute node
- All ClientApp dependencies installed in the execution environment

In the examples, replace each value in angle brackets with the corresponding value for
your cluster. The first two process-isolation examples select one compute node that can
run both jobs at the same time. The final example lets Slurm place each job.

.. note::

    You can install ClientApp dependencies in advance or let Flower install them when an
    app starts. For the available options and their network requirements, see
    :doc:`Install Flower App dependencies at runtime
    <how-to-install-app-dependencies-at-runtime>`.

**********************************
 Use default subprocess isolation
**********************************

By default, the SuperNode starts each ClientApp as a subprocess. This model needs only
one Slurm job, and the command does not require an ``--isolation`` option.

Create ``supernode-subprocess.sbatch``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-supernode
    #SBATCH --partition=<partition>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=4G
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-supernode-%j.log

    set -Eeuo pipefail

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    exec flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key <path-to-private-key>

Submit the batch script to deploy the SuperNode:

.. code-block:: shell

    $ sbatch supernode-subprocess.sbatch

Each ClientApp process that the SuperNode starts can use the CPU and memory assigned to
this job.

***********************
 Use process isolation
***********************

With ``process`` isolation, Slurm manages the SuperNode and SuperExec as separate jobs.
The SuperNode receives tasks from SuperGrid, while the SuperExec connects to the
SuperNode Runtime API and starts the ClientApp processes.

The Runtime API in this example listens on ``127.0.0.1``, so both jobs must use the same
compute node.

.. warning::

    The process-isolation examples below are minimal configurations for a dedicated or
    otherwise trusted compute node. On a multi-tenant node, binding the Runtime API to
    ``127.0.0.1`` does not prevent other local jobs from connecting to it. Another local
    process could claim ClientApp tasks and access their inputs. Do not use these
    examples unchanged on a shared or production compute node. Security hardening for
    those environments is outside the scope of this guide.

Create ``supernode-process.sbatch``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-supernode
    #SBATCH --partition=<partition>
    #SBATCH --nodelist=<compute-node>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=1G
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-supernode-%j.log

    set -Eeuo pipefail

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    exec flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key <path-to-private-key> \
        --isolation process \
        --host 127.0.0.1 \
        --port 9094

Create ``superexec-clientapp.sbatch``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-superexec
    #SBATCH --partition=<partition>
    #SBATCH --nodelist=<compute-node>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=4G
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-superexec-%j.log

    set -Eeuo pipefail

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    # Wait up to 60 seconds for the SuperNode Runtime API.
    for _ in {1..30}; do
        if bash -c '</dev/tcp/127.0.0.1/9094' 2>/dev/null; then
            exec flower-superexec \
                --insecure \
                --plugin-type clientapp \
                --runtime-api-address 127.0.0.1:9094
        fi
        sleep 2
    done

    echo "SuperNode Runtime API did not start on 127.0.0.1:9094." >&2
    exit 1

Submit the SuperNode job first, then submit the SuperExec job:

.. code-block:: shell

    $ sbatch supernode-process.sbatch
    $ sbatch superexec-clientapp.sbatch

The SuperExec waits for ClientApp tasks from the SuperNode. The ClientApp processes
inherit the CPU and memory allocation of the SuperExec job.

.. note::

    ``--insecure`` applies only to the local Runtime API connection between SuperExec
    and SuperNode. The connection from SuperNode to SuperGrid still uses TLS. To protect
    Runtime API traffic outside a trusted host, configure TLS as described in
    :doc:`Enable TLS connections <how-to-enable-tls-connections>`.

********************************************
 Use process isolation with a GPU ClientApp
********************************************

To run a ClientApp on a GPU, use the ``supernode-process.sbatch`` script from the
previous section without adding a GPU request. The SuperNode coordinates the work but
does not execute the ClientApp, so it does not need access to the GPU.

Request the GPU only in the SuperExec job. The ClientApp process then inherits the
SuperExec environment, including ``CUDA_VISIBLE_DEVICES``. Configure both batch scripts
with the same ``<partition>`` and ``<compute-node>`` values so that they run together on
the GPU-capable node.

Create ``superexec-clientapp-gpu.sbatch``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-superexec-gpu
    #SBATCH --partition=<partition>
    #SBATCH --nodelist=<compute-node>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --gres=gpu:1
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-superexec-gpu-%j.log

    set -Eeuo pipefail

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "Slurm did not assign a GPU to this job." >&2
        exit 1
    fi

    # Wait up to 60 seconds for the SuperNode Runtime API.
    for _ in {1..30}; do
        if bash -c '</dev/tcp/127.0.0.1/9094' 2>/dev/null; then
            exec flower-superexec \
                --insecure \
                --plugin-type clientapp \
                --runtime-api-address 127.0.0.1:9094
        fi
        sleep 2
    done

    echo "SuperNode Runtime API did not start on 127.0.0.1:9094." >&2
    exit 1

Submit the unchanged process-isolated SuperNode job first, then submit the GPU SuperExec
job:

.. code-block:: shell

    $ sbatch supernode-process.sbatch
    $ sbatch superexec-clientapp-gpu.sbatch

The ClientApp environment must include a GPU-enabled version of its machine learning
framework and the required GPU libraries. For example, a PyTorch ClientApp can use the
device that Slurm exposes through ``CUDA_VISIBLE_DEVICES``.

************************************************
 Discover the SuperNode address from its job ID
************************************************

The preceding process-isolation examples use ``--nodelist`` to place both jobs on the
same node. You can remove this constraint when the compute nodes can reach each other.
In this model, a submission script passes the SuperNode job ID to a dependent SuperExec
job. After the SuperNode starts, SuperExec uses that ID to discover the assigned node's
address.

Create ``supernode-process-dynamic.sbatch``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-supernode
    #SBATCH --partition=<partition>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=1G
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-supernode-%j.log

    set -Eeuo pipefail

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    exec flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key <path-to-private-key> \
        --isolation process \
        --host 0.0.0.0 \
        --port 9094

The SuperNode listens on all network interfaces because Slurm can place SuperExec on a
different node. Create ``superexec-clientapp-dynamic.sbatch`` without a ``--nodelist``
directive:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=flower-superexec
    #SBATCH --partition=<partition>
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --gres=gpu:1
    #SBATCH --time=01:00:00
    #SBATCH --output=flower-superexec-%j.log

    set -Eeuo pipefail

    : "${SUPERNODE_JOB_ID:?SuperNode job ID is required.}"

    export FLWR_HOME="${SLURM_TMPDIR:-/tmp}/flower-${SLURM_JOB_ID}"

    node_list="$(squeue \
        --noheader \
        --jobs="${SUPERNODE_JOB_ID}" \
        --format='%N')"
    node_list="${node_list//[[:space:]]/}"

    if [[ -z "${node_list}" || "${node_list}" == "(null)" ]]; then
        echo "SuperNode job has no assigned node." >&2
        exit 1
    fi

    supernode_node="$(scontrol show hostnames "${node_list}" | head -n 1)"
    node_record="$(scontrol show node "${supernode_node}" --oneliner)"
    node_address="$(printf '%s\n' "${node_record}" \
        | tr ' ' '\n' \
        | awk -F= '$1 == "NodeAddr" {print $2; exit}')"

    if [[ -z "${node_address}" ]]; then
        echo "Could not resolve the SuperNode address." >&2
        exit 1
    fi

    runtime_api_address="${node_address}:9094"
    runtime_host="${runtime_api_address%:*}"
    runtime_port="${runtime_api_address##*:}"

    for _ in {1..30}; do
        if bash -c "</dev/tcp/${runtime_host}/${runtime_port}" 2>/dev/null; then
            exec flower-superexec \
                --insecure \
                --plugin-type clientapp \
                --runtime-api-address "${runtime_api_address}"
        fi
        sleep 2
    done

    echo "SuperNode Runtime API did not become ready." >&2
    exit 1

This example requests one GPU for SuperExec. Remove ``#SBATCH --gres=gpu:1`` when the
ClientApp does not need a GPU.

Create ``submit-process-pair.sh`` to submit both jobs:

.. code-block:: bash

    #!/usr/bin/env bash

    set -Eeuo pipefail

    supernode_job="$(sbatch --parsable supernode-process-dynamic.sbatch)"
    supernode_job_id="${supernode_job%%;*}"
    echo "Submitted SuperNode job ${supernode_job_id}."

    superexec_job="$(sbatch \
        --parsable \
        --dependency="after:${supernode_job_id}" \
        --export="ALL,SUPERNODE_JOB_ID=${supernode_job_id}" \
        superexec-clientapp-dynamic.sbatch)"

    echo "Submitted SuperExec job ${superexec_job%%;*}."

Run the submission script:

.. code-block:: shell

    $ bash submit-process-pair.sh

The first ``sbatch`` command returns the SuperNode job ID. The second job uses an
``after:<job-id>`` dependency, so Slurm holds it until the SuperNode job starts. The
SuperExec script then uses ``squeue`` and ``scontrol`` to resolve the assigned
``NodeAddr``. The short readiness loop accounts for the time between job startup and the
Runtime API becoming available.

.. warning::

    This minimal example exposes an unauthenticated and unencrypted Runtime API on the
    compute-node network. Use it only in a trusted, isolated test environment. Do not
    use it unchanged on a shared or production cluster. Security hardening for those
    environments is outside the scope of this guide.

For more information about the two isolation modes and the Runtime API, see :doc:`Flower
Network Communication <ref-flower-network-communication>`. After the SuperNode is
online, see :doc:`Run Flower Apps on SuperGrid <how-to-run-flower-apps-on-supergrid>`.
