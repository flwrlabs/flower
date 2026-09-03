:og:description: Deploy Flower SuperNodes as Slurm jobs with subprocess isolation, process isolation, or GPU resources.
.. meta::
    :description: Deploy Flower SuperNodes as Slurm jobs with subprocess isolation, process isolation, or GPU resources.

###############################
 Deploy SuperNodes using Slurm
###############################

This guide shows how to deploy a SuperNode as a Slurm job with one of three execution
models:

1. Default ``subprocess`` isolation
2. ``process`` isolation with a separate SuperExec
3. ``process`` isolation with a GPU-enabled SuperExec

Each example connects a SuperNode to SuperGrid. Before you continue, register a separate
key pair for every SuperNode that you plan to deploy. See :doc:`Connect SuperNodes to
SuperGrid <how-to-connect-supernodes-to-supergrid>` for instructions.

You will need:

- Access to a Slurm cluster and a compute partition
- :doc:`Flower installed <how-to-install-flower>` on the compute nodes
- A registered SuperNode private key that is readable from the compute node
- All ClientApp dependencies installed in the execution environment

In the examples, replace each value in angle brackets with the corresponding value for
your cluster. When you use process isolation, select one compute node that can run both
jobs at the same time.

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
    and SuperNode. The connection from SuperNode to SuperGrid still uses TLS. If the
    Runtime API crosses a trusted-host boundary, configure TLS as described in
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

For more information about the two isolation modes and the Runtime API, see :doc:`Flower
Network Communication <ref-flower-network-communication>`. After the SuperNode is
online, see :doc:`Run Flower Apps on SuperGrid <how-to-run-flower-apps-on-supergrid>`.
