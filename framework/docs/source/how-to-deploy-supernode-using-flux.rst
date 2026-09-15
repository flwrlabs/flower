:og:description: Deploy Flower SuperNodes as Flux jobs with subprocess isolation, process isolation, or GPU resources.
.. meta::
    :description: Deploy Flower SuperNodes as Flux jobs with subprocess isolation, process isolation, or GPU resources.

##############################
 Deploy SuperNodes using Flux
##############################

This guide shows how to deploy a SuperNode as a Flux job with one of three deployment
patterns:

1. Default ``subprocess`` isolation
2. ``process`` isolation with a SuperExec in the same Flux allocation
3. ``process`` isolation with GPU resources for the ClientApp

Each example connects a SuperNode to SuperGrid. Before you continue, register a separate
key pair for every SuperNode that you plan to deploy. See :doc:`Connect SuperNodes to
SuperGrid <how-to-connect-supernodes-to-supergrid>` for instructions.

You will need:

- Access to a Flux instance with permission to run jobs
- :doc:`Flower installed <how-to-install-flower>` on the compute nodes
- A registered SuperNode private key that is readable from the compute node
- All ClientApp dependencies installed in the execution environment

In the examples, replace each value in angle brackets with the corresponding value for
your cluster. Flux exports the submission environment by default, so the scripts pass
their Flower configuration through environment variables.

.. note::

    You can install ClientApp dependencies in advance or let Flower install them when an
    app starts. For the available options and their network requirements, see
    :doc:`Install Flower App dependencies at runtime
    <how-to-install-app-dependencies-at-runtime>`.

**********************************
 Use default subprocess isolation
**********************************

By default, the SuperNode starts each ClientApp as a subprocess. This model needs only
one Flux job, and the command does not require an ``--isolation`` option.

Download :download:`supernode-subprocess.sh
<_static/flux/supernode-subprocess.sh>` and make it executable:

.. code-block:: shell

    $ chmod +x supernode-subprocess.sh

The launcher contains the following command:

.. literalinclude:: _static/flux/supernode-subprocess.sh
    :language: bash

Export the SuperGrid Fleet API address and the registered private key, then submit the
job with ``flux run``:

.. code-block:: shell

    $ export FLWR_SUPERLINK_ADDRESS="fleet-supergrid.flower.ai:443"
    $ export FLWR_SUPERNODE_PRIVATE_KEY="<path-to-private-key>"
    $ flux run \
        -N1 \
        -n1 \
        -c4 \
        --job-name=flower-supernode \
        --time-limit=1h \
        --output=flower-supernode-{{id}}.log \
        --error=flower-supernode-{{id}}.err \
        ./supernode-subprocess.sh

The ``flux run`` command remains attached to the job. Press :kbd:`Ctrl+C` to cancel the
job, or use ``flux job cancel <job-id>`` from another terminal. Each ClientApp process
that the SuperNode starts can use the CPU and memory assigned to this job.

The launcher gives every Flux job a separate ``FLWR_HOME`` under
``FLUX_JOB_TMPDIR``. This prevents multiple SuperNodes on a shared filesystem from
overwriting each other's installed Flower Apps and local state.

***********************
 Use process isolation
***********************

With ``process`` isolation, the SuperNode receives tasks from SuperGrid and a SuperExec
starts the ClientApp processes. Run both services inside one Flux allocation so that
they share the assigned node and the Runtime API can remain bound to ``127.0.0.1``.

.. warning::

    This is a minimal configuration for a dedicated or otherwise trusted compute node.
    On a multi-tenant node, binding the Runtime API to ``127.0.0.1`` does not prevent
    other local jobs from connecting to it. Another local process could claim ClientApp
    tasks and access their inputs. Do not use this example unchanged on a shared or
    production compute node. Security hardening for those environments is outside the
    scope of this guide.

Create ``supernode-process.sh``:

.. code-block:: bash

    #!/usr/bin/env bash

    set -Eeuo pipefail

    : "${FLWR_SUPERLINK_ADDRESS:?Set FLWR_SUPERLINK_ADDRESS.}"
    : "${FLWR_SUPERNODE_PRIVATE_KEY:?Set FLWR_SUPERNODE_PRIVATE_KEY.}"

    job_root="${FLUX_JOB_TMPDIR:-/tmp}/flower-${FLUX_JOB_ID}"
    runtime_address="127.0.0.1:9094"
    supernode_pid=""

    cleanup() {
        if [[ -n "${supernode_pid}" ]]; then
            kill "${supernode_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    FLWR_HOME="${job_root}/supernode" \
        flower-supernode \
        --superlink "${FLWR_SUPERLINK_ADDRESS}" \
        --auth-supernode-private-key "${FLWR_SUPERNODE_PRIVATE_KEY}" \
        --isolation process \
        --host 127.0.0.1 \
        --port 9094 &
    supernode_pid=$!

    # Wait up to 60 seconds for the SuperNode Runtime API.
    for _ in {1..30}; do
        if bash -c '</dev/tcp/127.0.0.1/9094' 2>/dev/null; then
            FLWR_HOME="${job_root}/superexec" \
                flower-superexec \
                --insecure \
                --plugin-type clientapp \
                --runtime-api-address "${runtime_address}"
            exit $?
        fi
        sleep 2
    done

    echo "SuperNode Runtime API did not start on ${runtime_address}." >&2
    exit 1

Submit the script in one allocation:

.. code-block:: shell

    $ chmod +x supernode-process.sh
    $ export FLWR_SUPERLINK_ADDRESS="fleet-supergrid.flower.ai:443"
    $ export FLWR_SUPERNODE_PRIVATE_KEY="<path-to-private-key>"
    $ flux run \
        -N1 \
        -n1 \
        -c4 \
        --job-name=flower-supernode-process \
        --time-limit=1h \
        --output=flower-supernode-process-{{id}}.log \
        --error=flower-supernode-process-{{id}}.err \
        ./supernode-process.sh

The ClientApp processes inherit the resources and environment of this allocation.

.. note::

    ``--insecure`` applies only to the local Runtime API connection between SuperExec
    and SuperNode. The connection from SuperNode to SuperGrid still uses TLS. To protect
    Runtime API traffic outside a trusted host, configure TLS as described in
    :doc:`Enable TLS connections <how-to-enable-tls-connections>`.

********************************************
 Use process isolation with a GPU ClientApp
********************************************

To run the ClientApp on a GPU, submit the process-isolation script with a GPU request:

.. code-block:: shell

    $ flux run \
        -N1 \
        -n1 \
        -c4 \
        -g1 \
        --job-name=flower-supernode-gpu \
        --time-limit=1h \
        --output=flower-supernode-gpu-{{id}}.log \
        --error=flower-supernode-gpu-{{id}}.err \
        ./supernode-process.sh

The ClientApp environment inherits the GPU visibility configured by Flux. It must also
include a GPU-enabled version of its machine learning framework and the required GPU
libraries. On clusters that allocate GPUs per node instead of per task, replace ``-g1``
with ``--gpus-per-node=1``.

*******************************************
 Launch Flux training from the ClientApp
*******************************************

Some deployments keep the SuperNode lightweight and let the ClientApp request the
resources for each training invocation. For example, a ClientApp can run a distributed
training command synchronously:

.. code-block:: python

    import subprocess

    subprocess.run(
        [
            "flux",
            "run",
            "-N4",
            "-n32",
            "--gpus-per-node=8",
            "python",
            "train.py",
        ],
        check=True,
    )

Use ``flux run`` rather than ``flux batch`` when the ClientApp needs the training result
before returning its reply to the SuperNode. ``flux run`` waits for the job to finish,
so the ClientApp can read the generated checkpoint or metrics and include them in its
Flower reply. A detached batch submission would return before those outputs exist.

Pass site-specific Flux options such as ``--queue``, ``--bank``, and ``--time-limit``
through the ClientApp configuration instead of hard-coding them. This lets each
SuperNode use the queue and account assigned to its site.

For more information about the two isolation modes and the Runtime API, see :doc:`Flower
Network Communication <ref-flower-network-communication>`. After the SuperNode is
online, see :doc:`Run Flower Apps on SuperGrid
<how-to-run-flower-apps-on-supergrid>`.
