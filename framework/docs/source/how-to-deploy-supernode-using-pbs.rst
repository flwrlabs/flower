:og:description: Deploy Flower SuperNodes with PBS, including process isolation, GPU ClientApps, and four-node deployment and simulation examples.
.. meta::
    :description: Deploy Flower SuperNodes with PBS, including process isolation, GPU ClientApps, and four-node deployment and simulation examples.

#############################
 Deploy SuperNodes using PBS
#############################

This guide shows how to deploy a Flower SuperNode as a PBS batch job, using either
default subprocess isolation or process isolation with a separate SuperExec. It also
links to a four-node PyTorch example for a self-hosted SuperLink and multi-node
simulation.

You will need:

- Access to a PBS cluster and an eligible batch queue
- :doc:`Flower installed <how-to-install-flower>` in an environment readable from the
  compute nodes, together with your ClientApp dependencies
- For SuperGrid, a separate registered private key for each SuperNode; see
  :doc:`how-to-connect-supernodes-to-supergrid`
- Outbound connectivity to SuperGrid, or compute-node connectivity to your own SuperLink

Replace values in angle brackets with your cluster's values. PBS resource and accounting
syntax varies by site. The single-node examples below use PBS Professional or OpenPBS
resource syntax; consult your site's documentation for queue, account, CPU, memory and
GPU requests. Install dependencies before submission when compute nodes cannot download
packages. See :doc:`how-to-install-app-dependencies-at-runtime` for runtime installation
options.

**********************************
 Use default subprocess isolation
**********************************

By default, a SuperNode starts ClientApps as subprocesses. One PBS job therefore
provides resources for both the SuperNode and its ClientApps.

Create ``supernode-subprocess.pbs``:

.. code-block:: bash

    #!/bin/bash
    #PBS -N flower-supernode
    #PBS -q <queue>
    #PBS -l select=1:ncpus=4:mem=4gb
    #PBS -l walltime=01:00:00
    #PBS -j oe

    set -Eeuo pipefail
    : "${PBS_JOBID:?Run inside a PBS job}"
    PYTHON_BIN="<absolute-virtualenv-path>/bin/python"
    export PATH="$(dirname "$PYTHON_BIN"):$PATH"
    export FLWR_HOME="${TMPDIR:-/tmp}/flower-${PBS_JOBID}"

    exec flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key <absolute-path-to-private-key>

Submit from a directory on shared storage:

.. code-block:: shell

    qsub supernode-subprocess.pbs

PBS copies the batch script at submission, but the environment, key and App files must
remain accessible when the job starts. PBS directives require literal values; shell
variables in ``#PBS`` lines are not expanded. Use ``#PBS -o`` or ``qsub -o`` with a
site-supported output path to choose where PBS retains the combined log.

***********************
 Use process isolation
***********************

In process isolation, the SuperNode coordinates work and a separate SuperExec starts
ClientApps. Run both services in the **same single-node allocation** so that the
SuperNode Runtime API can stay on loopback. Two independent PBS jobs are not guaranteed
to run on the same host or at the same time.

Create ``supernode-process.pbs`` with the same PBS directives and environment setup as
above, replacing the ``exec flower-supernode`` command with:

.. code-block:: bash

    pids=()
    cleanup() {
        if ((${#pids[@]})); then
            kill "${pids[@]}" 2>/dev/null || true
            wait "${pids[@]}" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT
    trap 'exit 1' INT TERM

    flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key <absolute-path-to-private-key> \
        --isolation process --host 127.0.0.1 --port 9094 &
    supernode_pid=$!
    pids+=("$supernode_pid")

    ready=false
    for _ in {1..30}; do
        kill -0 "$supernode_pid" 2>/dev/null || {
            echo "SuperNode exited before its Runtime API was ready" >&2
            exit 1
        }
        if bash -c '</dev/tcp/127.0.0.1/9094' 2>/dev/null; then
            ready=true
            break
        fi
        sleep 2
    done
    "$ready" || { echo "Runtime API did not become ready" >&2; exit 1; }

    flower-superexec --insecure \
        --runtime-api-address 127.0.0.1:9094 &
    pids+=("$!")
    # Either service exiting is unexpected for this long-lived deployment.
    wait -n "${pids[@]}" || exit 1
    exit 1

Submit the combined job with ``qsub supernode-process.pbs``. Both services and the
ClientApps share its CPU and memory allocation. The trap terminates the remaining
service if one exits or PBS terminates the shell. PBS must also clean up the job's
descendant processes on termination.

.. note::

    ``--insecure`` here applies to the local Runtime API connection. The SuperNode's
    connection to SuperGrid still uses TLS. Loopback does not protect the Runtime API
    against other users' processes on a shared compute node. Use these minimal
    process-isolation examples only on a trusted allocation; see
    :doc:`how-to-enable-tls-connections` for TLS configuration.

********************************************
 Use process isolation with a GPU ClientApp
********************************************

The SuperExec and ClientApps inherit the GPU devices assigned to their PBS job. Add your
site's GPU request to the process-isolation script. For a PBS site configured with an
``ngpus`` resource, replace the select directive with:

.. code-block:: bash

    #PBS -l select=1:ncpus=4:mem=8gb:ngpus=1

This syntax is site-dependent. Some clusters assign GPUs through the selected queue and
node allocation rather than an ``ngpus`` request. Load the site's required CUDA/runtime
modules and install a GPU-enabled ML framework that matches the compute-node
architecture. Do not overwrite the scheduler's ``CUDA_VISIBLE_DEVICES`` with another
node's device identifiers.

*************************************************
 Run a four-node deployment or simulation on PBS
*************************************************

The `PBS PyTorch example
<https://github.com/flwrlabs/flower/tree/main/examples/pbs-pytorch>`_ includes complete
scripts for ResNet-18 classification on CIFAR-10 in two configurations:

- **Deployment:** one SuperLink and ServerApp on the first physical node, with three
  real SuperNodes, one on each remaining physical node.
- **Simulation:** one SuperLink, ServerApp and Ray head on the first physical node, with
  ten virtual SuperNodes scheduled on three physical Ray worker nodes.

The example uses four exclusive GPU compute nodes and PBS-aware Open MPI to place one
service supervisor per node. MPI places processes; Flower and Ray perform their own
network communication. The launcher discovers the first host from ``PBS_NODEFILE``
rather than hardcoding a compute-node name. It checks that four distinct nodes were
allocated, waits for services to start, propagates failures and retains per-node logs
and training evidence on shared storage.

Inspect queue settings with ``qstat -Qf`` and consult your site's resource and account
policies. From the example directory, set absolute shared paths and submit:

.. code-block:: shell

    export PYTHON_BIN=<absolute-shared-virtualenv-path>/bin/python
    export DATA_ROOT=<absolute-shared-cifar10-path>
    export OUTPUT_BASE=<absolute-shared-results-path>
    qsub -v PYTHON_BIN,DATA_ROOT,OUTPUT_BASE deployment.pbs
    qsub -v PYTHON_BIN,DATA_ROOT,OUTPUT_BASE simulation.pbs

The example scripts use ``gpu-queue`` as a placeholder queue name and request
``select=4:ncpus=8:mpiprocs=1:ngpus=1`` with ``place=scatter:exclhost`` and a 30-minute
walltime. Replace the queue name, GPU resource request and module setup with your site's
settings, and add any required account option to ``qsub``. Adjust CPU, memory and
walltime requests as needed. The launcher requires four distinct hosts with one nodefile
entry per host. Install Python and GPU packages matching the compute-node architecture;
shared storage does not make a virtualenv portable across architectures.

Follow the example's README for installation, network requirements and retained
artifacts. A successful scheduler start is insufficient: check the final PBS exit status
and ``verification.json``. The verifier requires every configured client to complete
three training and evaluation rounds using CUDA, all three follower hosts to execute
clients, and Flower to report ``finished:completed``.

.. warning::

    The self-hosted example uses unencrypted, unauthenticated Fleet and Ray traffic on
    the compute-node network. Use it only in a trusted test allocation. For production,
    configure :doc:`Flower TLS <how-to-enable-tls-connections>` and :doc:`SuperNode
    authentication <how-to-authenticate-supernodes>`, and follow your site's Ray
    security requirements. Its Ray cleanup assumes exclusive whole nodes.

See :doc:`ref-flower-network-communication` for isolation modes and API connections, and
:ref:`multinodesimulations` for how the Simulation Runtime uses a Ray cluster.
