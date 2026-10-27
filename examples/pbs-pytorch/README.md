---
tags: [deployment, simulation, vision, PBS]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower on PBS: ResNet-18 and CIFAR-10

Train the same PyTorch Flower App in two configurations on four PBS GPU compute
nodes:

| Mode       | First physical node               | Other three physical nodes  | FL clients            |
| ---------- | --------------------------------- | --------------------------- | --------------------- |
| Deployment | SuperLink and ServerApp           | One real SuperNode per node | 3                     |
| Simulation | SuperLink, ServerApp and Ray head | One Ray worker per node     | 10 virtual SuperNodes |

Both use three FedAvg rounds, one complete local epoch per round, and all clients
participate in training and evaluation. The 50,000 CIFAR-10 training images and
10,000 test images are partitioned independently using seed 42, with no overlap
between clients. The model is torchvision's ResNet-18, initialized without
pretrained weights, with a 3x3 stride-one input convolution and no initial max
pool for 32x32 images. This is a deployment validation, not an accuracy benchmark.

## Prepare the environment

Use shared storage for this directory, the Python environment, dataset and outputs.
Use Python and CUDA-enabled PyTorch wheels matching the compute-node architecture
(for example, x86_64 or AArch64). An environment on shared storage remains
architecture-specific. Prepare a Python 3.11 environment using prebuilt wheels;
follow your site's policy for environment setup and source builds.

```bash
uv venv --python 3.11.14 /absolute/shared/path/venv
uv pip install --python /absolute/shared/path/venv/bin/python \
    --only-binary :all: 'flwr[simulation]==1.39.0'
uv pip install --python /absolute/shared/path/venv/bin/python \
    --only-binary :all: --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.10.0' 'torchvision==0.25.0'
```

To validate a development checkout, install `framework/` into this environment
inside a compute allocation and record the source commit. Keep the same
environment and source available until all queued jobs finish. No package is
installed at training time. CIFAR-10 is downloaded and checksum-verified by
`prepare_data.py` on the first compute node; compute nodes therefore need outbound
access for the first download, or an already populated torchvision CIFAR-10 cache.
For a short batch queue, populate that cache in a compute allocation before
submitting the training jobs, or select a walltime that also covers the first
download. Source-server throughput can make the download longer than training.

## Submit

The supplied scripts are PBS Professional/OpenPBS templates. Before submitting,
replace `gpu-queue` with your site's GPU queue and load the site's CUDA and
PBS-enabled Open MPI modules in both scripts. Inspect queue settings with
`qstat -Qf` and consult your site's resource and account policies.

The templates request four distinct, exclusive nodes with eight CPU slots, one
GPU and one MPI supervisor per node, using
`select=4:ncpus=8:mpiprocs=1:ngpus=1` and `place=scatter:exclhost`. GPU resource names
and allocation policies vary: replace `ngpus=1` with your site's GPU request, or
remove it if selecting the queue already allocates a GPU. Adjust CPU, memory and
walltime requests for your workload and add any required account option to `qsub`.
The launcher requires four distinct hosts and one nodefile entry per host.
See the [PBS reference guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSReferenceGuide2024.1.pdf)
for resource requests, host placement and queue status commands.

```bash
cd /absolute/path/to/flower/examples/pbs-pytorch
export PYTHON_BIN=/absolute/shared/path/venv/bin/python
export DATA_ROOT=/absolute/shared/path/cifar10
export OUTPUT_BASE=/absolute/shared/path/results
bash -n deployment.pbs simulation.pbs launch.sh
qsub -v PYTHON_BIN,DATA_ROOT,OUTPUT_BASE deployment.pbs
qsub -v PYTHON_BIN,DATA_ROOT,OUTPUT_BASE simulation.pbs
```

Submit from this directory because `PBS_O_WORKDIR` selects the App root. Paths
must not contain commas, which delimit PBS's `-v` environment list. The virtualenv
interpreter path must be absolute; retain its symlink rather than resolving it to
the base interpreter. Do not start training or Ray on a login node.

The deployment launcher starts the SuperLink Fleet API on the first allocated
node and connects one SuperNode from each other node. Their `partition-id` values
are 0, 1 and 2. Each SuperNode starts its ClientApp in the prepared environment.

The simulation launcher starts a Ray head with zero client CPU/GPU resources and
three workers with eight CPU slots and one GPU each. Each ClientApp requests two
CPU slots and 0.25 GPU, allowing four concurrent ClientApps per follower host and
twelve across the cluster. Ten virtual clients are scheduled onto this pool;
virtual clients do not correspond to persistent physical SuperNodes. The
`RAY_ADDRESS` inherited by the Simulation Runtime selects this cluster. GPU
fractions control scheduling rather than enforcing GPU-memory limits.

MPI only places the four supervisors. Flower uses its own network APIs; simulation
uses Ray for client scheduling. The supervisors retain per-host GPU visibility,
bound service startup and run waits, propagate failures to PBS, and stop their
services on completion or termination. Ray cleanup assumes **exclusive whole
nodes**; do not use these scripts unchanged for shared nodes or MIG allocations.

These examples use an unencrypted, unauthenticated Fleet API and Ray cluster on
the compute-node network. Use them only for testing in a trusted allocation.
For production deployment, configure Flower TLS and authentication as described
in the [deployment guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html),
and apply your site's Ray network security requirements. The Control and Runtime
HTTP APIs remain on each host's loopback interface.

## Check the results

Scheduler state alone does not prove training succeeded. Check the PBS exit
status and `$OUTPUT_BASE/<job-id>/<mode>/verification.json`. The verifier requires
four distinct physical hosts, all three follower hosts executing clients in
every round, all 3 or 10 clients participating without duplicates, all 50,000
training and 10,000 test examples processed per round, CUDA execution, changed
global model weights, and Flower's `finished:completed` run status.

The output directory contains:

- `placement-*.json`: physical hosts, interpreter, source paths and package versions.
- `events/*.json`: each client's round, partition, host, GPU, sample count and metrics.
- `result.json` and `final_model.pt`: aggregated metrics and global weights.
- `status.json`: final Flower run status.
- `ray-nodes.json`: live Ray nodes refreshed after training, immediately before
  final verification (simulation only).
- Per-rank service logs, dataset preparation log and MPI launcher log.

These local records contain hostnames, paths and device identifiers. Redact such
details before sharing logs or validation reports outside your cluster.

The verification runs on the allocated master before the job succeeds. To repeat
it later, use a compute allocation:

```bash
"$PYTHON_BIN" verify.py "$OUTPUT_BASE/<job-id>/deployment" deployment
"$PYTHON_BIN" verify.py "$OUTPUT_BASE/<job-id>/simulation" simulation
```

Inspect a failed rank's service log before resubmitting. Keep outputs from each
attempt in a different job directory; restarting the same mode within one job
would reuse readiness markers and is unsupported.
