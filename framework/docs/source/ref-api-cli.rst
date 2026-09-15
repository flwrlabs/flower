######################
 Flower CLI reference
######################

****************
 Basic Commands
****************

.. _flwr-apiref:

``flwr`` CLI
============

.. click:: flwr.cli.app:typer_click_object
    :prog: flwr
    :nested: full

.. _flower-superlink-apiref:

``flower-superlink``
====================

.. argparse::
    :module: flwr.superlink.cli.flower_superlink
    :func: _parse_args_run_superlink
    :prog: flower-superlink

.. _flower-supernode-apiref:

``flower-supernode``
====================

.. argparse::
    :module: flwr.supernode.cli.flower_supernode
    :func: _parse_args_run_supernode
    :prog: flower-supernode

*******************
 Advanced Commands
*******************

.. _flower-superexec-apiref:

``flower-superexec``
====================

.. argparse::
    :module: flwr.supercore.cli.flower_superexec
    :func: _parse_args
    :prog: flower-superexec

Warm executor resource overrides
--------------------------------

The Kubernetes executor applies ``resources`` to cold and warm TaskExecutor
Pods by default. Set ``warm-executor-resources`` to change only warm Pods. The
executor recursively merges this mapping over ``resources``, so fields omitted
from the warm override keep their base values. For example, this configuration
uses four CPUs for cold Pods and one CPU for warm Pods while both use the same
memory settings:

.. code-block:: yaml

    namespace: flower-system
    image: example.com/taskexecutor:latest
    warm-executor-owner: superexec-a
    warm-executor-pools:
      - task-type: flwr-agentapp
        size: 1
    resources:
      requests:
        cpu: "4"
        memory: 1Gi
      limits:
        cpu: "4"
        memory: 4Gi
    warm-executor-resources:
      requests:
        cpu: "1"
      limits:
        cpu: "1"

When the effective warm resource configuration changes, the executor replaces
incompatible idle warm Pods. A warm Pod that is already running a task remains
active until its task process exits. The warm resource settings remain on a Pod
after it accepts a task, so the one-CPU setting applies while that task runs.
