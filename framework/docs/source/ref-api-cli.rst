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

Direct ASGI Startup
-------------------

When starting the SuperLink ASGI app directly with Uvicorn, set
``FLWR_SUPEREXEC_AUTH_SECRET_FILE`` to enable shared-secret authentication for SuperExec
task acquisition on the Runtime API:

.. code-block:: bash

    FLWR_SUPEREXEC_AUTH_SECRET_FILE=/path/to/superexec-secret \
        uvicorn flwr.superlink.main:app --host 127.0.0.1 --port 8000

The file is read as exact raw bytes, including any trailing newline. Configure the
connecting ``flower-superexec`` with ``--superexec-auth-secret-file`` pointing to a file
containing the same bytes. An unreadable or empty file prevents the ASGI app from
starting. If the environment variable is unset, shared-secret authentication remains
disabled.

This environment variable applies to direct ASGI startup. When using
``flower-superlink``, the CLI configuration takes precedence, including when no
shared-secret file is specified.

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
