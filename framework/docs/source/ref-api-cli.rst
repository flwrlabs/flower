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
    :module: flwr.server.app
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
