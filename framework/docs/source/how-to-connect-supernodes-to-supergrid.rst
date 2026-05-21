:og:description: Connect SuperNodes to Flower SuperGrid by registering their public keys and starting them from Python or Docker.
.. meta::
    :description: Connect SuperNodes to Flower SuperGrid by registering their public keys and starting them from Python or Docker.

#################################
 Connect SuperNodes to SuperGrid
#################################

This guide shows how to register a SuperNode in SuperGrid and start the
``flower-supernode`` process so it can connect to SuperGrid. Once connected, the
SuperNode can participate in runs submitted to federations that include it.

You will need:

- Access to SuperGrid at https://flower.ai/supernodes/.
- A public/private key pair for each SuperNode you want to connect. This tutorial shows
  how to create these with ``ssh-keygen`` in the terminal.
- A machine where the SuperNode process can keep running.

.. note::

    To be able to connect a SuperNode to SuperGrid, you might need to first ask for
    access. Contact hello@flower.ai to request access.

************************************
 Register SuperNodes with SuperGrid
************************************

Each SuperNode uses its own key pair. The public key is registered with SuperGrid, and
the private key stays on the machine that runs the SuperNode.

Create a key pair for the first SuperNode:

.. code-block:: shell

    $ mkdir -p ~/.flwr/supernodes
    $ ssh-keygen -t ecdsa -b 384 -N "" -f ~/.flwr/supernodes/supernode-1

This creates two files:

- ``~/.flwr/supernodes/supernode-1``: the private key, used when starting the SuperNode.
- ``~/.flwr/supernodes/supernode-1.pub``: the public key, used when registering the
  SuperNode in SuperGrid.

Go to https://flower.ai/supernodes/ and click ``Register SuperNode``. In the dialog,
paste the contents of ``supernode-1.pub`` and click ``Register``. The SuperNode should
then appear in the list of registered SuperNodes.

To connect more SuperNodes, create and register a separate key pair for each one. Do not
reuse a key pair across multiple SuperNodes.

*********************************
 Connect SuperNodes to SuperGrid
*********************************

There are two common ways to run a SuperNode: directly from a Python environment, or
with the official SuperNode Docker image. In both cases, use the private key that
matches the public key you registered in SuperGrid.

.. note::

    The examples below use ``fleet-supergrid.flower.ai:443`` as the SuperGrid address.
    If the SuperGrid UI shows a different address or provides a complete connection
    command, use the value shown there.

Start from a Python environment
===============================

Install Flower in a Python environment on the machine that will run the SuperNode:

.. code-block:: shell

    $ pip install -U flwr

Start the SuperNode:

.. code-block:: shell

    $ flower-supernode \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key ~/.flwr/supernodes/supernode-1

Keep this process running for as long as you want the SuperNode to remain connected.

Start with Docker
=================

You can also run the SuperNode with the official Docker image. Mount the directory
containing the private key into the container, then pass the key path to
``flower-supernode``:

.. code-block:: shell

    $ docker run --rm \
        -v "$HOME/.flwr/supernodes:/keys:ro" \
        flwr/supernode:|stable_flwr_version| \
        --superlink fleet-supergrid.flower.ai:443 \
        --auth-supernode-private-key /keys/supernode-1

Use this form when you prefer to run SuperNodes from a container image instead of
installing Flower directly in a Python environment.

************************
 Check SuperNode status
************************

Return to https://flower.ai/supernodes/ after starting the SuperNode. Its status should
change to ``online``. If you stop the ``flower-supernode`` process or stop the Docker
container, the status will change to ``offline``. Starting it again with the same
private key will bring the same registered SuperNode back online.

.. note::

    SuperGrid does not collect or display the logs produced by your SuperNode. To debug
    startup or connection issues, inspect the terminal output or container logs on the
    machine where the SuperNode is running.
