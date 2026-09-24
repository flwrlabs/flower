:og:description: Learn how to register SuperNodes to the Flower SuperGrid with authentication keys using this comprehensive step-by-step guide.
.. meta::
    :description: Learn how to register SuperNodes to the Flower SuperGrid with authentication keys using this comprehensive step-by-step guide.

#############################################
 Register SuperNodes on the Flower SuperGrid
#############################################

This guide provides a complete, step-by-step procedure for registering SuperNodes to a
SuperGrid utilizing authentication keys.

***************
 Prerequisites
***************

Before initiating the registration process, the following prerequisites must be
fulfilled:

Flower Account Creation
=======================

Navigate to the `flower.ai <https://flower.ai>`_ website, select the **Sign Up / Sign
In** option located in the upper right quadrant of the landing page, and complete the
required registration procedure.

Federations Dashboard Access
============================

Proceed to the `Federations Dashboard <https://flower.ai/federations>`_. This dashboard
serves as the central repository for your defined federation names, a list of associated
members, currently registered SuperNodes, and a history of previous workflow executions.

Required Tools Installation
===========================

Verify the installation and operational status of the following essential tools:

.. code-block:: bash

    # Install Flower Command Line Interface (CLI)
    pip install flwr>=1.24.0

    # Verify Docker daemon is running
    docker --version

*******************************
 Authentication Key Generation
*******************************

SuperNode registration mandates the creation of authentication keys, which is
accomplished via the official Flower authentication example.

Certificate Configuration Adjustment
====================================

Flower authentication example can be cloned from the `Flower GitHub repository
<https://github.com/adap/flower/tree/main/examples/flower-authentication>`_. The
``certificate.conf`` file in this example requires modification to incorporate SuperGrid
DNS support. This specific line is not included by default in the Flower GitHub
repository.

Open ``certificate.conf`` and insert ``DNS.2 = fleet-supergrid.flower.ai`` as follows:

.. code-block:: ini

    [req]
    default_bits = 4096
    prompt = no
    default_md = sha256
    req_extensions = req_ext
    distinguished_name = dn

    [dn]
    C = DE
    ST = HH
    O = Flower
    CN = localhost

    [req_ext]
    subjectAltName = @alt_names

    [alt_names]
    DNS.1 = localhost
    DNS.2 = fleet-supergrid.flower.ai    # ← ADD THIS LINE
    IP.1 = ::1
    IP.2 = 127.0.0.1

Number of Keys Specification
============================

Edit the ``generate_auth_keys.sh`` script (specifically line 15) to define the exact
quantity of SuperNode key pairs to be generated:

.. code-block:: bash

    generate_client_credentials() {
        local num_clients=${1:-<NUM_SUPERNODES>}    # ← Line 15: Set number here
        for ((i=1; i<=num_clients; i++))
        do
            ssh-keygen -t ecdsa -b 384 -N "" -f "${KEY_DIR}/client_credentials_$i" -C ""
        done
    }

**Example:** To produce 5 key pairs, replace ``<NUM_SUPERNODES>`` with the numeral
``5``.

Execution of Key Generation Script
==================================

Execute the script to create your authentication keys:

.. code-block:: bash

    bash <PATH>/generate_auth_keys.sh

This operation populates the ``/keys`` directory with key pairs, such as
``client_credentials_1`` (the private key) and ``client_credentials_1.pub`` (the public
key). Subsequent pairs, such as ``client_credentials_2`` / ``client_credentials_2.pub``,
will also be generated.

.. warning::

    **IMPORTANT SECURITY NOTE:**

    The generated keys must be stored in secure, protected storage. The script should
    **NOT** be executed multiple times without first backing up existing keys, as this
    action will overwrite them. Immediate backup of all generated keys is mandatory.

***********************
 Project Configuration
***********************

Incorporate the following configuration block into your project's ``pyproject.toml``
file:

.. code-block:: toml

    [tool.flwr.federations.supergrid]
    address = "supergrid.flower.ai"
    federation = "<FEDERATION_NAME>"    # ← Replace with your federation name
    enable-account-auth = true

.. note::

    Substitute ``<FEDERATION_NAME>`` with the actual federation name retrieved from your
    `flower.ai/federations <https://flower.ai/federations>`_ dashboard (e.g.,
    ``"flowerlabs"``).

*****************
 SuperGrid Login
*****************

Login Command
=============

Run the login command from your project's root directory:

.. code-block:: bash

    flwr login . supergrid

A browser prompt will request authorization to grant access to Flower SuperGrid. Select
**"Yes"** to confirm authorization.

Login Verification
==================

Confirm successful login status by executing the following commands:

.. code-block:: bash

    # View all runs associated with your federation
    flwr list . supergrid

    # View detailed account, node, and run IDs
    flwr federation show . supergrid

The expected output is the display of your federation information and any existing
workflow runs.

************************
 SuperNode Registration
************************

Each SuperNode is registered using its corresponding public key:

.. code-block:: bash

    flwr supernode register <PATH>/keys/client_credentials_1.pub . supergrid

Node ID Preservation
====================

The registration command output will include a unique Node ID.

.. important::

    **CRITICAL ACTION REQUIRED:**

    Record each Node ID immediately, and inform the designated Flower Labs team members
    by sharing the Node IDs. The Flower Labs team is for now responsible for registering
    these SuperNodes on the backend system.

Registration of Multiple SuperNodes
===================================

Repeat the registration process for every remaining key pair, for example:

.. code-block:: bash

    # Registration of SuperNode 2
    flwr supernode register <PATH>/keys/client_credentials_2.pub . supergrid

    # Registration of SuperNode 3
    flwr supernode register <PATH>/keys/client_credentials_3.pub . supergrid

    # Registration of SuperNode 4
    flwr supernode register <PATH>/keys/client_credentials_4.pub . supergrid

Confirmation Protocol
=====================

Following the submission of Node IDs to the Flower Labs team, await formal confirmation
that the SuperNodes have been registered. Verify the status on the
`flower.ai/federations <https://flower.ai/federations>`_ dashboard, where registered
SuperNodes should be listed with a status of **"Registered."**

**************************************
 SuperNode Initialization with Docker
**************************************

Utilize the official Flower Docker images to launch each SuperNode.

Start SuperNode 1
=================

.. code-block:: bash

    docker run --rm -d \
      --name supernode_1 \
      --volume ./keys/:/app/keys/:ro \
      -it flwr/supernode:1.24.0-py3.13-ubuntu24.04 \
      --superlink fleet-supergrid.flower.ai:443 \
      --auth-supernode-private-key keys/client_credentials_1

Initiate Additional SuperNodes
==============================

.. code-block:: bash

    # SuperNode 2
    docker run --rm -d \
      --name supernode_2 \
      --volume ./keys/:/app/keys/:ro \
      -it flwr/supernode:1.24.0-py3.13-ubuntu24.04 \
      --superlink fleet-supergrid.flower.ai:443 \
      --auth-supernode-private-key keys/client_credentials_2

    # SuperNode 3
    docker run --rm -d \
      --name supernode_3 \
      --volume ./keys/:/app/keys/:ro \
      -it flwr/supernode:1.24.0-py3.13-ubuntu24.04 \
      --superlink fleet-supergrid.flower.ai:443 \
      --auth-supernode-private-key keys/client_credentials_3

Command Argument Summary
========================

- ``--rm -d``: Runs the container in detached mode and automatically removes it upon
  exit.
- ``--name supernode_X``: Assigns a user-friendly name for container identification.
- ``--volume ./keys/:/app/keys/:ro``: Mounts the local ``./keys/`` directory into the
  container as read-only.
- ``flwr/supernode:1.24.0-py3.13-ubuntu24.04``: Specifies the official Flower Docker
  image.
- ``--superlink fleet-supergrid.flower.ai:443``: Defines the SuperGrid endpoint for
  connection.
- ``--auth-supernode-private-key keys/client_credentials_X``: Provides the private key
  necessary for SuperNode authentication.

****************************
 Online Status Verification
****************************

Consult the `flower.ai/federations <https://flower.ai/federations>`_ dashboard. Your
SuperNodes should display an **"Online"** status, indicating they are prepared to
execute federated learning tasks.

**************************
 Federated Task Execution
**************************

Create New Project (If Needed)
==============================

If an existing project is unavailable, initialize a new one:

.. code-block:: bash

    # Create new Flower example project
    flwr new

    # Follow the interactive setup prompts:
    # - Select option 6 (Numpy example)
    # - Input the desired project name and other metadata

Login in New Directory (If Applicable)
======================================

If a new project directory was created, re-authentication may be necessary:

.. code-block:: bash

    flwr login . supergrid

Federated Workflow Execution
============================

Run your defined federated task on the SuperGrid:

.. code-block:: bash

    flwr run . supergrid --stream

Workflow Execution Description
==============================

Your application is deployed to all currently connected SuperNodes. The federated
workflow commences execution across all online nodes. Results are collected, aggregated,
and displayed. The ``--stream`` flag enables the display of real-time execution logs.

*****************
 Troubleshooting
*****************

Common Issue: "Read-only file system" Error
===========================================

If the error message ``OSError: [Errno 30] Read-only file system: 'final_model.npz'`` is
encountered, it is because the ServerApp is attempting to persist model weights to a
file system location lacking write permissions.

**Corrective Action:** Comment out or remove the code responsible for weight-saving
within ``server_app.py``:

.. code-block:: python

    # Comment out these lines in server_app.py
    # np.savez("final_model.npz", **arrays)

After implementing this change, re-execute the workflow:

.. code-block:: bash

    flwr run . supergrid --stream

Other Frequently Encountered Issues
===================================

**"Not logged in"**
    Re-run the login command: ``flwr login . supergrid``.

**"Public key already registered"**
    The key has been previously used. Verify your documented Node IDs. If necessary,
    generate and utilize a distinct key file.

**SuperNode Fails to Display Online Status**
    - Inspect Docker logs for the specific container (``docker logs supernode_1``).
    - Confirm the Node ID has been registered by the Flower Labs team.
    - Ensure the private and public key pair utilized is correct and matching.
    - Verify network connectivity to the endpoint ``fleet-supergrid.flower.ai:443``.

**"Federation not found"**
    - Confirm the federation name in ``pyproject.toml`` is correct.
    - Verify current login status: ``flwr federation show . supergrid``.

*******************
 Summary Checklist
*******************

The full workflow is summarized below:

1. Establish an account at `flower.ai <https://flower.ai>`_.
2. Access the `flower.ai/federations <https://flower.ai/federations>`_ dashboard.
3. Install the Flower CLI and Docker.
4. Modify ``certificate.conf`` to include ``DNS.2 = fleet-supergrid.flower.ai``.
5. Configure ``generate_auth_keys.sh`` for the intended number of SuperNodes.
6. Execute ``generate_auth_keys.sh`` to create key pairs.
7. Securely back up all generated keys.
8. Add the SuperGrid federation configuration to ``pyproject.toml``.
9. Run the login command: ``flwr login . supergrid``.
10. Register each SuperNode: ``flwr supernode register keys/client_credentials_X.pub .
    supergrid``.
11. Record and transmit Node IDs to the Flower Labs team.
12. Await official confirmation from the team.
13. Start SuperNode Docker containers, providing the private keys.
14. Verify SuperNodes appear as "Online" on the dashboard.
15. Initiate the federated task: ``flwr run . supergrid --stream``.
16. Monitor execution results and logs.

**************************
 Quick Reference Commands
**************************

.. code-block:: bash

    # Log in to SuperGrid
    flwr login . supergrid

    # Register a SuperNode
    flwr supernode register <PATH>/keys/client_credentials_X.pub . supergrid

    # Start a SuperNode container
    docker run --rm -d \
      --name supernode_X \
      --volume ./keys/:/app/keys/:ro \
      -it flwr/supernode:1.24.0-py3.13-ubuntu24.04 \
      --superlink fleet-supergrid.flower.ai:443 \
      --auth-supernode-private-key keys/client_credentials_X

    # Execute the federated workflow
    flwr run . supergrid --stream

    # Check SuperNode logs (real-time)
    docker logs -f supernode_X

    # Stop a SuperNode container
    docker stop supernode_X

    # List all runs in the federation
    flwr list . supergrid

    # Show detailed federation information
    flwr federation show . supergrid

**********************
 Additional Resources
**********************

- `Flower Documentation <https://flower.ai/docs/>`_
- `Authentication Example
  <https://github.com/adap/flower/tree/main/examples/flower-authentication>`_
- `SuperGrid Information <https://flower.ai/federations>`_
- `Flower Community <https://flower.ai/join-slack>`_

For assistance, please contact the Flower Labs team or visit the community Slack
channel.
