:og:description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.
.. meta::
    :description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.

.. |flower_cli_supernode_link| replace:: ``Flower CLI``

.. _flower_cli_supernode_link: ref-api-cli.html#flwr-supernode

#########################
 Authenticate SuperNodes
#########################

When running a Flower Federation (see :doc:`ref-flower-network-communication`) it is
fundamental that an authentication mechanism is available between the SuperLink and the
SuperNodes that connect to it. Flower comes with two different mechanisms to
authenticate SuperNodes that connect to a running SuperLink:

- **Automatic authentication**: In this mode, the SuperLink checks the timestamp-based
  signature in each request from SuperNodes to prevent impersonation and replay attacks.
  The goal of this mode is to confirm the identity of connected SuperNodes; however, it
  does **not** restrict which SuperNodes can connect to the SuperLink. Consequently, any
  SuperNode is allowed to connect to the SuperLink.
- **CLI-managed authentication**: This mode operates similarly to automatic
  authentication but requires starting the SuperLink with the
  ``--enable-supernode-auth`` flag. To connect a SuperNode to the SuperLink, its public
  key must first be registered using the |flower_cli_supernode_link|_. Only registered
  SuperNodes are permitted to connect to the SuperLink, making this mode more secure by
  restricting connections to authorized SuperNodes only.

The automatic authentication mode works out of the box and therefore requires no
configuration. On the other hand, CLI-managed authentication mode is more sophisticated
and how it works and how it can be used is presented reminder of this guide. Flower's
CLI-managed SuperNode authentication leverages a signature-based mechanism to verify
each SuperNode's identity and is only available when encrypted connections (SSL/TLS) are
enabled:

- Each SuperNode must already possess a unique Elliptic Curve (EC) public/private key
  pair.
- The SuperLink (server) maintains a whitelist of EC public keys for all trusted
  SuperNodes (clients), managed through the Flower CLI.
- A SuperNode signs a timestamp with its private key and sends the signed timestamp to
  the SuperLink.
- The SuperLink verifies the signature and timestamp using the SuperNode's public key.

.. note::

    This guide builds on the Flower App setup presented in the
    :doc:`how-to-enable-tls-connections` guide and extends it to introduce node
    authentication to the SuperLink ↔ SuperNode connection.

.. tip::

    Checkout the `Flower Authentication
    <https://github.com/flwrlabs/flower/tree/main/examples/supernode-authentication>`_
    example for a complete self-contained example on how to setup TLS and node
    authentication.

******************************
 Generate authentication keys
******************************

To establish an authentication mechanism by which only authorized SuperNodes can connect
to a running SuperLink, a set of key pairs for both SuperLink and SuperNodes need to be
created.

We have prepared a script that can be used to generate such set of keys. While using
these are fine for prototyping, we advice you to follow the standards set in your
team/organization and generated the keys and share them with the corresponding parties.
Refer to the **Generate public and private keys for SuperNode authentication** section
in the example linked at the top of this guide.

.. code-block:: bash

    # In the example directory, generate the public/private key pairs
    # along side the TLS certificates.
    $ python generate_creds.py

This will generate the keys in a new ``keys/`` directory. By default it creates a key
pair for the SuperLink and one for each SuperNode. Copy this directory into the
directory of your app (e.g. a directory generated earlier via ``flwr new``).

*****************************************
 Enable node authentication in SuperLink
*****************************************

To launch a SuperLink with SuperNode authentication enabled, you need to provide three
aditional files in addition to the certificates needed for the TLS connections. Recall
that the authentication feature can only be enabled in the presence of TLS.

.. code-block:: bash
    :emphasize-lines: 5

    $ flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --enable-supernode-auth

.. dropdown:: Understand the command

    * ``--enable-supernode-auth``: Enables SuperNode authentication therefore only Supernodes that are first register on the SuperLink will be able to establish a connection.

*********************
 Register SuperNodes
*********************

Once your SuperLink is running, the next step is to register the SuperNodes that will be
allowed to connect to it. This process is handled through the
|flower_cli_supernode_link|_ using the public keys previously generated for each
SuperNode you plan to connect to the SuperLink.

First, ensure your Flower configuration file has a SuperLink connection profile that
points to your running SuperLink and that includes a ``root-certificates`` field with
the path to the CA certificate used to launch the SuperLink. For example:

.. code-block:: toml
    :caption: config.toml
    :emphasize-lines: 3

    [superlink.local-deployment]
    address = "127.0.0.1:9093"
    root-certificates = "/absolute/path/to/certificates/ca.crt"

You can verify that your connection is correctly configured by running a ``flwr ls``
command:

.. code-block:: bash

    $ flwr ls local-deployment

    📄 Listing all runs...
    ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ Run ID ┃ Federation ┃ App ┃ Status ┃ Elapsed ┃ Status Changed @ ┃
    ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    └────────┴────────────┴─────┴────────┴─────────┴──────────────────┘

.. tip::

    You can setup your ``local-deployment`` profile as the default so you don't have to
    specify it in every Flower CLI command that needs to connect to the SuperLink. For
    that and more details about the Flower configuration, refer to the :doc:`the Flower
    Configuration <ref-flower-configuration>` guide.

With a correctly configured Flower CLI, let's proceed to register two SuperNodes:

.. code-block:: bash

    # flwr supernode register <supernode-pub-key> <superlink>
    $ flwr supernode register keys/supernode_credentials_1.pub local-deployment

Next, let's register the second SuperNode as well:

.. code-block:: bash

    $ flwr supernode register keys/supernode_credentials_2.pub local-deployment

You can list the registered SuperNodes using the following command:

.. code-block:: bash

    # flwr supernode list <superlink>
    $ flwr supernode list local-deployment

This will display the IDs of the SuperNodes you just registered as well as their status.
You should see a table similar to the following:

.. code-block:: bash

    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃       Node ID        ┃ Owner ┃   Status   ┃ Elapsed  ┃   Status Changed @   ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ 16019329408659850374 │ none  │ registered │          │ N/A                  │
    ├──────────────────────┼───────┼────────────┼──────────┼──────────────────────┤
    │ 8392976743692794070  │ none  │ registered │          │ N/A                  │
    └──────────────────────┴───────┴────────────┴──────────┴──────────────────────┘

The status of the SuperNodes will change after they connect to the SuperLink. Let's
proceed and laucnh the SuperNodes.

*****************************************
 Enable node authentication in SuperNode
*****************************************

Connecting a SuperNode to a SuperLink that has node authentication enabled requires
passing one additional argument (i.e. the private key of the SuperNode) in addition to
the TLS certificate.

.. code-block:: bash
    :emphasize-lines: 6

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 127.0.0.1:9094 \
        --node-config="partition-id=0 num-partitions=2" \
        --auth-supernode-private-key keys/supernode_credentials_1

.. dropdown:: Understand the command

    * ``--auth-supernode-private-key``: the private key of this SuperNode.

Follow the same procedure to launch the second SuperNode by passing its corresponding
private key:

.. code-block:: bash
    :emphasize-lines: 6

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 127.0.0.1:9095 \
        --node-config="partition-id=1 num-partitions=2" \
        --auth-supernode-private-key keys/supernode_credentials_2

After connecting both SuperNodes, you can check the status of the SuperNodes again. You
will notice their status is now ``online``:

.. code-block:: bash

    $ flwr supernode list local-deployment

    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃       Node ID        ┃ Owner ┃ Status  ┃ Elapsed ┃   Status Changed @   ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ 16019329408659850374 │ none  │ online  │ 1m 35s  │ 2025-10-13 13:40:47Z │
    ├──────────────────────┼───────┼─────────┼─────────┼──────────────────────┤
    │ 8392976743692794070  │ none  │ online  │ 79s     │ 2025-10-13 13:52:21Z │
    └──────────────────────┴───────┴─────────┴─────────┴──────────────────────┘

***********************
 Unregister SuperNodes
***********************

.. warning::

    This is a destructive operation. Unregistering a SuperNode is permanent and cannot
    be undone. If you wish to connect a SuperNode again, a new key pair is needed.

At anypoint you can unregister a SuperNode from the SuperLink (even if it has never
connected). This will prevent the SuperNode from making future request to the SuperLink.
In other words, it will no longer be authorized to pull/send, or participate in ongoing
or future runs. Unregistering a SuperNode can be done via the
|flower_cli_supernode_link|_ as follows:

.. code-block:: bash

    # flwr supernode unregister <node-id> <superlink>
    $ flwr supernode unregister 16019329408659850374 local-deployment

The above command unregisters the first SuperNode. You can verify this by listing the
SuperNodes again:

.. code-block:: bash

    $ flwr supernode list . local-deployment

    📄 Listing all nodes...
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃       Node ID        ┃ Owner ┃ Status  ┃ Elapsed ┃   Status Changed @   ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ 8392976743692794070  │ none  │ online  │ 79s     │ 2025-10-13 13:52:21Z │
    └──────────────────────┴───────┴─────────┴─────────┴──────────────────────┘

If you pass the ``--verbose`` flag to the previous command you'll see that the status of
the unregistered SuperNode has changed to ``unregistered``. By default, unregistered
SuperNodes are hidden because they can no longer reconnect to the SuperLink. That's
right, **if you wish to connect a second SuperNode a new EC key pair is needed.**

.. code-block:: bash

    $ flwr supernode list local-deployment --verbose

    📄 Listing all nodes...
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃       Node ID        ┃ Owner ┃    Status    ┃ Elapsed ┃   Status Changed @   ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ 16019329408659850374 │ none  │ online       │ 1m 35s  │ 2025-10-13 13:40:47Z │
    ├──────────────────────┼───────┼──────────────┼─────────┼──────────────────────┤
    │ 8392976743692794070  │ none  │ unregistered │ 79s     │ 2025-10-13 13:52:21Z │
    └──────────────────────┴───────┴──────────────┴─────────┴──────────────────────┘

*****************
 Security notice
*****************

The system's security relies on the credentials of the SuperLink and each SuperNode.
Therefore, it is imperative to safeguard and safely store the credentials to avoid
security risks such as Public Key Infrastructure (PKI) impersonation attacks. The node
authentication mechanism also involves human interaction, so please ensure that all of
the communication is done in a secure manner, using trusted communication methods.

************
 Conclusion
************

You should now have learned how to start a long-running Flower SuperLink and SuperNode
with node authentication enabled. You should also know the significance of the private
key and store it securely to minimize risks.

.. note::

    Refer to the :doc:`docker/index` documentation to learn how to setup a federation
    where each component runs in its own Docker container. You can make use of TLS and
    other security features in Flower such as implement a SuperNode authentication
    mechanism.
