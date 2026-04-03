:og:description:  Guide to starting a TLS-secured Flower server (“SuperLink”) and connecting a Flower client (“SuperNode”) with secure connections.
.. meta::
    :description: Guide to starting a TLS-secured Flower server (“SuperLink”) and connecting a Flower client (“SuperNode”) with secure connections.

########################
 Enable TLS connections
########################

Transport Layer Security (TLS) ensures the communication between endpoints is encrypted.
This guide describes how to establish secure TLS Superlink ↔ SuperNodes as well as User
↔ SuperLink connections.

.. note::

    This guide builds on the Flower App setup presented in
    :doc:`how-to-run-flower-with-deployment-engine` guide and extends it to replace the
    use of ``--insecure`` in favour of TLS.

.. tip::

    Checkout the `Flower Authentication
    <https://github.com/flwrlabs/flower/tree/main/examples/supernode-authentication>`_
    example for a complete self-contained example on how to setup TLS and (optionally)
    node authentication. Check out the :doc:`how-to-authenticate-supernodes` guide to
    learn more about adding an authentication layer to SuperLink ↔ SuperNode
    connections.

**************
 Certificates
**************

Using TLS-enabled connections expects some certificates generated and passed when
launching the SuperLink, the SuperNodes and when a user (e.g. a data scientist that
wants to submit a ``Run``) interacts with the federation via the `flwr CLI
<ref-api-cli.html>`_.

We have prepared a script that can be used to generate such set of certificates. While
using these are fine for prototyping, we advice you to follow the standards set in your
team/organization and generated the certificates and share them with the corresponding
parties. Refer to the **Generate TLS certificates** section in the example linked at the
top of this guide.

.. code-block:: bash

    # In the example directory, generate the certificates
    $ python generate_creds.py

This will generate the TLS certificates in a new ``certificates/`` directory. Copy this
directory into the directory of your app (e.g. a directory generated earlier via ``flwr
new``).

The approach for generating TLS certificates in the context of this example can serve as
an inspiration and starting point, but it should not be used as a reference for
production environments. Please refer to other sources regarding the issue of correctly
generating certificates for production environments. For non-critical prototyping or
research projects, it might be sufficient to use the self-signed certificates generated
using the scripts mentioned in this guide.

**********************************
 Launching the SuperLink with TLS
**********************************

This section describes how to launch a SuperLink that works on TLS-enabled connections.
The code snippet below assumes the `certificates/` directory is in the same directory
where you execute the command from. Edit the paths accordingtly if that is not the case.
When providing certificates, the SuperLink expects a tuple of three certificates paths:
CA certificate, server certificate and server private key.

.. code-block:: bash
    :emphasize-lines: 2,3,4

    $ flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key

.. dropdown:: Understand the command

    * ``--ssl-ca-certfile``: Specify the location of the CA certificate file in your file. This file is a certificate that is used to verify the identity of the SuperLink.
    * | ``--ssl-certfile``: Specify the location of the SuperLink's TLS certificate file. This file is used to identify the SuperLink and to encrypt the packages that are transmitted over the network.
    * | ``--ssl-keyfile``: Specify the location of the SuperLink's TLS private key file. This file is used to decrypt the packages that are transmitted over the network.

************************************
 Connecting the SuperNodes with TLS
************************************

This section describes how to launch a SuperNode that works on TLS-enabled connections.
The code snippet below assumes the `certificates/` directory is in the same directory
where you execute the command from. To enable TLS, the only change required when
launching the SuperNode is replacing ``--insecure`` with ``--root-certificates``

.. code-block:: bash
    :emphasize-lines: 2,2

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 127.0.0.1:9094 \
        --node-config="partition-id=0 num-partitions=2"

.. dropdown:: Understand the command

    * ``--root-certificates``:This specifies the location of the CA certificate file. The ``ca.crt`` file is used to verify the identity of the SuperLink.

Follow the same procedure, i.e. replacing ``--insecure`` with ``--root-certificates``,
to launch the second SuperNode.

.. code-block:: bash
    :emphasize-lines: 2,2

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 127.0.0.1:9095 \
        --node-config="partition-id=1 num-partitions=2"

************************
 TLS-enabled Flower CLI
************************

The `Flower CLI <ref-api-cli.html>`_ (e.g. ``flwr run`` command) is the way a user (e.g.
a data scientist) can interact with a deployed federation. The Flower CLI commands are
processed by the SuperLink and therefore, if it has been configured to only operate on
TLS conenction, the requests sent by the Flower CLI need to make use of a TLS
certificate. To do so, replace the ``insecure = true`` field in your Flower
Configuration TOML file with a new field that reads the certificate:

.. code-block:: toml
    :caption: config.toml
    :emphasize-lines: 3,3

    [superlink.local-deployment]
    address = "127.0.0.1:9093"
    root-certificates = "/absolute/path/to/certificates/ca.crt"

Note that the path to the ``root-certificates`` is relative to the root of the project.
Now, you can run the example by executing ``flwr run``:

.. code-block:: bash

    $ flwr run . local-deployment --stream

.. tip::

    You can setup your ``local-deployment`` profile as the default so you don't have to
    specify it in every Flower CLI command that needs to connect to the SuperLink. For
    that and more details about the Flower configuration, refer to the :doc:`the Flower
    Configuration <ref-flower-configuration>` guide.

************
 Conclusion
************

You should now have learned how to generate self-signed certificates using the given
script, start an TLS-enabled server and have two clients establish secure connections to
it. You should also have learned how to run your Flower project using ``flwr run`` with
TLS enabled. All other commands in the `Flower CLI <ref-api-cli.html>`_ will also be
TLS-enabled.

.. note::

    Refer to the :doc:`docker/index` documentation to learn how to setup a federation
    where each component runs in its own Docker container. You can make use of TLS and
    other security features in Flower such as implement a SuperNode authentication
    mechanism.

**********************
 Additional resources
**********************

These additional sources might be relevant if you would like to dive deeper into the
topic of certificates:

- `Let's Encrypt <https://letsencrypt.org/docs/>`_
- `certbot <https://certbot.eff.org/>`_
