:og:description: Enable TLS in Flower for secure communication with encrypted data transmission using PEM-encoded certificates and private keys.
.. meta::
    :description: Enable TLS in Flower for secure communication with encrypted data transmission using PEM-encoded certificates and private keys.

###################################
 Enable TLS for Secure Connections
###################################

When operating in a production environment, it is strongly recommended to enable
Transport Layer Security (TLS) for each Flower component to ensure secure communication.

.. note::

    For testing purposes, you can generate your own self-signed certificates. The
    `Enable SSL connections
    <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`__
    page contains a section that will guide you through the process.

.. note::

    When working with Docker on Linux, you may need to change the ownership of the
    directory containing the certificates to ensure proper access and permissions.

    By default, Flower containers run with a non-root user ``app``. The mounted files
    and directories must have the proper permissions for the user ID ``49999``.

    For example, to change the user ID of all files in the ``certificates/`` directory,
    you can run ``sudo chown -R 49999:49999 certificates/*``.

    If you later want to delete the directory, you can change the user ID back to the
    current user ID by running ``sudo chown -R $USER:$(id -gn) certificates``.

.. tab-set::

    .. tab-item:: Isolation Mode ``subprocess``

        By default, the ServerApp is executed as a subprocess within the SuperLink Docker
        container, and the ClientApp is run as a subprocess within the SuperNode Docker
        container. You can learn more about the different process modes here:
        :doc:`run-as-subprocess`.

        To enable TLS between the SuperLink and SuperNode, as well as between the SuperLink and the ``flwr``
        CLI, you will need a PEM-encoded root certificate, private key, and certificate chain.

        **SuperLink**

        Assuming all files we need are in the local ``superlink-certificates`` directory,
        we can use the flag ``--volume`` to mount the local directories into the SuperLink container:

        .. code-block:: bash

            $ docker run --rm \
                --volume ./superlink-certificates/:/app/certificates/:ro \
                <superlink-image> \
                --ssl-ca-certfile certificates/ca.crt \
                --ssl-certfile certificates/server.pem \
                --ssl-keyfile certificates/server.key \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/:/app/certificates/:ro``: Mount the ``superlink-certificates``
              | directory in the current working directory of the host machine as a read-only volume
              | at the ``/app/certificates`` directory inside the container.
              |
              | This allows the container to access the TLS certificates that are stored in the certificates
              | directory.
            * ``<superlink-image>``: The name of your SuperLink image to be run.
            * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
              | inside the container.
              |
              | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
              | SuperLink.
            * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperLink's
              | TLS certificate file inside the container.
              |
              | The ``certificates/server.pem`` file is used to identify the SuperLink and to encrypt the
              | data that is transmitted over the network.
            * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperLink's
              | TLS private key file inside the container.
              |
              | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
              | the network.

        **SuperNode**

        .. note::

            If you're generating self-signed certificates and the ``ca.crt`` certificate doesn't
            exist on the SuperNode, you can copy it over after the generation step.

        .. code-block:: bash

            $ docker run --rm \
                --volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro \
                <supernode-image> \
                --root-certificates ca.crt \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro``: Mount the ``ca.crt``
              | file from the ``superlink-certificates`` directory of the host machine as a read-only
              | volume at the ``/app/ca.crt`` directory inside the container.
            * ``<supernode-image>``: The name of your SuperNode image to be run.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA certificate file
              | inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperLink.

    .. tab-item:: Isolation Mode ``process``

        In isolation mode ``process``, the ServerApp and ClientApp run in their own processes.
        Unlike in isolation mode ``subprocess``, the SuperLink or SuperNode does not attempt to
        create the respective processes; instead, they must be created externally.

        It is possible to run only the SuperLink in isolation mode ``subprocess`` and the
        SuperNode in isolation mode ``process``, or vice versa, or even both with isolation mode
        ``process``.

        **SuperLink and ServerApp**

        To enable TLS between the SuperLink and SuperNode, as well as between the SuperLink and the ``flwr``
        CLI, you will need a PEM-encoded root certificate, private key, and certificate chain.

        Assuming all files we need are in the local ``superlink-certificates`` directory, we can
        use the flag ``--volume`` to mount the local directory into the SuperLink container:


        .. code-block:: bash
            :substitutions:

            $ docker run --rm \
                --volume ./superlink-certificates/:/app/certificates/:ro \
                flwr/superlink:|stable_flwr_version| \
                --ssl-ca-certfile certificates/ca.crt \
                --ssl-certfile certificates/server.pem \
                --ssl-keyfile certificates/server.key \
                --isolation process \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/:/app/certificates/:ro``: Mount the
              | ``superlink-certificates`` directory in the current working directory of the host
              | machine as a read-only volume at the ``/app/certificates`` directory inside the container.
              |
              | This allows the container to access the TLS certificates that are stored in the certificates
              | directory.
            * | :substitution-code:`flwr/superlink:|stable_flwr_version|`: The name of the image to be run and the specific
              | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
            * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
              | inside the container.
              |
              | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
              | SuperLink.
            * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperLink's
              | TLS certificate file inside the container.
              |
              | The ``certificates/server.pem`` file is used to identify the SuperLink and to encrypt the
              | data that is transmitted over the network.
            * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperLink's
              | TLS private key file inside the container.
              |
              | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
              | the network.
            * | ``--isolation process``: Tells the SuperLink that the ServerApp is created by separate
              | independent process. The SuperLink does not attempt to create it.

        Start the ServerApp container:

        .. code-block:: bash

            $ docker run --rm \
                <serverapp-image> \
                --insecure \
                <additional-args>

        .. dropdown:: Understand the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * ``<serverapp-image>``: The name of your ServerApp image to be run.
            * | ``--insecure``:  This flag tells the container to operate in an insecure mode, allowing
              | unencrypted communication. Secure connections will be added in future releases.

        **SuperNode and ClientApp**

        .. note::

            If you're generating self-signed certificates and the ``ca.crt`` certificate doesn't
            exist on the SuperNode, you can copy it over after the generation step.

        Start the SuperNode container:

        .. code-block:: bash
            :substitutions:

            $ docker run --rm \
                --volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro \
                flwr/supernode:|stable_flwr_version| \
                --root-certificates ca.crt \
                --isolation process \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro``: Mount the ``ca.crt`` file from the
              | ``superlink-certificates`` directory of the host machine as a read-only volume at the ``/app/ca.crt``
              | directory inside the container.
            * | :substitution-code:`flwr/supernode:|stable_flwr_version|`: The name of the image to be run and the specific
              | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA certificate file
              | inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperLink.
            * | ``--isolation process``: Tells the SuperNode that the ClientApp is created by separate
              | independent process. The SuperNode does not attempt to create it.

        Start the ClientApp container:

        .. code-block:: bash

            $ docker run --rm \
                <clientapp-image> \
                --insecure \
                <additional-args>

        .. dropdown:: Understand the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * ``<clientapp-image>``: The name of your ClientApp image to be run.
            * | ``--insecure``:  This flag tells the container to operate in an insecure mode, allowing
              | unencrypted communication. Secure connections will be added in future releases.

Locate the Flower Configuration TOML file in your machine.

.. code-block:: console
    :emphasize-lines: 3

    $ flwr config list

    Flower Config file: /path/to/.flwr/config.toml
    SuperLink connections:
      supergrid
      local (default)

Append the following lines to the end of the ``config.toml`` file to add a new SuperLink
connection and save it:

.. code-block:: toml
    :caption: config.toml

    [superlink.local-deployment-tls]
    address = "127.0.0.1:9093"
    root-certificates = "/absolute/path/to/superlink-certificates/ca.crt"

.. note::

    You can customize the string that follows ``superlink.`` to fit your needs. However,
    please note that the string cannot contain a dot (``.``).

    In this example, ``local-deployment`` has been used. Just remember to replace
    ``local-deployment`` with your chosen name in both the ``superlink.`` string and the
    corresponding ``flwr run .`` command. Refer to the `Flower Configuration
    <ref-flower-configuration.html>`_ for more information.
