:og:description: Configure the logging level for your Flower processes.
.. meta::
    :description: Configure the logging level for your Flower processes.

###################
 Configure logging
###################

By default, the Flower logger uses logging level ``INFO``. This can be changed via the
``FLWR_LOG_LEVEL`` environment variable to any other levels that Python's `logging
module <https://docs.python.org/3/library/logging.html#logging-levels>`_ supports. For
example, to launch your ``SuperLink`` with ``DEBUG`` logs, use:

.. code-block:: shell
    :emphasize-lines: 2,11

    # Launch the SuperLink with TLS (or use --insecure)
    FLWR_LOG_LEVEL=DEBUG flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key

    WARNING 2025-08-20 17:13:30,391:   DEBUG logs enabled. Do not use this in production, as it may expose sensitive details.
    INFO 2025-08-20 17:13:31,360:      Starting Flower SuperLink
    INFO 2025-08-20 17:13:31,378:      Flower Deployment Runtime: Starting Control API on 0.0.0.0:9093
    INFO 2025-08-20 17:13:31,381:      Flower Deployment Runtime: Starting Runtime API on 0.0.0.0:9091
    DEBUG 2025-08-20 17:13:31,382:     Automatic node authentication enabled
    INFO 2025-08-20 17:13:31,382:      Flower Deployment Runtime: Starting Fleet API (gRPC-rere) on 0.0.0.0:9092
    WARNING 2025-08-20 17:13:31,515:   DEBUG logs enabled. Do not use this in production, as it may expose sensitive details.
    INFO 2025-08-20 17:13:32,324:      Starting Flower SuperExec

.. note::

    You can make use of the ``FLWR_LOG_LEVEL`` environment variable when executing other
    Flower commands to provision the different components in a Flower Federation (see
    :doc:`how-to-run-flower-with-deployment-engine`) or using the `flwr CLI
    <ref-api-cli.html>`_.

************************
 Configure gRPC logging
************************

Flower uses `gRPC <https://grpc.io/>`_ to communicate between each component (see
:doc:`ref-flower-network-communication`). You can set the verbosity level of ``gRPC``
logs using `gRPC environment variables
<https://github.com/grpc/grpc/blob/master/doc/environment_variables.md>`_.
