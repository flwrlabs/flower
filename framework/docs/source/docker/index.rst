:og:description: Learn how to run Flower for federated learning using Docker with pre-made images, including advanced options and Docker Compose for seamless deployment.
.. meta::
    :description: Learn how to run Flower for federated learning using Docker with pre-made images, including advanced options and Docker Compose for seamless deployment.

#########################
 Run Flower using Docker
#########################

Start your Flower journey with our pre-made Docker images on Docker Hub, supporting
``amd64`` and ``arm64v8`` architectures.

Starting with Flower 1.40.0, the Alpine variants of ``flwr/base``, ``flwr/superlink``,
and ``flwr/supernode`` use Alpine 3.23 and Python 3.13. Alpine 3.22 images are no longer
built for new releases. When upgrading to Flower 1.40.0 or later, update the Flower
version in an explicitly pinned tag and use the ``-py3.13-alpine3.23`` suffix.

Our Quickstart guide walks you through containerizing a Flower project and running it
end to end using Docker.

*****************
 Getting Started
*****************

.. toctree::
    :maxdepth: 1

    tutorial-quickstart-docker

***********************
 Running in Production
***********************

.. toctree::
    :maxdepth: 1

    enable-tls
    persist-superlink-state

******************
 Advanced Options
******************

.. toctree::
    :maxdepth: 1

    set-environment-variables
    run-as-root-user
    run-as-subprocess
    pin-version
    use-a-different-version

*********************************
 Run Flower using Docker Compose
*********************************

.. toctree::
    :maxdepth: 1

    tutorial-quickstart-docker-compose
    run-quickstart-examples-docker-compose
    tutorial-deploy-on-multiple-machines
