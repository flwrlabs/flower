.. |authenticate_supernodes| replace:: Authenticate Supernodes

.. _authenticate_supernodes: how-to-authenticate-supernodes.html

.. |enable_tls_connections| replace:: Enable TLS Connections

.. _enable_tls_connections: how-to-enable-tls-connections.html

.. |flower_architecture_link| replace:: Flower Architecture

.. _flower_architecture_link: explanation-flower-architecture.html

.. |flower_docker_index| replace:: Run Flower using Docker

.. _flower_docker_index: docker/index.html

#####################
 Run Flower on Azure
#####################

.. note::

    There are many ways to deploy Flower on Microsoft Azure. The instructions provided
    in this guide is just a basic walkthrough, step-by-step guide on how to quickly
    setup and run a Flower application on a Federated Learning environment on Microsoft
    Azure.

In this how-to guide, we want to create a Federated Learning environment on Microsoft
Azure using three Virtual Machines (VMs). From the three machines, one machine will be
used as the Federation server and two as the Federation clients. Our goal is to create a
Flower federation on Microsoft Azure where we can run Flower apps from our local
machine, e.g., laptop.

On the Federation server VM we will deploy the long-running Flower server
(``SuperLink``) and on the two Federation client VMs we will deploy the long-running
Flower client (``SuperNode``). For more details For more details regarding the
``SuperLink`` and ``SuperNode`` concepts, please see the |flower_architecture_link|_ .

***********
 Azure VMs
***********

First we need to create the three VMs configure their Python environments, and inbound
networking rules to allow cross-VM communication.

VM Create
=========

Assuming we are already inside the Microsoft Azure portal, we navigate to the ``Create``
page and we select ``Azure virtual machine``. In the new page, for each VM we edit the
properties as follows:

.. list-table::
    :align: left
    :widths: 13 30
    :header-rows: 0

    - - **Virtual machine name**
      - for server machine we can use ``flower-server`` and for clients,
        ``flower-client-1`` and ``flower-client-2``
    - - **Image**
      - in this guide, we use ``Ubuntu Server 24.04 - x64 Gen2 LTS``
    - - **Size**
      - in this guide, we use ``Standard_D2s_v3 - 2 vcpus, 8GiB memory``

.. tip::

    For resource group, we can create a new group and assign it to all VMs.

When each VM instance has been created the portal will allow you to download the public
key (.pem) of each instance. Make sure you save this key in safe place and change its
permissions to user read only, i.e., run the ``chmod 400 <PATH_TO_PEM_FILE>`` command
for every .pem file.

Once all three VMs are created then navigate to the overview page where all three VMs
are listed and open every other VM, and copy its Public IP address. Using the Public IP
address and the public key (after changing the permissions), login to the instances from
our local machine by running the following command (by default Azure creates the
``azureuser``):

.. code-block:: bash

    ssh -i <PATH_TO_PEM_FILE> azureuser@<PUBLIC_IP>

VM Networking
=============

During the execution of the Flower application, the server VM (``SuperLink``) will be
responsible to orchestrate the execution of the application across the client VMs
(``SuperNode``). When the SuperLink server starts, by default, it listens to the
following ports: ``{9092, 9093}``. Port `9092` is used to communicate with the
Federation clients (``SuperNode``) and port ``9093`` to receive and execute Flower
applications.

Therefore, to enable this communication we need to allow inbound traffic to the server
VM instance. To achieve this, we need to navigate to the Networking page of the server
VM in the Microsoft Azure portal. There, we will click the ``Add inbound port rule``. In
the new window that appears, we edit the rule properties as follows:

The rest of the fields can be left at their default values.

.. list-table::
    :align: left
    :widths: 25 30
    :header-rows: 0

    - - **Source**
      - ``IP Addresses``
    - - **Source IP addresses/CIDR ranges**
      - add client VMs' Public IP (separated by comma)
    - - **Destination**
      - ``Any``
    - - **Service**
      - ``custom``
    - - **Destination port ranges**
      - ``9092``
    - - **Protocol**
      - ``TCP``

Finally, we need to also open port 9093 to allow receiving and executing incoming
application requests. To enable this we just need to repeat the steps above, i.e.,
create a new inbound rule, where for port range we assign port 9093. If we already know
the Public IP from which our local machine (e.g., laptop) will be submitting
applications to the Azure cluster, then we just need to specify the Source IP
address/CIDR range. However, if we want to keep the port widely open we simply need to
change source to ``Any``.

To be more precise, if we know the Public IP of our machine, then we make the following
changes:

.. list-table::
    :align: left
    :widths: 25 25
    :header-rows: 0

    - - **Source IP addresses/CIDR ranges**
      - add machine's Public IP
    - - **Destination port ranges**
      - ``9093``

Otherwise, we change the properties as follows:

.. list-table::
    :align: left
    :widths: 25 25
    :header-rows: 0

    - - **Source**
      - ``Any``
    - - **Destination port ranges**
      - ``9093``

********************
 Flower Environment
********************

Assuming we have been able to login to each VM, and create a Python environment with
Flower and all its dependencies installed (``pip install flwr``), we can create a Flower
application by running the ``flwr new`` command. The console will then prompt us to give
a name to the project, e.g., ``flwr_azure_test``, the name of the author and select the
type of the Flower Framework we want to run, e.g., ``numpy``.

.. note::

    An alternative approach would be to use Docker in each VM, with each image
    containing the necessary environment and dependencies. For more details please refer
    to the |flower_docker_index|_ guide.

Server Initialization
=====================

After configuring the Flower application environment, we proceed by starting the Flower
long-running processes (i.e., ``SuperLink`` and ``SuperNode``) at each VM instance. In
particular, we need to run the following commands, first in the server (``SuperLink``)
and then at each client (``SuperNode``).

.. note::

    To enable authentication and encrypted communication during the execution lifecycle
    of the Flower application, please have a look at the following resources:
    |authenticate_supernodes|_, |enable_tls_connections|_

.. code-block:: bash

    # Server VM (SuperLink)
    flower-superlink --insecure

    # Client-1 VM (SuperNode-1)
    flower-supernode \
      --insecure \
      --superlink="SUPERLINK_PUBLIC_IP:9092"  # SuperLink public ip and port

    # Client-2 VM (SuperNode-2)
    flower-supernode \
      --insecure \
      --superlink="SUPERLINK_PUBLIC_IP:9092"  # SuperLink public ip and port

Run Flower App
==============

Finally, after all running Flower processes have been initialized on the Microsoft Azure
cluster, in our local machine, we first need to install Flower and create a Flower App.

.. code-block:: bash

    # Install flower
    pip install -U flwr

    # This creates a basic Flower App using the numpy framework
    flwr new @flwrlabs/quickstart-numpy

Next, we need to create a new SuperLink connection in the Flower Configuration file:

1. Run ``flwr config list`` to locate the Flower configuration file on your machine and
   view available SuperLink connections.

   .. code-block:: console
       :emphasize-lines: 3

         $ flwr config list

         Flower Config file: /path/to/.flwr/config.toml
         SuperLink connections:
           supergrid
           local (default)

2. Open the Flower Configuration file (``config.toml``) and add a new SuperLink
   connection at the end:

   .. code-block:: toml
       :caption: config.toml

       [superlink.my-federation]
       address = "SUPERLINK_PUBLIC_IP:9093"  # Address of the SuperLink Control API
       insecure = true

Then from our local machine we need to run ``flwr run . my-federation``.

************
 Next Steps
************

.. warning::

    This guide is not suitable for production environments due to missing authentication
    and TLS security.

To enable authentication and establish secure connections, please refer to the following
resources: |authenticate_supernodes|_, |enable_tls_connections|_
