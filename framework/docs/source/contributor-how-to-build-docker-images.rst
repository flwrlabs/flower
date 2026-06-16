###########################################
 How to Build Docker Flower Images Locally
###########################################

Flower provides pre-made docker images on `Docker Hub <https://hub.docker.com/u/flwr>`_
that include all necessary dependencies for running the SuperLink, SuperNode or
ServerApp. You can also build your own custom docker images from scratch with a
different version of Python or Linux distribution (Ubuntu/Alpine) if that is what you
need. In this guide, we will explain what images exist and how to build them locally.

Before we can start, we need to meet a few prerequisites in our local development
environment.

1. Clone the ``flower`` repository.

       .. code-block:: bash

           $ git clone --depth=1 https://github.com/flwrlabs/flower.git && cd flower

2. Verify the Docker daemon is running.

   The build instructions that assemble the images are located in the respective
   Dockerfiles. You can find them in the subdirectories of ``framework/docker``.

   Flower Docker images are configured via build arguments. Through build arguments, we
   can make the creation of images more flexible. For example, in the base image, we can
   specify the version of Python to install using the ``PYTHON_VERSION`` build argument.
   Some of the build arguments have default values, others must be specified when
   building the image. All available build arguments for each image are listed in one of
   the tables below.

*************************
 Building the Base Image
*************************

.. list-table::
    :widths: 25 45 15 15
    :header-rows: 1

    - - Build argument
      - Description
      - Required
      - Example
    - - ``DISTRO``
      - The Linux distribution to use as the base image.
      - No
      - ``ubuntu``
    - - ``DISTRO_VERSION``
      - Version of the Linux distribution.
      - No
      - :substitution-code:`|ubuntu_version|`
    - - ``PYTHON_VERSION``
      - Version of ``python`` to be installed.
      - No
      - ``3.11`` or ``3.11.1``
    - - ``PIP_VERSION``
      - Version of ``pip`` to be installed.
      - Yes
      - :substitution-code:`|pip_version|`
    - - ``SETUPTOOLS_VERSION``
      - Version of ``setuptools`` to be installed.
      - Yes
      - :substitution-code:`|setuptools_version|`
    - - ``FLWR_VERSION``
      - Version of Flower to be installed.
      - Yes
      - :substitution-code:`|stable_flwr_version|`
    - - ``FLWR_PACKAGE``
      - The Flower package to be installed.
      - No
      - ``flwr`` or ``flwr-nightly``
    - - ``FLWR_VERSION_REF``
      - A `direct reference
        <https://packaging.python.org/en/latest/specifications/version-specifiers/#direct-references>`_
        without the ``@`` specifier. If both ``FLWR_VERSION`` and ``FLWR_VERSION_REF``
        are specified, the ``FLWR_VERSION_REF`` has precedence.
      - No
      - `Direct Reference Examples`_

The following example creates a base Ubuntu/Alpine image with Python ``3.11.0``, pip
:substitution-code:`|pip_version|`, setuptools :substitution-code:`|setuptools_version|`
and Flower :substitution-code:`|stable_flwr_version|`:

.. code-block:: bash
    :substitutions:

    $ cd framework/docker/base/<ubuntu|alpine>
    $ docker build \
      --build-arg PYTHON_VERSION=3.11.0 \
      --build-arg FLWR_VERSION=|stable_flwr_version| \
      --build-arg PIP_VERSION=|pip_version| \
      --build-arg SETUPTOOLS_VERSION=|setuptools_version| \
      -t flwr_base:0.1.0 .

In this example, we specify our image name as ``flwr_base`` and the tag as ``0.1.0``.
Remember that the build arguments as well as the name and tag can be adapted to your
needs. These values serve as examples only.

********************************
 Building a Flower Binary Image
********************************

.. list-table::
    :widths: 25 45 15 15
    :header-rows: 1

    - - Build argument
      - Description
      - Required
      - Example
    - - ``BASE_REPOSITORY``
      - The repository name of the base image.
      - No
      - ``flwr/base``
    - - ``BASE_IMAGE``
      - The Tag of the Flower base image.
      - Yes
      - :substitution-code:`|stable_flwr_version|-py3.11-ubuntu|ubuntu_version|`

For example, to build a SuperLink image with the latest Flower version, Python 3.11 and
Ubuntu 22.04, run the following:

.. code-block:: bash
    :substitutions:

    $ cd framework/docker/superlink
    $ docker build \
      --build-arg BASE_IMAGE=|stable_flwr_version|-py3.11-ubuntu22.04 \
      -t flwr_superlink:0.1.0 .

If you want to use your own base image instead of the official Flower base image, all
you need to do is set the ``BASE_REPOSITORY`` build argument to ``flwr_base`` (as we've
specified above).

.. code-block:: bash

    $ cd framework/docker/superlink/
    $ docker build \
      --build-arg BASE_REPOSITORY=flwr_base \
      --build-arg BASE_IMAGE=0.1.0
      -t flwr_superlink:0.1.0 .

After creating the image, we can test whether the image is working:

.. code-block:: bash

    $ docker run --rm flwr_superlink:0.1.0 --help

***************************
 Direct Reference Examples
***************************

.. code-block:: bash
    :substitutions:

    # main branch
    git+https://github.com/flwrlabs/flower.git@main#subdirectory=framework

    # commit hash
    git+https://github.com/flwrlabs/flower.git@4bc1bca3d0576dd2233972d9d91c2c7e8eb03edd#subdirectory=framework

    # tag
    git+https://github.com/flwrlabs/flower.git@|stable_flwr_version|#subdirectory=framework

    # artifact store
    https://artifact.flower.ai/py/main/latest/flwr-|stable_flwr_version|-py3-none-any.whl
