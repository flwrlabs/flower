:og:description: Learn how Flower can automatically install Python dependencies for Flower Apps at runtime.
.. meta::
    :description: Learn how Flower can automatically install Python dependencies for Flower Apps at runtime.

############################################
 Install Flower App dependencies at runtime
############################################

Flower Apps declare their Python package dependencies in ``pyproject.toml``. Flower can
use this metadata to install app dependencies automatically before the app is executed.

Runtime dependency installation is useful when different apps need different Python
packages, or when the machine running Flower does not have every app dependency
preinstalled. For production deployments where startup time, network access, or exact
image contents matter, you can still preinstall dependencies and disable runtime
installation.

.. note::

    SuperLink enables automatic dependency installation by default; SuperNodes do not.

    To disable this behavior in SuperLink, pass
    ``--disable-runtime-dependency-installation`` or set
    ``FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`` before starting it. To enable it
    in SuperNode, pass ``--allow-runtime-dependency-installation``.

.. note::

    In SuperGrid, automatic dependency installation is enabled. This does not change the
    default for :doc:`SuperNodes connected to SuperGrid
    <how-to-connect-supernodes-to-supergrid>`; SuperNode operators decide separately
    whether to enable it.

**************************
 Declare app dependencies
**************************

Add runtime dependencies to the ``[project].dependencies`` section of your Flower App's
``pyproject.toml``:

.. code-block:: toml
    :substitutions:

    [project]
    name = "my-flower-app"
    version = "1.0.0"
    dependencies = [
        "flwr>=|stable_flwr_version|",
        "torch==2.12.0",
        "torchvision==0.27.0",
    ]

Flower reads this list when installing dependencies at runtime. The ``flwr`` dependency
is skipped during runtime installation so the running Flower process is not replaced by
the app's requirement. Other entries are installed into the runtime environment.

.. tip::

    Keep ``pyproject.toml`` up to date with all dependencies your app needs. If a
    dependency is missing from ``[project].dependencies``, automatic dependency
    installation will not install it.

**************************
 Enable it for SuperNodes
**************************

SuperNodes run ``ClientApp`` code. To let a SuperNode install the dependencies declared
by received apps, launch it with ``--allow-runtime-dependency-installation``. The
following example shows how to connect a SuperNode to a locally running SuperLink for
prototyping:

.. code-block:: shell
    :emphasize-lines: 4

    $ flower-supernode \
        --superlink 127.0.0.1:9092 \
        --insecure \
        --allow-runtime-dependency-installation

Use this when the SuperNode host is allowed to install Python packages at runtime. If
the host has no package index access, or if you want stricter control over installed
packages, preinstall the ClientApp dependencies in the SuperNode environment instead.

************************************************
 Enable dependency installation in process mode
************************************************

When running SuperLink or SuperNode with ``--isolation=process``, the runtime dependency
installation flags passed to ``flower-superlink`` or ``flower-supernode`` do not affect
the app process. In this mode, SuperExec is started separately. Enable dependency
installation on ``flower-superexec`` instead:

.. code-block:: shell
    :emphasize-lines: 4

    $ flower-superexec \
        --appio-api-address <appio-api-address> \
        --plugin-type <choice-of-plugin> \
        --allow-runtime-dependency-installation

************************
 How installation works
************************

When runtime dependency installation is enabled, Flower:

1. reads ``[project].dependencies`` from the installed Flower App,
2. creates an isolated runtime environment under ``$FLWR_HOME/runtime-envs``,
3. invokes ``uv sync`` in the app project directory using the current Python executable,
4. installs the declared app dependencies into the isolated runtime environment,
5. activates that environment for the current app process, and
6. deletes the environment after app execution is complete.

Each run gets its own environment, so concurrently running apps do not modify each
other's Python packages. The app code itself comes from the Flower App Bundle (FAB);
runtime dependency installation only installs the package dependencies needed by that
app.
