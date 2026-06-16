:og:description: Guide for running Flower Apps on Flower SuperGrid Deployment and Simulation federations.
.. meta::
    :description: Guide for running Flower Apps on Flower SuperGrid Deployment and Simulation federations.

##############################
 Run Flower Apps on SuperGrid
##############################

To run a Flower App on SuperGrid, you need a federation. The federation can be a
Simulation federation or a Deployment federation. For Deployment federations, connected
SuperNodes must also be added to the federation. If you still need to create a
federation, see :doc:`how-to-create-and-manage-federations`. If you need to register and
connect SuperNodes first, see :doc:`how-to-connect-supernodes-to-supergrid`.

This guide shows two ways to run a Flower App on SuperGrid: directly from `Flower Hub
<https://flower.ai/apps/>`__, and from a Flower App project on your machine. It uses the
`@flwrlabs/quickstart-numpy <https://flower.ai/apps/flwrlabs/quickstart-numpy/>`__ app
as an example, but the same workflow applies to other Flower Apps, whether or not they
are listed on Flower Hub.

The sections below show the SuperGrid UI workflow. At the end of this page, the same
steps are shown in compact form with the Flower CLI.

*********************************
 Run Flower Apps from Flower Hub
*********************************

Running directly from Flower Hub is the quickest way to get started. Open an app page,
for example `@flwrlabs/quickstart-numpy
<https://flower.ai/apps/flwrlabs/quickstart-numpy/>`__, and click ``Run``.

.. image:: ./_static/demo_app.png
    :alt: Flower App page on Flower Hub
    :align: center
    :target: ./_static/demo_app.png

Select the federation to run the app on, then click ``Run app``.

.. image:: ./_static/run_app_button.png
    :alt: Run app dialog in SuperGrid
    :align: center
    :target: ./_static/run_app_button.png

After the run starts, open the federation in the `SuperGrid dashboard
<https://flower.ai/federations/>`__ to inspect the run status.

.. image:: ./_static/run_started_dashboard.png
    :alt: Run details page showing the progress of the app execution
    :align: center
    :target: ./_static/run_started_dashboard.png

Open the ``Logs`` tab to inspect the app logs.

.. image:: ./_static/run_logs_dashboard.png
    :alt: Logs tab showing logs from the app execution
    :align: center
    :target: ./_static/run_logs_dashboard.png

If you return to the federation page, the run appears in the list of runs for that
federation.

.. image:: ./_static/federation_dashboard_shows_run.png
    :alt: Federation details page showing the run that was launched
    :align: center
    :target: ./_static/federation_dashboard_shows_run.png

***********************************
 Run Flower apps from your machine
***********************************

You can also run a Flower App on SuperGrid from your local machine. Use this workflow
when you want to inspect or modify an app before submitting it to SuperGrid, or when you
are developing a Flower App locally and do not want to list it on Flower Hub.

This section assumes you have a Python environment set up. Install Flower with ``pip``:

.. code-block:: shell

    $ pip install -U flwr

Pull the app locally with ``flwr new``:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-numpy
    $ cd quickstart-numpy

Then, log in to SuperGrid with ``flwr login``. This opens a browser window where you can
authenticate with your SuperGrid account. Finally, run the app with ``flwr run`` and use
``--federation`` to choose the federation:

.. note::

    Replace ``@<username>/<federation-name>`` with the unique name of your federation,
    for example ``@peter123/my-federation``.

.. code-block:: shell

    # This opens a browser window where you can log in to SuperGrid.
    $ flwr login supergrid

    $ flwr run . --federation @<username>/<federation-name>

The run appears in the same SuperGrid federation dashboard.

.. image:: ./_static/second_run_started_dashboard.png
    :alt: SuperGrid dashboard showing the newly started run in the federation
    :align: center
    :target: ./_static/second_run_started_dashboard.png

Customize the run configuration
===============================

Flower Apps can define default runtime settings in the ``[tool.flwr.app.config]``
section of their ``pyproject.toml``. The ``quickstart-numpy`` app used in this guide
defines ``num-server-rounds`` to control how many rounds its ``ServerApp`` runs:

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3

Apps can define additional settings in this section and read them from the app code, for
example learning rates, model sizes, batch sizes, or dataset-specific options.

You can override app-defined values for a single run with the ``--run-config`` flag. The
following command runs ``quickstart-numpy`` on SuperGrid with ``num-server-rounds`` set
to ``5``:

.. code-block:: shell

    $ flwr run . supergrid --federation @<username>/<federation-name> \
        --run-config "num-server-rounds=5"

.. note::

    For more details on Flower App configuration in ``pyproject.toml``, see
    :doc:`how-to-configure-pyproject-toml`.

**********
 Advanced
**********

Everything shown above in the SuperGrid UI can also be done with the :doc:`Flower CLI
<ref-api-cli>`.

Log in to SuperGrid:

.. code-block:: shell

    $ flwr login supergrid

Run an app directly from Flower Hub:

.. code-block:: shell

    $ flwr run @flwrlabs/quickstart-numpy supergrid \
        --federation @<username>/<federation-name> \
        --stream

Inspect runs and logs from the terminal:

.. code-block:: shell

    # List all your runs (across federations)
    $ flwr list supergrid
    # Show additional details of a run
    $ flwr list --run-id <run-id> supergrid
    # Show federation details (including runs) for a specific federation
    $ flwr federation list supergrid --federation @<username>/<federation-name>
    # Show logs of a run
    $ flwr log <run-id> supergrid
