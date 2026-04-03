:og:description: Learn how local `flwr run` uses a managed local SuperLink, how to inspect runs, stream logs, stop runs, and stop the background local SuperLink process.
.. meta::
    :description: Learn how local `flwr run` uses a managed local SuperLink, how to inspect runs, stream logs, stop runs, and stop the background local SuperLink process.

#############################################
 Run Flower Locally with a Managed SuperLink
#############################################

When you use a local profile in the :doc:`Flower configuration
<ref-flower-configuration>` with ``address = ":local:"``, ``flwr`` does not call the
simulation runtime directly. Instead, Flower starts a managed local ``flower-superlink``
on demand, submits the run through the Control API, and the local SuperLink executes the
run with the simulation runtime. The SuperLink will keep running in the background
accepting commands until you :ref:`stop it manually <stop-background-local-superlink>`.

Flower stores this managed local runtime under ``$HOME/.flwr/local-superlink``. If you
would like to change this location, set the ``FLWR_HOME`` environment variable to a
different directory.

.. note::

    The remainder of this guide assumes you have set ``[superlink.local]`` as the
    default profile in your Flower configuration. This should already be the case if you
    have installed Flower for the first time or upgraded from a previous version that
    didn't have the Flower Configuration functionality. For more information check
    :doc:`the Flower Configuration <ref-flower-configuration>` guide.

****************************
 What Flower starts for you
****************************

On the first command that needs the local Control API (e.g. ``flwr run``, ``flwr list``,
etc.), Flower starts a local ``flower-superlink`` process automatically. That process:

- listens on ``127.0.0.1:39093`` for the Control API
- binds ServerAppIo to a free local port chosen by the OS
- keeps running in the background after your command finishes
- is reused by later ``flwr run``, ``flwr list``, ``flwr log``, and ``flwr stop``
  commands

You can override the default Control API port with the ``FLWR_LOCAL_CONTROL_API_PORT``
environment variable.

**************
 Submit a run
**************

From your Flower App directory, submit a run as usual:

.. code-block:: shell

    $ flwr run .

Representative output:

.. code-block:: text

    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032

Plain ``flwr run .`` submits the run, prints the run ID, and returns. If you want to
submit the run and immediately follow the logs in the same terminal, use:

.. code-block:: shell

    $ flwr run . --stream

***********
 List runs
***********

To see all runs known to the local SuperLink:

.. code-block:: shell

    $ flwr list

To inspect one run in detail:

.. code-block:: shell

    $ flwr list --run-id 1859953118041441032

***********
 View logs
***********

To stream logs continuously:

.. code-block:: shell

    $ flwr log 1859953118041441032 --stream

To fetch the currently available logs once and return:

.. code-block:: shell

    $ flwr log 1859953118041441032 --show

Representative streamed output:

.. code-block:: text

    INFO :      Starting FedAvg strategy:
    INFO :          Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    ...

************
 Stop a run
************

To stop a submitted or running run:

.. code-block:: shell

    $ flwr stop 1859953118041441032

This stops the run only. It does **not** stop the background local SuperLink process.

*******************************
 Local runtime files and state
*******************************

The managed local SuperLink keeps its files in ``$FLWR_HOME/local-superlink/``:

- ``state.db`` stores the local SuperLink state
- ``ffs/`` stores SuperLink file artifacts
- ``superlink.log`` stores the local SuperLink process output

These files persist across local runs until you remove them yourself.

.. _stop-background-local-superlink:

*************************************
 Stop the background local SuperLink
*************************************

There is currently no dedicated ``flwr`` command to stop the managed local SuperLink
process. To stop it, first inspect the matching process and then terminate it.

macOS/Linux
===========

Inspect the process:

.. code-block:: shell

    $ ps aux | grep '[f]lower-superlink.*--control-api-address 127.0.0.1:39093'

Stop the process:

.. code-block:: shell

    $ pkill -f 'flower-superlink.*--control-api-address 127.0.0.1:39093'

Windows PowerShell
==================

Inspect the process:

.. code-block:: powershell

    PS> Get-CimInstance Win32_Process |
    >>   Where-Object {
    >>     $_.CommandLine -like '*flower-superlink*--control-api-address 127.0.0.1:39093*'
    >>   } |
    >>   Select-Object ProcessId, CommandLine

Stop the process:

.. code-block:: powershell

    PS> Get-CimInstance Win32_Process |
    >>   Where-Object {
    >>     $_.CommandLine -like '*flower-superlink*--control-api-address 127.0.0.1:39093*'
    >>   } |
    >>   ForEach-Object { Stop-Process -Id $_.ProcessId }

If you changed the local Control API port with ``FLWR_LOCAL_CONTROL_API_PORT``, replace
``39093`` in the commands above.

*****************
 Troubleshooting
*****************

If you see SQL database errors such as ``database is locked``, see :ref:`FAQ
<faq-local-superlink-db-error>`.

If a local run fails before it starts, or if the managed local SuperLink does not come
up correctly, inspect:

.. code-block:: text

    $FLWR_HOME/local-superlink/superlink.log

That log contains the output of the background ``flower-superlink`` process and is the
first place to check for startup errors, port conflicts, or runtime failures.
