#####################
 Use CLI JSON output
#####################

The `Flower CLI <ref-api-cli.html>`_ can return JSON output for automation and
integration with other tools.

.. note::

    JSON output is available for the commands documented here because they operate
    through the SuperLink Control API. This includes remote SuperLinks as well as the
    managed local SuperLink used by local simulation profiles marked with ``address =
    ":local:"``.

This guide shows JSON output for:

- |flwr_run|
- |flwr_list|
- |flwr_stop|

.. |flwr_run| replace:: ``flwr run``

.. |flwr_list| replace:: ``flwr list``

.. |flwr_stop| replace:: ``flwr stop``

.. _flwr_list: ref-api-cli.html#flwr-list

.. _flwr_run: ref-api-cli.html#flwr-run

.. _flwr_stop: ref-api-cli.html#flwr-stop

**************************
 ``flwr run`` JSON output
**************************

The |flwr_run| command submits a Flower App run. For a local app, the CLI first builds a
FAB and then starts the run through the Control API.

Representative default output:

.. code-block:: bash

    $ flwr run . local --stream
    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032
    ...

To return structured JSON instead, use ``--format json``:

.. code-block:: bash

    $ flwr run . local --format json
    {
      "success": true,
      "run-id": "1859953118041441032",
      "fab-id": "flwrlabs/myawesomeapp",
      "fab-name": "myawesomeapp",
      "fab-version": "1.0.0",
      "fab-hash": "014c8eb3",
      "fab-filename": "flwrlabs.myawesomeapp.1-0-0.014c8eb3.fab"
    }

The |flwr_run| JSON output contains:

- ``success``: ``true`` if the command succeeded
- ``run-id``: the submitted run ID
- ``fab-id``: the Flower App identifier
- ``fab-name``: the Flower App name
- ``fab-version``: the Flower App version
- ``fab-hash``: the short FAB hash
- ``fab-filename``: the built FAB filename

If the command fails, the JSON output contains ``success: false`` and ``error-message``.

***************************
 ``flwr list`` JSON output
***************************

The |flwr_list| command queries runs from the current SuperLink connection.

Representative default output:

.. code-block:: bash

    $ flwr list
    Listing all runs...
    ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Run ID              ┃ FAB              ┃ Status             ┃ Elapsed  ┃ Pending At         ┃ Running At         ┃ Finished At        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ 1859953118041441032 │ flwrlabs/myawes… │ finished:completed │ 00:00:55 │ 2024-12-16         │ 2024-12-16         │ 2024-12-16         │
    │                     │ (v1.0.0)         │                    │          │ 11:12:33Z          │ 11:12:33Z          │ 11:13:28Z          │
    ├─────────────────────┼──────────────────┼────────────────────┼──────────┼────────────────────┼────────────────────┼────────────────────┤
    │ 1420074065701160142 │ flwrlabs/myawes… │ running            │ 00:00:09 │ 2024-12-16         │ 2024-12-16         │ N/A                │
    │ 0                   │ (v1.0.0)         │                    │          │ 12:18:39Z          │ 12:18:39Z          │                    │
    └─────────────────────┴──────────────────┴────────────────────┴──────────┴────────────────────┴────────────────────┴────────────────────┘

To return structured JSON instead:

.. code-block:: bash

    $ flwr list --format json
    {
      "success": true,
      "runs": [
        {
          "run-id": "1859953118041441032",
          "federation": "",
          "fab-id": "flwrlabs/myawesomeapp",
          "fab-name": "myawesomeapp",
          "fab-version": "1.0.0",
          "fab-hash": "014c8eb3",
          "status": "finished:completed",
          "elapsed": 55.0,
          "pending-at": "2024-12-16 11:12:33Z",
          "starting-at": "2024-12-16 11:12:33Z",
          "running-at": "2024-12-16 11:12:33Z",
          "finished-at": "2024-12-16 11:13:28Z",
          "network-traffic": {
            "inbound-bytes": 12345,
            "outbound-bytes": 6789,
            "total-bytes": 19134
          },
          "compute-time": {
            "serverapp-seconds": 5.2,
            "clientapp-seconds": 42.7,
            "total-seconds": 47.9
          }
        },
        {
          "run-id": "14200740657011601420",
          "federation": "",
          "fab-id": "flwrlabs/myawesomeapp",
          "fab-name": "myawesomeapp",
          "fab-version": "1.0.0",
          "fab-hash": "014c8eb3",
          "status": "running",
          "elapsed": 9.0,
          "pending-at": "2024-12-16 12:18:39Z",
          "starting-at": "2024-12-16 12:18:39Z",
          "running-at": "2024-12-16 12:18:39Z",
          "finished-at": "N/A",
          "network-traffic": {
            "inbound-bytes": 4567,
            "outbound-bytes": 2345,
            "total-bytes": 6912
          },
          "compute-time": {
            "serverapp-seconds": 0.6,
            "clientapp-seconds": 8.1,
            "total-seconds": 8.7
          }
        }
      ]
    }

Each entry under ``runs`` contains:

- ``run-id``: the run ID
- ``federation``: the federation identifier, if any
- ``fab-id`` / ``fab-name`` / ``fab-version`` / ``fab-hash``: Flower App metadata
- ``status``: the current run status
- ``elapsed``: elapsed run time in seconds
- ``pending-at`` / ``starting-at`` / ``running-at`` / ``finished-at``: run timestamps
- ``network-traffic``: inbound, outbound, and total bytes
- ``compute-time``: ServerApp, ClientApp, and total compute time in seconds

To return the detail view for a single run, use:

.. code-block:: bash

    $ flwr list --run-id 1859953118041441032 --format json

This returns the same top-level structure with one entry in ``runs``.

***************************
 ``flwr stop`` JSON output
***************************

The |flwr_stop| command stops a submitted or running run by run ID.

Representative default output:

.. code-block:: bash

    $ flwr stop 1859953118041441032
    Stopping run ID 1859953118041441032...
    Run 1859953118041441032 successfully stopped.

To return structured JSON instead:

.. code-block:: bash

    $ flwr stop 1859953118041441032 --format json
    {
      "success": true,
      "run-id": "1859953118041441032"
    }

If the command fails, the JSON output contains ``success: false`` and ``error-message``.
