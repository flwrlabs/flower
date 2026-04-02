.. |context_link| replace:: ``Context``

.. _context_link: https://flower.ai/docs/framework/ref-api/flwr.app.Context.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: https://flower.ai/docs/framework/ref-api/flwr.clientapp.ClientApp.html

Generate Demo Data for SuperNodes
=================================

In Flower simulations, datasets are downloaded and partitioned on-the-fly.
While convenient for prototyping, production deployments require SuperNodes
to have pre-existing data on disk. This ensures immediate startup, data
persistence across restarts, and a setup that mirrors real-world federated
AI where each node owns its local data.

Flower Datasets enables you to generate pre-partitioned datasets for
deployment prototyping using the Flower Datasets CLI. By materializing partitions to disk ahead of time, each
SuperNode can read from its designated partition—just as it would in
production.

.. note::

   This guide is intended for generating demo data for testing deployments. For
   production deployments, ensure that each SuperNode has access to its own
   local data partition.


Using the Flower Datasets CLI
-----------------------------

The ``flwr-datasets create`` command enables you to download a dataset,
partition it, and save each partition to disk in a single step. For complete
details on all available options, see the :doc:`ref-api-cli`.

For example, to generate demo data from the `MNIST dataset <https://huggingface.co/datasets/ylecun/mnist>`_ with five
partitions and store the result in the ``./demo_data`` directory (it will be created if it doesn't exist), run the
following command in your terminal:

.. code-block:: bash

   # flwr-datasets create <dataset> --num-partitions <n> --out-dir <dir>
   flwr-datasets create ylecun/mnist --num-partitions 5 --out-dir demo_data

   # The output will look similar to this:
   Saving the dataset (1/1 shards): 100%|████████████| 12000/12000 [00:00<00:00, 3085.94 examples/s]
   Saving the dataset (1/1 shards): 100%|████████████| 12000/12000 [00:00<00:00, 4006.59 examples/s]
   Saving the dataset (1/1 shards): 100%|████████████| 12000/12000 [00:00<00:00, 4001.21 examples/s]
   Saving the dataset (1/1 shards): 100%|████████████| 12000/12000 [00:00<00:00, 4010.60 examples/s]
   Saving the dataset (1/1 shards): 100%|████████████| 12000/12000 [00:00<00:00, 3990.48 examples/s]
   🎊 Created 5 partitions for 'ylecun/mnist' in '/path/to/demo_data'

The above command generates the following directory structure:

.. code-block:: text

   demo_data/
   ├── partition_0/
   │   ├── data-00000-of-00001.arrow
   │   ├── dataset_info.json
   │   └── state.json
   ...
   └── partition_4/
       ├── data-00000-of-00001.arrow
       ├── dataset_info.json
       └── state.json


Using Generated Demo Data in SuperNodes
---------------------------------------

Once you have generated the partitions, each SuperNode can be configured to
load its designated partition. The recommended approach is to pass the
partition path as a node configuration parameter when starting the SuperNode.

Passing the Data Path to a SuperNode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``--node-config`` flag to specify the path to the partition when 
launching a SuperNode. In the example below, the selected key ``data-path`` 
is arbitrary and provided for illustration only; any application-appropriate 
key may be used.

.. code-block:: bash

   flower-supernode \
       --insecure \
       --superlink <SUPERLINK-FLEET-API> \
       --node-config="data-path='/path/to/demo_data/partition_0'"


Loading the Dataset in Your ClientApp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your |clientapp_link|_, you can access the configured data path through the
|context_link|_ and load the dataset using the
``load_from_disk`` function from the Huggingface ``datasets`` module:

.. code-block:: python
   :emphasize-lines: 12,15

   from flwr.app import Context, Message
   from flwr.clientapp import ClientApp
   from datasets import load_from_disk

   app = ClientApp()


   @app.train()
   def train(msg: Message, context: Context) -> Message:
       """Train the model on local data."""
       # Retrieve the data path from node configuration
       dataset_path = context.node_config["data-path"]

       # Load the partition from disk
       partition = load_from_disk(dataset_path)

       # Use the dataset for training
       # ...


.. tip::

   For a complete guide on how to run Flower SuperNodes, refer to the
   `Deployment Runtime Documentation <https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html>`_.

