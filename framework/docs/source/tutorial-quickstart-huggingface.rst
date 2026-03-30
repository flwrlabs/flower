:og:description: Learn how to train a large language model on the IMDB dataset using federated learning with Flower and 🤗 Hugging Face in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a large language model on the IMDB dataset using federated learning with Flower and 🤗 Hugging Face in this step-by-step tutorial.

.. _quickstart-huggingface:

Quickstart 🤗 Transformers
==========================

In this federated learning tutorial we will learn how to train a large language model
(LLM) on the `IMDB <https://huggingface.co/datasets/stanfordnlp/imdb>`_ dataset using
Flower and the 🤗 Hugging Face Transformers library. It is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+🤗 Hugging Face project. It will
generate all the files needed to run a federation of 10 nodes using |fedavg|_. By
default, the generated app uses a local simulation profile that ``flwr run`` submits to
a managed local SuperLink, which then executes the run with the Flower Simulation
Runtime. The dataset will be partitioned using
|flowerdatasets|_'s |iidpartitioner|_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr[simulation]

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-huggingface

After running it you'll notice a new directory named ``quickstart-huggingface`` has been created.
It should have the following structure:

.. code-block:: shell

    quickstart-huggingface
    ├── huggingface_example
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

If you haven't yet installed the project and its dependencies, you can do so by:

.. code-block:: shell

    # From the directory where your pyproject.toml is
    $ pip install -e .

To run the project, do:

.. code-block:: shell

    # Run with default arguments and stream logs
    $ flwr run . --stream

Plain ``flwr run .`` submits the run, prints the run ID, and returns without streaming
logs. For the full local workflow, see :doc:`how-to-run-flower-locally`.

With default arguments you will see streamed output like this:

.. code-block:: shell

    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.6974}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'val_loss': 0.0223, 'val_accuracy': 0.5024}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 151.02s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also run the project with GPU as follows:

.. code-block:: shell

    # Run with default arguments
    $ flwr run . localhost-gpu --stream

This will use the default arguments where each ``ClientApp`` will use 4 CPUs and at most
4 ``ClientApp``\s will run in a given GPU.

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 fraction-train=0.2"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

The Data
--------

This tutorial uses |flowerdatasets|_ to easily download and partition the `IMDB
<https://huggingface.co/datasets/stanfordnlp/imdb>`_ dataset. In this example you'll
make use of the |iidpartitioner|_ to generate ``num_partitions`` partitions. You can
choose |otherpartitioners|_ available in Flower Datasets. To tokenize the text, we will
also load the tokenizer from the pre-trained Transformer model that we'll use during
training - more on that in the next section. Each ``ClientApp`` will call this function
to create dataloaders with the data that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="stanfordnlp/imdb",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)


    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True
        )


    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

The Model
---------

We will leverage 🤗 Hugging Face to federate the training of language models over
multiple clients using Flower. More specifically, we will fine-tune a pre-trained
Transformer model (|berttiny|_) for sequence classification over the dataset of IMDB
ratings. The end goal is to detect if a movie rating is positive or negative. If you
have access to larger GPUs, feel free to use larger models!

.. code-block:: python

    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

Note that here, ``model_name`` is a string that will be loaded from the ``Context`` in
the ClientApp and ServerApp.

In addition to loading the pretrained model weights and architecture, we also include
two utility functions to perform both training (i.e. ``train()``) and evaluation (i.e.
``test()``) using the above model. These functions should look fairly familiar if you
have some prior experience with PyTorch. Note these functions do not have anything
specific to Flower. That being said, the training function will normally be called, as
we'll see later, from a Flower client passing its own data. In summary, your clients can
use standard training/testing functions to perform local training or evaluation:

.. code-block:: python

    def train_fn(net, trainloader, epochs, device) -> None:
        optimizer = AdamW(net.parameters(), lr=5e-5)
        net.train()
        for _ in range(epochs):
            for batch in trainloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = net(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


    def test_fn(net, testloader, device) -> tuple[Any | float, Any]:
        metric = load_metric("accuracy")
        loss = 0
        net.eval()
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = net(**batch)
            logits = outputs.logits
            loss += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        loss /= len(testloader.dataset)
        accuracy = metric.compute()["accuracy"]
        return loss, accuracy

The ClientApp
-------------

The main changes we have to make to use 🤗 Hugging Face with Flower have to do with
converting the |arrayrecord_link|_ received in the |message_link|_ into a PyTorch
``state_dict`` and vice versa when generating the reply ``Message`` from the ClientApp.
We can make use of the built-in methods in the ``ArrayRecord`` to make these
conversions:

.. code-block:: python

    # Load the model
    model = get_model(model_name)

    # Extract ArrayRecord from Message and convert to PyTorch state_dict
    arrays = msg.content["arrays"]
    # Load state_dict into the model
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    # ... do some training

    # Convert state_dict back into an ArrayRecord
    model_record = ArrayRecord(model.state_dict())

The rest of the functionality is directly inspired by the centralized case. The
|clientapp_link|_ comes with three core methods (``train``, ``evaluate``, and ``query``)
that we can implement for different purposes. For example: ``train`` to train the
received model using the local data; ``evaluate`` to assess its performance of the
received model on a validation set; and ``query`` to retrieve information about the node
executing the ``ClientApp``. In this tutorial we will only make use of ``train`` and
``evaluate``.

Let's see how the ``train`` method can be implemented. It receives as input arguments a
|message_link|_ from the ``ServerApp``. By default it carries:

- an ``ArrayRecord`` with the arrays of the model to federate. By default they can be
  retrieved with key ``"arrays"`` when accessing the message content.
- a ``ConfigRecord`` with the configuration sent from the ``ServerApp``. By default it
  can be retrieved with key ``"config"`` when accessing the message content.

The ``train`` method also receives the ``Context``, giving access to configs for your
run and node. The run config hyperparameters are defined in the ``pyproject.toml`` of
your Flower App. The node config can only be set when running Flower with the Deployment
Runtime and is not directly configurable during simulations.

.. code-block:: python

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context) -> Message:
        """Train the model on local data."""

        # Get this client's dataset partition
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        model_name = context.run_config["model-name"]
        trainloader, _ = load_data(partition_id, num_partitions, model_name)

        # Load model
        model = get_model(model_name)

        # Initialize it with the received weights
        arrays = msg.content["arrays"]
        model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train the model on local data
        train_fn(model, trainloader, epochs=1, device=device)

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = MetricRecord({"num-examples": len(trainloader)})
        # Construct RecordDict and add ArrayRecord and MetricRecord
        content = RecordDict({"arrays": model_record, "metrics": metrics})
        return Message(content=content, reply_to=msg)

The ``@app.evaluate()`` method would be near identical with two exceptions: (1) the
model is not locally trained, instead it is used to evaluate its performance on the
locally held-out validation set; (2) including the model in the reply Message is no
longer needed because it is not locally modified.

The ServerApp
-------------

To construct a |serverapp_link|_ we define its ``@app.main()`` method. This method
receive as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the |fedavg|_ and configure it with a specific value of
``fraction_train`` which is read from the run config. You can find the default value
defined in the ``pyproject.toml``. Then, the execution of the strategy is launched when
invoking its |strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to be federated.
- a ``ConfigRecord`` with the training hyperparameters to be sent to the clients. The
  strategy will also insert the current round number in this config before sending it to
  the participating nodes.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:

        # Define model to federate and extract parameters
        model_name = context.run_config["model-name"]
        model = get_model(model_name)
        arrays = ArrayRecord(model.state_dict())

        # Instantiate strategy
        fraction_train = context.run_config["fraction-train"]
        fraction_evaluate = context.run_config["fraction-evaluate"]
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )

        num_rounds = context.run_config["num-server-rounds"]
        # Start the strategy
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

Note the ``start`` method of the strategy returns a result object. This object contains
all the relevant information about the FL process, including the final model weights as
an ``ArrayRecord``, and federated training and evaluation metrics as ``MetricRecords``.
You can easily log the metrics using Python's `pprint
<https://docs.python.org/3/library/pprint.html>`_ and save the global model `state_dict`
using ``torch.save``.

Congratulations! You've successfully built and run your first federated learning system
for an LLM.


.. tip::

    Check the :doc:`how-to-run-simulations` documentation to learn
    more about how to configure and run Flower simulations.


.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_hf_link|_ in the Flower GitHub repository. For a comprehensive example
    of a federated fine-tuning of an LLM with Flower, refer to the |flowertune|_ example
    in the Flower GitHub repository.

.. |quickstart_hf_link| replace:: ``examples/quickstart-huggingface``

.. |fedavg| replace:: ``FedAvg``

.. |iidpartitioner| replace:: ``IidPartitioner``

.. |otherpartitioners| replace:: other partitioners

.. |berttiny| replace:: ``bert-tiny``

.. |flowerdatasets| replace:: Flower Datasets

.. |flowertune| replace:: FlowerTune LLM

.. _berttiny: https://huggingface.co/prajjwal1/bert-tiny

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html

.. _flowerdatasets: https://flower.ai/docs/datasets/

.. _flowertune: https://github.com/flwrlabs/flower/tree/main/examples/flowertune-llm

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner

.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

.. _quickstart_hf_link: https://github.com/flwrlabs/flower/tree/main/examples/quickstart-huggingface

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start
