#############################
 Communicate custom Messages
#############################

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.common.Message.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.common.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.app.ConfigRecord.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

Welcome to the fourth part of the Flower federated learning tutorial. In the previous
parts of this tutorial, we introduced federated learning with PyTorch and Flower
(:doc:`part 1 <tutorial-series-get-started-with-flower-pytorch>`), we learned how
strategies can be used to customize the execution on both the server and the clients
(:doc:`part 2 <tutorial-series-use-a-federated-learning-strategy-pytorch>`) and we built
our own custom strategy from scratch (:doc:`part 3
<tutorial-series-build-a-strategy-from-scratch-pytorch>`).

In this final tutorial, we turn our attention again to the ``ClientApp`` and show how to
communicate arbitrary Python objects via a ``Message`` and how to use it on the
``ServerApp``. This can be useful if you want to send additional information between
``ClientApp <--> ServerApp`` without the need for custom protocols.

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on Flower Discuss and the Flower Slack to connect, ask questions,
    and get help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's go deeper and see how to serialize arbitrary Python objects and communicate them!
🌼

*************
 Preparation
*************

Before we begin with the actual code, let's make sure that we have everything we need.

Installing dependencies
=======================

.. note::

    If you've completed part 1 of the tutorial, you can skip this step.

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

After running it you'll notice a new directory named ``quickstart-pytorch`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-pytorch
    ├── pytorchexample
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file:

.. code-block:: shell

    $ cd quickstart-pytorch
    $ pip install -e .

*************************************
 Revisiting replying from ClientApps
*************************************

Let's remind ourselves how the communication between ``ClientApp`` and ``ServerApp``
works. A ``ClientApp`` function wrapped with ``@app.train()`` would typically return the
locally updated model parameters in addition to some metrics relevant to the training
process, such as the training loss and accuracy. In code, this would look like:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # ... prepare model, load data, train locally

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

Then, on the ``ServerApp``, the Flower strategy will automatically aggregate the
|arrayrecord_link|_ and |metricrecord_link|_ from each client into a single
``ArrayRecord`` and ``MetricRecord`` that can be used to update the global model and log
the aggregated metrics. Now, what if we wanted to send additional information from the
``ClientApp`` to the ``ServerApp``? For example, let's say we want to send how long the
execution of the ``ClientApp`` took. We can do this by adding a new metric to the
``MetricRecord``. It will also be aggregated automatically by the strategy. If you do
for example:

.. code-block:: python
    :emphasize-lines: 1,8,12,13,20

    import time


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        start_time = time.time()

        # ... prepare model, load data, train locally

        end_time = time.time()
        training_time = end_time - start_time

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            "training_time": training_time,  # New metric
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

If you'd like to communicate other types of objects and leave them out of the
aggregation process, you can use a |configrecord_link|_. In addition to integers and
floats, you can use a ``ConfigRecord`` to send strings, booleans and even bytes. In the
next section we'll learn to communicate arbitrary Python objects by first serializing
them to bytes.

*********************************
 Communicating arbitrary objects
*********************************

Let's assume the training stage of our ``ClientApp`` produces a dataclass like the one
below and we would like to communicate it to the ``ServerApp`` via the ``Message``.
Let's go ahead and define this in ``task.py``:

.. code-block:: python

    from dataclasses import dataclass


    @dataclass
    class TrainProcessMetadata:
        """Metadata about the training process."""

        training_time: float
        converged: bool
        training_losses: dict[str, float]  # e.g. { "epoch_1": 0.5, "epoch_2": 0.3 }

Now, let's see how the ``ClientApp`` can serialize this object, send it to the
``ServerApp``, make the strategy deserialize it back to the original object, and use it.

Sending from ClientApps
=======================

Let's assume our ``ClientApp`` trains the model locally and generates an instance of
``TrainProcessMetadata``. In order to send it as part of the message reply, we need to
serialize it to bytes. In this case, we can use the ``pickle`` module from the Python
standard library. We can then send the serialized object in a ``ConfigRecord`` in the
``Message`` reply. Let's see how this would look like in code:

.. warning::

    The following code is for demonstration purposes only. In real-world applications,
    since `pickle <https://docs.python.org/3/library/pickle.html>`_ can execute
    arbitrary code during unpickling, you should use a **SAFE** serialization method
    than ``pickle``, such as ``json`` or a simple custom solution if the object is not
    too complex. ``pickle`` is used here solely for simplicity.

.. code-block:: python
    :emphasize-lines: 1,10,20,22,35

    import pickle


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # ... prepare model, load data, train locally
        # The train function returns a TrainProcessMetadata object
        train_metadata = train_fn(...)
        # For example:

        # TrainProcessMetadata(
        #     training_time=123.45,
        #     converged=True,
        #     training_losses={"epoch1": 0.56, "epoch2": 0.34}
        # )

        # Serialize the TrainProcessMetadata object to bytes
        train_meta_bytes = pickle.dumps(train_metadata)
        # Construct a ConfigRecord
        config_record = ConfigRecord({"meta": train_meta_bytes})

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict(
            {
                "arrays": model_record,
                "metrics": metric_record,
                "train_metadata": config_record,
            }
        )
        return Message(content=content, reply_to=msg)

Let's see next how the strategy on the ``ServerApp`` can deserialize the object back to
its original form and use it.

Receiving on ServerApps
=======================

As you know, a Flower strategy will automatically aggregate the ``ArrayRecord`` and
``MetricRecord`` from each client. However, it will not do anything with the
``ConfigRecord`` we just sent. We can override the ``aggregate_train`` method of our
strategy to handle the deserialization and use of the ``TrainProcessMetadata`` object.

.. note::

    We override the ``aggregate_train`` method because we sent the object from a
    ``@app.train()`` function. If we had sent it from an ``@app.evaluate()`` function,
    we would override the ``aggregate_evaluate`` method instead.

Let's create a new custom strategy (or reuse the one created in part 2 and part 3 of
this tutorial) in ``server_app.py`` that extends the ``FedAdagrad`` strategy and
overrides the ``aggregate_train`` method to deserialize the ``TrainProcessMetadata``
object from each client and print the training time and convergence status:

.. code-block:: python
    :emphasize-lines: 1,8,18,19,21

    import pickle
    from dataclasses import asdict
    from typing import Iterable, Optional


    class CustomFedAdagrad(FedAdagrad):

        def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
        ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            """Aggregate ArrayRecords and MetricRecords in the received Messages."""

            for reply in replies:
                if reply.has_content():
                    # Retrieve the ConfigRecord from the message
                    config_record = reply.content["train_metadata"]
                    metadata_bytes = config_record["meta"]
                    # Deserialize it
                    train_meta = pickle.loads(metadata_bytes)
                    print(asdict(train_meta))
            # Aggregate the ArrayRecords and MetricRecords as usual
            return super().aggregate_train(server_round, replies)

Finally, we run the Flower App.

.. code-block:: shell

    $ flwr run . --stream

Plain ``flwr run .`` submits the run, prints the run ID, and returns without streaming
logs. See :doc:`how-to-run-flower-locally` for the full local workflow.

You will observe that the training metadata from each client is logged to the console of
the ``ServerApp``. If you finish embedding the creation of the ``TrainProcessMetadata``
object in the ``ClientApp``, you should see output similar to this:

.. code-block:: console

    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 25 nodes (out of 1000)
    {'training_time': 123.45, 'converged': True, 'training_losses': {'epoch1': 0.56, 'epoch2': 0.34}}
    {'training_time': 130.67, 'converged': False, 'training_losses': {'epoch1': 0.60, 'epoch2': 0.40}}
    ...

You can now use this information in your strategy logic as needed. For example, to
implement a custom aggregation method based on convergence status or to log additional
metrics.

*******
 Recap
*******

In this part of the tutorial, we've seen how to communicate arbitrary Python objects
between the ``ClientApp`` and the ``ServerApp`` by serializing them to bytes and sending
them as a ``ConfigRecord`` in a ``Message``. We also learned how to deserialize them
back to their original form on the server side and use them in a custom strategy. Note
that the steps presented here are identical if you need to serialize objects in the
strategy to send them to the clients.

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

This is the final part of the Flower tutorial (for now!), congratulations! You're now
well equipped to understand the rest of the documentation. There are many topics we
didn't cover in the tutorial, we recommend the following resources:

- `Read Flower Docs <https://flower.ai/docs/>`__
- `Check out Flower Code Examples <https://flower.ai/docs/examples/>`__
- `Use Flower Baselines for your research <https://flower.ai/docs/baselines/>`__
- `Watch Flower AI Summit 2025 videos
  <https://flower.ai/events/flower-ai-summit-2025/>`__
