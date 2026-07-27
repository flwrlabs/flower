#############################
 Communicate custom Messages
#############################

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.app.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.app.ConfigRecord.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

Welcome to the next part of the Flower collaborative AI tutorial!

In the previous tutorials, you created a simulated federation on SuperGrid, ran and
customized Flower Apps, moved from the NumPy demo to the PyTorch quickstart app,
customized the strategy used by the ``ServerApp``, and then built a more customized
strategy. In this tutorial, you'll turn your attention back to the ``ClientApp`` and
learn how to communicate additional information between ``ClientApp`` and ``ServerApp``
through ``Message`` objects.

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on `Flower Discuss <https://discuss.flower.ai/>`__ or `Flower Slack
    <https://flower.ai/join-slack>`__ to introduce yourself, ask questions, and get
    help.

Let's go deeper and see how to serialize arbitrary Python objects and communicate them!
🌼

*************
 Preparation
*************

This tutorial continues from the earlier PyTorch tutorials in this series. If you
already have the ``quickstart-pytorch`` app, open that directory and continue from
there.

If you are starting here directly, install Flower and fetch the same app:

.. code-block:: shell

    # Install Flower
    $ pip install -U flwr
    # Fetch the app from Flower Hub
    $ flwr new @flwrlabs/quickstart-pytorch
    # Navigate to the app directory
    $ cd quickstart-pytorch

Constructing Messages
=====================

In Flower, the server and clients communicate by sending and receiving |message_link|_
objects. A ``Message`` carries a ``RecordDict`` as its main payload. The ``RecordDict``
is like a Python dictionary that can contain multiple records of different types. There
are three main types of records:

- |arrayrecord_link|_: Contains model parameters as a dictionary of NumPy arrays
- |metricrecord_link|_: Contains training or evaluation metrics as a dictionary of
  integers, floats, lists of integers, or lists of floats.
- |configrecord_link|_: Contains configuration parameters as a dictionary of integers,
  floats, strings, booleans, or bytes. Lists of these types are also supported.

Let's see a few examples of how to work with these types of records and, ultimately,
construct a ``RecordDict`` that can be sent over a ``Message``.

.. code-block:: python

    from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict

    # ConfigRecord can be used to communicate configs between ServerApp and ClientApp
    # They can hold scalars, but also strings and booleans
    config = ConfigRecord(
        {"batch_size": 32, "use_augmentation": True, "data-path": "/my/dataset"}
    )

    # MetricRecords expect scalar-based metrics (i.e. int/float/list[int]/list[float])
    # By limiting the types Flower can aggregate MetricRecords automatically
    metrics = MetricRecord({"accuracy": 0.9, "losses": [0.1, 0.001], "perplexity": 2.31})

    # ArrayRecord objects are designed to communicate arrays/tensors/weights from ML models
    array_record = ArrayRecord(my_model.state_dict())  # for a PyTorch model
    array_record_other = ArrayRecord(my_model.to_numpy_ndarrays())  # for other ML models

    # A RecordDict is like a dictionary that holds named records.
    # This is the main payload of a Message
    rd = RecordDict({"my-config": config, "metrics": metrics, "my-model": array_record})

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
    :emphasize-lines: 3,12,16,17,24

    # ... unchanged
    # add this to the imports
    import time

    # ... unchanged


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

    # ... unchanged imports at the top of the file
    # add this at the bottom of the imports
    from dataclasses import dataclass


    @dataclass
    class TrainProcessMetadata:
        """Metadata about the training process."""

        training_time: float
        converged: bool
        training_losses: dict[str, float]  # e.g. { "epoch_1": 0.5, "epoch_2": 0.3 }


    # ... unchanged code starting with class Net(nn.Module):

Now, let's see how the ``ClientApp`` can serialize this object, send it to the
``ServerApp``, make the strategy deserialize it back to the original object, and use it.

Sending from ClientApps
=======================

Let's assume our ``ClientApp`` trains the model locally and generates an instance of
``TrainProcessMetadata``. In order to send it as part of the message reply, we need to
serialize it to bytes. In this case, we can use the ``pickle`` module from the Python
standard library. We can then send the serialized object in a ``ConfigRecord`` in the
``Message`` reply. Let's see how this would look like in code:

The example below focuses on the additional metadata logic; keep the model and data
setup from your existing ``train`` function unchanged.

.. warning::

    The following code is for demonstration purposes only. In real-world applications,
    since `pickle <https://docs.python.org/3/library/pickle.html>`_ can execute
    arbitrary code during unpickling, you should use a safer serialization method than
    ``pickle``, such as ``json`` or a simple custom solution if the object is not too
    complex. ``pickle`` is used here solely for simplicity.

.. code-block:: python
    :emphasize-lines: 3-5,20-24,27,29,42

    # ... unchanged
    # add this to the imports
    import pickle
    from pytorchexample.task import TrainProcessMetadata
    from flwr.app import ConfigRecord

    # ... unchanged


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        start_time = time.time()

        # ... prepare model, load data, train locally
        # The train function returns the training loss
        train_loss = train_fn(...)
        # Construct a TrainProcessMetadata object
        train_metadata = TrainProcessMetadata(
            training_time=time.time() - start_time,
            converged=True,
            training_losses={"final": train_loss},
        )

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

Let's create a new custom strategy, or reuse the one created in the previous strategy
tutorials, in ``server_app.py`` that extends the ``FedAdagrad`` strategy and overrides
the ``aggregate_train`` method to deserialize the ``TrainProcessMetadata`` object from
each client and print the training time and convergence status:

.. code-block:: python
    :emphasize-lines: 3-5,8,13,22,26-27,29

    # ... make sure you have these imports.
    # ... Some may exist from previous tutorials
    import pickle
    from dataclasses import asdict
    from typing import Iterable, Optional

    # ... unchanged
    from pytorchexample.task import TrainProcessMetadata


    class CustomFedAdagrad(FedAdagrad):

        def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
        ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            """Aggregate ArrayRecords and MetricRecords in the received Messages."""

            # Convert replies to a list before iterating over them so the parent
            # strategy can still aggregate the same replies afterwards.
            replies = list(replies)
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


    # ... unchanged

Finally, we run the Flower App.

.. code-block:: shell

    $ flwr run . local --stream

Plain ``flwr run . local`` submits the run, prints the run ID, and returns without
streaming logs. See :doc:`how-to-run-flower-locally` for the full local workflow.

You will observe that the training metadata from each client is logged to the console of
the ``ServerApp``. If you finish embedding the creation of the ``TrainProcessMetadata``
object in the ``ClientApp``, you should see output similar to this:

.. code-block:: console

    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 SuperNodes (out of 50)
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
- `Watch Flower AI Summit 2026 videos
  <https://flower.ai/events/flower-ai-summit-2026/>`__
