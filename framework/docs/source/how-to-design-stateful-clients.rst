:og:description: Build stateful ClientApps in Flower with context objects, enabling efficient simulations and deployments.
.. meta::
    :description: Build stateful ClientApps in Flower with context objects, enabling efficient simulations and deployments.

############################
 Design stateful ClientApps
############################

.. _array: ref-api/flwr.common.Array.html

.. _arrayrecord: ref-api/flwr.common.ArrayRecord.html

.. _clientapp: ref-api/flwr.client.ClientApp.html

.. _configrecord: ref-api/flwr.common.ConfigRecord.html

.. _context: ref-api/flwr.common.Context.html

.. _metricrecord: ref-api/flwr.common.MetricRecord.html

.. _numpyclient: ref-api/flwr.client.NumPyClient.html

.. _recorddict: ref-api/flwr.common.RecordDict.html#recorddict

By design, ClientApp_ objects are stateless. This means that the ``ClientApp`` object is
recreated each time a new ``Message`` is to be processed. This behavior is identical
with Flower's Simulation Runtime and Deployment Runtime. For the former, it allows us to
simulate the running of a large number of nodes on a single machine or across multiple
machines. For the latter, it enables each ``SuperNode`` to be part of multiple runs,
each running a different ``ClientApp``.

When a ``ClientApp`` is executed it receives a Context_. This context is unique for each
``ClientApp``, meaning that subsequent executions of the same ``ClientApp`` from the
same node will receive the same ``Context`` object. In the ``Context``, the ``.state``
attribute (of type RecordDict_) can be used to store information that you would like the
``ClientApp`` to have access to for the duration of the run. This could be anything from
intermediate results such as the history of training losses (e.g. as a list of ``float``
values with a new entry appended each time the ``ClientApp`` is executed), certain parts
of the model that should persist on the client side, or some other arbitrary Python
objects. These items would need to be serialized before saving them into the context.

*******************************
 Saving metrics to the context
*******************************

This section will demonstrate how to save metrics such as accuracy/loss values to the
Context_ so they can be used in subsequent executions of the ``ClientApp``.

Let's begin with a simple setting in which ``ClientApp`` is defined as follows. The
``train()`` function only generates a random number, prints it, and return an empty
message.

.. tip::

    You can create a PyTorch project with ready-to-use ``ClientApp`` and other
    components by running ``flwr new``.

.. code-block:: python

    import random
    from flwr.app import Context, Message, RecordDict
    from flwr.clientapp import ClientApp

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Generate a random integer between 0 and 10
        n = random.randint(0, 10)
        print(n)
        return Message(RecordDict(), reply_to=msg)

With the minimal ``ClientApp`` above, each time a ``Message`` is addressed to this
``train`` function, a new random integer will be generated and printed. Let's say we
want to save that randomly generated integer and append it to a list that persists in
the ``Context``. This way, each time the function executes,, it prints the history of
random integers. Let's see how this looks in code:

.. tip::

    Recall, the ``state`` attribute of a ``Context`` object is of type RecordDict_,
    which is a special dictionary for different types of records available in Flower.
    This means that you can save to it not just MetricRecord_ as in the example below,
    but also ArrayRecord_ and ConfigRecord_ objects.

.. code-block:: python

    import random
    from flwr.app import Context, Message, RecordDict
    from flwr.clientapp import ClientApp

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Generate a random integer between 0 and 10
        n = random.randint(0, 10)
        print(n)

        # Append to list in context or initialize if it doesn't exist
        if "random-metrics" not in context.state:
            # Initialize MetricRecord in state
            context.state["random-metrics"] = MetricRecord({"random-ints": []})

        # Append to record
        context.state["random-metrics"]["random-ints"].append(n)

        # Print history
        print(context.state["random-metrics"])
        return Message(RecordDict(), reply_to=msg)

If you run a Flower App including the above logic in your ``ClientApp`` and having just
two clients in your federation sampled in each round, you'll see an output similar to
the one below. See how after each round the ``random-metrics`` record in the ``Context``
gets one additional integer? Note that, in Simulation Runtime, the order of log messages
may change each round due to the random ordering of simulated clients.

.. code-block:: shell

    # round 1
    config_records={'random-metrics': {'random-ints': [2]}}
    config_records={'random-metrics': {'random-ints': [7]}}

    # round 2
    config_records={'random-metrics': {'random-ints': [2, 5]}}
    config_records={'random-metrics': {'random-ints': [7, 4]}}

    # round 3
    config_records={'random-metrics': {'random-ints': [2, 5, 1]}}
    config_records={'random-metrics': {'random-ints': [7, 4, 2]}}

****************************************
 Saving model parameters to the context
****************************************

Using ConfigRecord_ or MetricRecord_ to save "simple" components is fine (e.g., float,
integer, boolean, string, bytes, and lists of these types. Note that MetricRecord_ only
supports float, integer, and lists of these types). Flower has a specific type of
record, an ArrayRecord_, for storing model parameters, or more generally, data arrays.

Let's see a couple of examples of how to save NumPy arrays first and then how to save
parameters of PyTorch and TensorFlow models.

.. note::

    The examples below omit the definition of a ``ClientApp`` to keep the code blocks
    concise. To make use of ``ArrayRecord`` objects in your ``ClientApp`` you can follow
    the same principles as outlined earlier.

Saving NumPy arrays to the context
==================================

Elements stored in an ``ArrayRecord`` are of type Array_, which is a data structure that
holds ``bytes`` and metadata that can be used for deserialization. Let's see how to
create an ``Array`` from a NumPy array and insert it into an ``ArrayRecord``.

.. note::

    Array_ objects carry bytes as their main payload and additional metadata to use for
    deserialization. You can also implement your own serialization/deserialization.

Let's see how to use those functions to store a NumPy array into the context.

.. code-block:: python

    import numpy as np
    from flwr.app import Array, ArrayRecord, Context


    # Let's create a simple NumPy array
    arr_np = np.random.randn(3, 3)

    # If we print it
    # array([[-1.84242409, -1.01539537, -0.46528405],
    #        [ 0.32991896,  0.55540414,  0.44085534],
    #        [-0.10758364,  1.97619858, -0.37120501]])

    # Now, let's serialize it and construct an Array
    arr = Array(arr_np)

    # If we print it (note the binary data)
    # Array(dtype='float64', shape=[3, 3], stype='numpy.ndarray', data=b'\x93NUMPY\x01\x00v\x00...)

    # It can be inserted in an ArrayRecord like this
    arr_record = ArrayRecord()
    arr_record["my_array"] = arr
    # You can also do it via the constructor
    # arr_record = ArrayRecord({"my_array": arr})

    # If you don't need the keys, you can also pass a list of Numpy arrays
    # arr_record = ArrayRecord([arr_np])

    # Then, it can be added to the state in the context
    context.state["some_parameters"] = arr_record

To extract the data in an ``ArrayRecord``, you just need to deserialize the array of
interest. For example, following the example above:

.. code-block:: python

    # Get Array from context
    arr = context.state["some_parameters"]["my_array"]

    # If you constructed the ArrayRecord with a list of Numpy, then do
    # arr = context.state["some_parameters"].to_numpy_ndarrays()[0]  # get first array

    # Deserialize it
    arr_deserialized = arr.numpy()

    # If we print it (it should show the exact same values as earlier)
    # array([[-1.84242409, -1.01539537, -0.46528405],
    #        [ 0.32991896,  0.55540414,  0.44085534],
    #        [-0.10758364,  1.97619858, -0.37120501]])

Saving PyTorch parameters to the context
========================================

Flower offers one-liner utilities to convert PyTorch model parameters to/from
``ArrayRecord`` objects. Let's see how to do that.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from flwr.app import ArrayRecord


    class Net(nn.Module):
        """A very simple model"""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 5)
            self.fc = nn.Linear(1024, 10)

        def forward(self, x):
            x = F.relu(self.conv(x))
            return self.fc(x)


    # Instantiate model as usual
    model = Net()

    # Save the state_dict into a single ArrayRecord
    arr_record = ArrayRecord(model.state_dict())

    # Add to a context
    context.state["net_parameters"] = arr_record

Let's say now you want to apply the parameters stored in your context to a new instance
of the model (as it happens each time a ``ClientApp`` is executed). You will need to:

1. Retrieve the ``ArrayRecord`` from the context
2. Construct a ``state_dict`` and load it

.. code-block:: python

    state_dict = {}
    # Extract record from context
    arr_record = context.state["net_parameters"]

    # Deserialize the parameters
    state_dict = arr_record.to_torch_state_dict()

    # Apply state dict to a new model instance
    model_ = Net()
    model_.load_state_dict(state_dict)
    # now this model has the exact same parameters as the one created earlier
    # You can verify this by doing
    for p, p_ in zip(model.state_dict().values(), model_.state_dict().values()):
        assert torch.allclose(p, p_), "`state_dict`s do not match"

And that's it! Recall that even though this example shows how to store the entire
``state_dict`` in an ``ArrayRecord``, you can just save part of it. The process would be
identical, but you might need to adjust how it is loaded into an existing model using
PyTorch APIs.

Saving Tensorflow/Keras parameters to the context
=================================================

Follow the same steps as done above but replace the ``state_dict`` logic with simply
`get_weights() <https://www.tensorflow.org/api_docs/python/tf/keras/Layer#get_weights>`_
to convert the model parameters to a list of NumPy arrays that can then be saved into an
``ArrayRecord``. Then, after deserialization, use `set_weights()
<https://www.tensorflow.org/api_docs/python/tf/keras/Layer#set_weights>`_ to apply the
new parameters to a model.

.. code-block:: python

    import tensorflow as tf
    from flwr.app import ArrayRecord

    # Define a simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    # Save model weights into an ArrayRecord and add to a context
    context.state["model_weights"] = ArrayRecord(model.get_weights())

    ...

    # Extract record from context and apply to the model
    model.set_weights(context.state["model_weights"].to_numpy_ndarrays())
