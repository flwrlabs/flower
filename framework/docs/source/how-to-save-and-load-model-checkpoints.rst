:og:description: Save and load model checkpoints in Flower with custom strategies, including PyTorch checkpoints, for efficient federated learning workflows.
.. meta::
    :description: Save and load model checkpoints in Flower with custom strategies, including PyTorch checkpoints, for efficient federated learning workflows.

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

#################################
 Save and load model checkpoints
#################################

This how-to guide describes the steps to save (and load) model checkpoints in
``ClientApp`` and ``ServerApp``.

************************************************
 How to save model checkpoints in ``ClientApp``
************************************************

Model updates are saved in |arrayrecord_link|_ and transmitted between |serverapp_link|_
and |clientapp_link|_. To save model checkpoints in |clientapp_link|_, you need to
convert the |arrayrecord_link|_ into a format compatible with your ML framework (e.g.,
PyTorch, TensorFlow, or NumPy). Include the following code in your functions registered
with the ``ClientApp`` (e.g., in your training function decorated with
``@app.train()``):

PyTorch

.. code-block:: python

    # Convert ArrayRecord to PyTorch state dict.
    state_dict = arrays.to_torch_state_dict()

    # Save model weights to disk
    torch.save(state_dict, "model.pt")

TensorFlow

.. code-block:: python

    # Convert ArrayRecord to NumPy ndarrays
    ndarrays = arrays.to_numpy_ndarrays()

    # Load weights to a keras model
    model.set_weights(ndarrays)

    # Save model weights to disk
    model.save("model.keras")

NumPy

.. code-block:: python

    # Convert ArrayRecord to NumPy ndarrays
    ndarrays = arrays.to_numpy_ndarrays()

    # Save model weights to disk
    numpy.savez("model.npz", *ndarrays)

************************************************
 How to save model checkpoints in ``ServerApp``
************************************************

To save model checkpoints in |serverapp_link|_ across different FL rounds, you can
implement this in a customized ``evaluate_fn`` and pass it to the strategy's
|strategy_start_link|_ method. Here's an example showing how to save the global PyTorch
model:

.. code-block:: python

    def get_evaluate_fn(save_every_round, total_round, save_path):
        def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            # Save model every `save_every_round` round and for the last round
            if server_round != 0 and (
                server_round == total_round or server_round % save_every_round == 0
            ):
                # Convert ArrayRecord to PyTorch state dict
                state_dict = arrays.to_torch_state_dict()

                # Save model weights to disk
                torch.save(state_dict, f"{save_path}/model_{server_round}.pt")

            return MetricRecord()

        return evaluate

Then, pass it to the |strategy_start_link|_ method of the defined strategy:

.. code-block:: python

    strategy.start(
        ...,
        evaluate_fn=get_evaluate_fn(save_every_round, total_round, save_path),
    )

If you are interested, checkout the details in `Advanced PyTorch Example
<https://github.com/flwrlabs/flower/tree/main/examples/advanced-pytorch>`_.
