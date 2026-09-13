:og:description: Aggregate custom evaluation results from federated clients in Flower using a callable function that applies weighted averaging for metrics like accuracy.
.. meta::
    :description: Aggregate custom evaluation results from federated clients in Flower using a callable function that applies weighted averaging for metrics like accuracy.

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.app.MetricRecord.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |recorddict_link| replace:: ``RecordDict``

.. _recorddict_link: ref-api/flwr.app.RecordDict.html

##############################
 Aggregate evaluation results
##############################

Flower strategies (e.g. |fedavg_link|_ and all that derive from it) automatically
aggregate the metrics in the |metricrecord_link|_ in the ``Messages`` replied by the
``ClientApps``. By default, a weighted aggregation is performed for all metrics using as
weight the value assigned to the ``weighted_by_key`` attribute of a strategy.

When constructing your strategy, you can set both the key used to perform weighted
aggregation but also the callback function used to aggregate metrics.

.. note::

    By default, Flower strategies use as ``weighted_by_key="num-examples"``. If you are
    interested, see the full implementation of how the default weighted aggregation
    callback works `here
    <https://github.com/flwrlabs/flower/blob/b174b2e02bb34cae9ba9f2a124c610a844cee870/framework/py/flwr/serverapp/strategy/strategy_utils.py#L109>`_.

.. code-block:: python

    from flwr.serverapp.strategy import FedAvg
    from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

    strategy = FedAvg(
        # ... other parameters ...
        weighted_by_key="your-key",  # Key to use for weighted averaging
        evaluate_metrics_aggr_fn=my_metrics_aggr_function,  # Custom aggregation function
    )

Let's see how we can define a custom aggregation function for ``MetricRecord`` objects
received in the reply of an evaluation round.

.. note::

    Note that Flower strategies also have a ``train_metrics_aggr_fn`` attribute that
    allows you to define a custom aggregation function for received ``MetricRecord``
    objects in reply messages of a training round. By default, it performs weighted
    averaging using the value assigned to the ``weighted_by_key`` exactly as the
    ``evaluate_metrics_aggr_fn`` presented earlier.

*********************************************
 Using a custom metrics aggregation function
*********************************************

The ``evaluate_metrics_aggr_fn`` can be customized to support any evaluation results
aggregation logic you need. Its definition is:

.. code-block:: python

    Callable[[list[RecordDict], str], MetricRecord]

It takes a list of |recorddict_link|_ and a weighting key as inputs and returns a
|metricrecord_link|_. For example, the function below extracts and returns the minimum
value for each metric key across all |message_link|_:

.. code-block:: python

    from flwr.app import MetricRecord, RecordDict


    def custom_metrics_aggregation_fn(
        records: list[RecordDict], weighting_metric_name: str
    ) -> MetricRecord:
        """Extract the minimum value for each metric key."""
        aggregated_metrics = MetricRecord()

        # Track current minimum per key in a plain dict,
        # then copy into MetricRecord at the end
        mins = {}

        for record in records:
            for record_item in record.metric_records.values():
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        # We exclude the weighting key from the aggregated MetricRecord
                        continue

                    if key in mins:
                        if value < mins[key]:
                            mins[key] = value
                    else:
                        mins[key] = value

        for key, value in mins.items():
            aggregated_metrics[key] = value

        return aggregated_metrics
