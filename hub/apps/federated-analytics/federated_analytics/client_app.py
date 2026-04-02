"""federated_analytics: A Flower / Federated Analytics app."""

import warnings

from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_analytics.task import query_database

warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context) -> Message:
    """Query PostgreSQL database and report aggregated results to `ServerApp`."""

    # Get database connection details from node config
    db_url: str = context.node_config.get(
        "db-url",
        "postgresql+psycopg://flwrlabs:flwrlabs@localhost:5432/flwrlabs",
    )
    table_name: str = context.node_config.get("table-name", "person_measurements")

    selected_features: list[str] = msg.content["config"]["selected_features"]
    feature_aggregation: list[str] = msg.content["config"]["feature_aggregation"]

    # Query database
    df = query_database(
        db_url=db_url,
        table_name=table_name,
        selected_features=selected_features,
    )

    # Compute aggregation metrics
    metrics = {}
    for feature in selected_features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset columns.")

        for agg in feature_aggregation:
            if agg == "mean":
                metrics[f"{feature}_{agg}_sum"] = sum(df[feature])
                metrics[f"{feature}_{agg}_count"] = len(df[feature])
            elif agg == "std":
                metrics[f"{feature}_{agg}_sum"] = sum(df[feature])
                metrics[f"{feature}_{agg}_count"] = len(df[feature])
                metrics[f"{feature}_{agg}_sum_sqd"] = sum(df[feature] ** 2)
            else:
                raise ValueError(f"Aggregation method '{agg}' not recognized.")

    reply_content = RecordDict({"query_results": MetricRecord(metrics)})

    return Message(reply_content, reply_to=msg)
