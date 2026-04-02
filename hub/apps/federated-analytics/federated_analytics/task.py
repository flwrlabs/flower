"""federated_analytics: A Flower / Federated Analytics app."""

from collections.abc import Iterable
from logging import INFO

import numpy as np
import pandas as pd
from flwr.app import Message
from flwr.common.logger import log
from sqlalchemy import create_engine


def query_database(
    db_url: str,
    table_name: str,
    selected_features: list[str],
) -> pd.DataFrame:
    """Query PostgreSQL database and return selected features as DataFrame.

    Args:
        db_url: Database connection URL (e.g., postgresql+psycopg://user:pass@host:port/dbname)
        table_name: Name of the table to query
        selected_features: List of column names to select

    Returns:
        DataFrame containing the selected features
    """
    # Create database connection
    engine = create_engine(db_url)

    # Execute query and load selected features into DataFrame
    df = pd.read_sql_table(table_name, engine, columns=selected_features)
    engine.dispose()

    return df


def aggregate_features(
    messages: Iterable[Message],
    selected_features: list[str],
    feature_aggregation: list[str],
) -> dict[str, dict[str, float | None]]:
    """Aggregate feature statistics from client messages.

    Args:
        messages: Messages from client nodes containing query results
        selected_features: List of feature names to aggregate
        feature_aggregation: List of aggregation methods ('mean', 'std')

    Returns:
        Dictionary with aggregated statistics for each feature and method
    """

    def _initialize_aggregation_stats() -> dict[str, dict[str, dict[str, float]]]:
        """Initialize nested dictionary structure for aggregation statistics."""
        stats: dict[str, dict[str, dict[str, float]]] = {}
        for feature in selected_features:
            stats[feature] = {}
            for agg_method in feature_aggregation:
                if agg_method == "mean":
                    stats[feature]["mean"] = {"sum": 0.0, "count": 0}
                elif agg_method == "std":
                    stats[feature]["std"] = {"sum": 0.0, "count": 0, "sum_sqd": 0.0}
        return stats

    def _accumulate_statistics(
        aggregated_stats: dict, query_results, feature: str, agg_method: str
    ) -> None:
        """Accumulate statistics from a single client's query results."""
        # Handle different RecordDict types from flwr
        if hasattr(query_results, "__getitem__"):
            results = query_results
        else:
            results = dict(query_results)

        if agg_method == "mean":
            aggregated_stats[feature]["mean"]["sum"] += results[f"{feature}_mean_sum"]
            aggregated_stats[feature]["mean"]["count"] += results[
                f"{feature}_mean_count"
            ]
        elif agg_method == "std":
            aggregated_stats[feature]["std"]["sum"] += results[f"{feature}_std_sum"]
            aggregated_stats[feature]["std"]["count"] += results[f"{feature}_std_count"]
            aggregated_stats[feature]["std"]["sum_sqd"] += results[
                f"{feature}_std_sum_sqd"
            ]

    def _compute_final_statistics(
        aggregated_stats: dict, feature: str, agg_method: str
    ) -> float:
        """Compute final aggregated statistic for a feature and aggregation method."""
        if agg_method == "mean":
            stats = aggregated_stats[feature]["mean"]
            if stats["count"] == 0:
                raise ValueError(
                    f"No data available for feature '{feature}' mean calculation"
                )
            return stats["sum"] / stats["count"]

        elif agg_method == "std":
            stats = aggregated_stats[feature]["std"]
            if stats["count"] <= 1:
                raise ValueError(
                    f"Insufficient data for feature '{feature}' std calculation (need > 1 samples)"
                )

            mean = stats["sum"] / stats["count"]
            # Using the formula: Var = (sum_sqd - n * mean^2) / (n - 1)
            variance = (stats["sum_sqd"] - stats["count"] * mean**2) / (
                stats["count"] - 1
            )

            if variance < 0:
                # Handle numerical precision issues
                variance = 0.0

            return np.sqrt(variance)

        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

    # Initialize aggregated statistics
    aggregated_stats = _initialize_aggregation_stats()

    # Aggregate statistics from all valid client messages
    valid_message_count = 0
    for message in messages:
        if message.has_error():
            continue

        query_results = message.content["query_results"]
        valid_message_count += 1

        for feature in selected_features:
            for agg_method in feature_aggregation:
                _accumulate_statistics(
                    aggregated_stats, query_results, feature, agg_method
                )

    if valid_message_count == 0:
        raise ValueError("No valid messages received from clients")

    # Compute final aggregated statistics
    final_stats: dict[str, dict[str, float | None]] = {}
    for feature in selected_features:
        final_stats[feature] = {}
        for agg_method in feature_aggregation:
            try:
                final_stats[feature][agg_method] = _compute_final_statistics(
                    aggregated_stats, feature, agg_method
                )
            except ValueError as e:
                log(INFO, "Warning: %s", e)
                final_stats[feature][agg_method] = None

    return final_stats
