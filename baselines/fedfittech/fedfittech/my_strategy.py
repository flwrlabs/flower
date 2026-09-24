"""Implemented custom Strategy for server app."""

# pylint: disable=invalid-name

import os
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from omegaconf import DictConfig

from fedfittech.flwr_utils.client_utils import get_net_and_config
from fedfittech.flwr_utils.server_plotting_function import (
    weighted_eval_average_plottinng,
)
from fedfittech.task import set_weights
from flwr.common import Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


class CustomFedAvg(FedAvg):
    """Re-implementation of FedAvg."""

    def __init__(
        self,
        cfg: DictConfig,
        model_path: str,
        plot_path: str,
        csv_dir_path: str,
        val_f1_df: pd.DataFrame,
        val_f1_client_vs_labels_table: pd.DataFrame,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        """Initializes the class with configuration settings, file paths, and validation
        data.

        Args:
            cfg (dictConfig): Configuration dictionary.
            model_path (str | Path): Path to the model.
            plot_path (str | Path): Directory path for saving plots.
            csv_dir_path (str | Path): Directory path for saving CSVs.
            val_f1_df (pd.DataFrame): DataFrame containing validation F1 scores.
            val_f1_client_vs_labels_table (pd.DataFrame): DataFrame containing F1
                scores per client and label.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        self.cfg = cfg
        self.model_path = model_path
        self.plot_path = plot_path
        self.csv_dir_path = csv_dir_path
        self.best_eval_f1 = None
        self.val_f1_df = (
            val_f1_df  # Dataframe to store val_f1 scores of all clients per round
        )
        # Dataframe for F1 scores of cliennts vs labels
        self.val_f1_client_vs_labels_table = val_f1_client_vs_labels_table

    def aggregate_fit(self, server_round, results, failures):
        """Save global model.

        Once the global model is aggregated then it will be saved in the model
        directory as global model per round.

        Args:
            server_round (int): The current server round number
        """
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # convert parameterrs to ndarrays
        nndarrays = parameters_to_ndarrays(parameters_aggregated)

        # instantiate model
        model, cfg = get_net_and_config()
        set_weights(model, nndarrays)

        # Save the global model in the standard PyTorch way
        torch.save(
            model.state_dict(),
            os.path.join(self.model_path, f"global_model_round_{server_round}.pth"),
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        metrics_df, self.best_eval_f1, self.val_f1_df = weighted_eval_average_plottinng(
            cfg=self.cfg,
            results=results,
            plt_path=self.plot_path,
            csv_path=self.csv_dir_path,
            server_round=server_round,
            best_eval_f1=self.best_eval_f1,
            val_f1_df=self.val_f1_df,
            f1_labels_df=self.val_f1_client_vs_labels_table,
        )

        pd.set_option("display.max_rows", None)  # Show all rows
        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.max_colwidth", None)

        print(f"Best F1 is {self.best_eval_f1} for server round {server_round}")
        # print(self.val_f1_df.loc[self.val_f1_df["Server_Round"] <= server_round])

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, aggregated_metrics
