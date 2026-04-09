"""..."""

from flwr.common import FitIns
from flwr.server.strategy import FedAvg, FedProx


class FedAvgCustom(FedAvg):
    """..."""

    def __init__(
        self,
        *,
        fraction_fit,
        fraction_evaluate,
        min_fit_clients,
        min_evaluate_clients,
        min_available_clients,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        mashed_data=None,  # for consistency, to avoid change in main function
        lr_decay_after_each_round,
    ):
        """..."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.decay = lr_decay_after_each_round

    def configure_fit(self, server_round, parameters, client_manager):
        """..."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "lr_decay_accumulated": self.decay ** (server_round - 1)
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]


class FedProxCustom(FedProx):
    """..."""

    def __init__(
        self,
        *,
        fraction_fit,
        fraction_evaluate,
        min_fit_clients,
        min_evaluate_clients,
        min_available_clients,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        mashed_data=None,  # for consistency, to avoid change in main function
        lr_decay_after_each_round,
        proximal_mu,
    ):
        """..."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            proximal_mu=proximal_mu,
        )

        self.decay = lr_decay_after_each_round

    def configure_fit(self, server_round, parameters, client_manager):
        """..."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "lr_decay_accumulated": self.decay ** (server_round - 1)
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]


class FedMix(FedAvg):
    """..."""

    def __init__(
        self,
        *,
        fraction_fit,
        fraction_evaluate,
        min_fit_clients,
        min_evaluate_clients,
        min_available_clients,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        mashed_data,
        mixup_ratio,
        lr_decay_after_each_round,
    ):
        """..."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # a better way (for production) may be to store mashed data as files whenever
        # local data is updated. here this works since we do not update client datasets
        self.mashed_data = mashed_data
        self.mixup_ratio = mixup_ratio
        self.decay = lr_decay_after_each_round

    def configure_fit(self, server_round, parameters, client_manager):
        """..."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "mashed_data": self.mashed_data,
                        "mixup_ratio": self.mixup_ratio,
                        "lr_decay_accumulated": self.decay ** (server_round - 1)
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

 
class FedProxMix(FedProx):
    """..."""

    def __init__(
        self,
        *,
        fraction_fit,
        fraction_evaluate,
        min_fit_clients,
        min_evaluate_clients,
        min_available_clients,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        mashed_data,
        mixup_ratio,
        lr_decay_after_each_round,
        proximal_mu,
    ):
        """..."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            proximal_mu=proximal_mu,
        )

        # a better way (for production) may be to store mashed data as files whenever
        # local data is updated. here this works since we do not update client datasets
        self.mashed_data = mashed_data
        self.mixup_ratio = mixup_ratio
        self.decay = lr_decay_after_each_round

    def configure_fit(self, server_round, parameters, client_manager):
        """..."""
        """..."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "mashed_data": self.mashed_data,
                        "mixup_ratio": self.mixup_ratio,
                        "lr_decay_accumulated": self.decay ** (server_round - 1)
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]