import math
from flwr.common import Parameters
import flwr.server.strategy.aggregate as agg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple
from flwr.common import NDArrays, log
from logging import DEBUG, WARNING


class AsynchronousStrategy:
    """Abstract base class for all asynchronous strategies."""

    def __init__(self, total_samples: int, staleness_alpha: float, fedasync_mixing_alpha: float, fedasync_a: float, num_clients: int, async_aggregation_strategy: str,
                 use_staleness: bool, use_sample_weighing: bool, send_gradients: bool) -> None:
        self.total_samples = total_samples
        self.staleness_alpha = staleness_alpha
        self.fedasync_a = fedasync_a
        self.fedasync_mixing_alpha = fedasync_mixing_alpha
        self.num_clients = num_clients
        self.async_aggregation_strategy = async_aggregation_strategy
        self.use_staleness = use_staleness
        self.use_sample_weighing = use_sample_weighing
        self.send_gradients = send_gradients

    def average(self, global_parameters: Parameters, model_update_parameters: Parameters, t_diff: float, num_samples: int) -> Parameters:
        """Compute the average of the global and client parameters."""
        if self.async_aggregation_strategy == "fedasync":
            if self.send_gradients:
                return self.weighted_merge_fedasync(global_parameters, model_update_parameters, t_diff, num_samples)
            else:
                return self.weighted_average_fedasync(global_parameters, model_update_parameters, t_diff, num_samples)
        # elif self.async_aggregation_strategy == "asyncfeded":
        #     return self.weighted_average_asyncfeded(global_parameters, model_update_parameters, t_diff, num_samples)
        # elif self.async_aggregation_strategy == "unweighted":
        #     return self.unweighted_average(global_parameters, model_update_parameters)
        else:
            raise ValueError(
                f"Invalid async aggregation strategy: {self.async_aggregation_strategy}")

    def unweighted_average(self, global_parameters: Parameters, model_update_parameters: Parameters) -> Parameters:
        """Compute the unweighted average of the global and client parameters."""
        return ndarrays_to_parameters(agg.aggregate([(parameters_to_ndarrays(global_parameters), 1),
                                                     (parameters_to_ndarrays(model_update_parameters), 1)]))

    def weighted_average_asyncfeded(self, global_parameters: Parameters, model_update_parameters: Parameters, t_diff: float, num_samples: int) -> Parameters:
        """Compute the weighted average of the global and client parameters. Inspired by the paper asyncFedED : https://arxiv.org/pdf/2205.13797.pdf"""
        return ndarrays_to_parameters(self.aggregate_asyncfeded(parameters_to_ndarrays(global_parameters), parameters_to_ndarrays(model_update_parameters), t_diff, num_samples=num_samples))

    def get_sample_weight_coeff(self, num_samples: int) -> float:
        """Compute the sample weight coefficient."""
        return num_samples / self.total_samples

    def weighted_average_fedasync(self, global_parameters: Parameters, model_update_parameters: Parameters, t_diff: float, num_samples: int) -> Parameters:
        """Compute the weighted average of the global and client parameters. Inspired by the paper Fedasync : https://arxiv.org/pdf/1903.03934.pdf"""
        return ndarrays_to_parameters(self.aggregate_fedasync(parameters_to_ndarrays(global_parameters), parameters_to_ndarrays(model_update_parameters), t_diff, num_samples=num_samples))

    def aggregate_fedasync(self, global_param_arr: NDArrays, model_update_param_arr: NDArrays, t_diff: float, num_samples: int) -> NDArrays:
        """Compute weighted average with the formula params_new = (1-alpha) * params_old + alpha * (model_update_params)"""
        # Calculate the total number of examples used during training
        alpha_coeff = self.fedasync_mixing_alpha
        if self.use_staleness:
            alpha_coeff *= self.get_staleness_weight_coeff_fedasync_poly(
                t_diff=t_diff)
        if self.use_sample_weighing:
            alpha_coeff *= self.get_sample_weight_coeff(num_samples)

        # log(DEBUG, f"t_diff: {t_diff}\nalpha_coeff: {alpha_coeff}")

        return [(1 - alpha_coeff) * layer_global + alpha_coeff * layer_update for layer_global, layer_update in zip(global_param_arr, model_update_param_arr)]

    def weighted_merge_fedasync(self, global_parameters: Parameters, gradients: Parameters, t_diff: float, num_samples: int) -> Parameters:
        """Add gradients to the global model. Inspired by the paper Fedasync : https://arxiv.org/pdf/1903.03934.pdf
        It is not however the same procedure as in original paper, because they aggregate MODELS and we aggregate GRADIENTS.
        """
        return ndarrays_to_parameters(self.add_grads_fedasync(parameters_to_ndarrays(global_parameters), parameters_to_ndarrays(gradients), t_diff, num_samples=num_samples))

    def add_grads_fedasync(self, global_param_arr: NDArrays, gradients_arr: NDArrays, t_diff: float, num_samples: int) -> NDArrays:
        """Compute weighted average with the formula params_new = (1-alpha) * params_old + alpha * (params_old + update_grads)"""
        # Calculate the total number of examples used during training
        alpha_coeff = self.fedasync_mixing_alpha
        if self.use_staleness:
            alpha_coeff *= self.get_staleness_weight_coeff_fedasync_poly(
                t_diff=t_diff)
        if self.use_sample_weighing:
            alpha_coeff *= self.get_sample_weight_coeff(num_samples)

        # log(DEBUG, f"t_diff: {t_diff}\nalpha_coeff: {alpha_coeff}")

        return [(1 - alpha_coeff) * layer_global + alpha_coeff * (layer_global + layer_grad) for layer_global, layer_grad in zip(global_param_arr, gradients_arr)] 

    # See paper: https://arxiv.org/pdf/2205.13797.pdf
    def aggregate_asyncfeded(self, global_param_arr: NDArrays, model_update_param_arr: NDArrays, t_diff: float, num_samples: int) -> NDArrays:
        """Computing the new parameters using the formula params_new = params_old + nu * (model_update_params)
            Where nu is influenced by the staleness of the model update and/or the number of samples.
        """
        eta = 1
        if self.use_staleness:
            # Staleness weighted coefficient
            eta *= self.get_staleness_weight_coeff_paflm(t_diff=t_diff)
        if self.use_sample_weighing:
            eta *= self.get_sample_weight_coeff(num_samples)

        log(DEBUG, f"t_diff: {t_diff}\nnu: {eta}")
        return [layer_global + eta * (layer_update - layer_global) for layer_global, layer_update in zip(global_param_arr, model_update_param_arr)]

    # See paper for more details : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9022982
    def get_staleness_weight_coeff_paflm(self, t_diff: float) -> float:
        mu_staleness = t_diff
        exponent = ((1 / float(self.num_clients)) * mu_staleness) - 1
        beta_P = math.pow(self.staleness_alpha, exponent)
        return beta_P

    # Paper: https://arxiv.org/pdf/1903.03934.pdf
    def get_staleness_weight_coeff_fedasync_constant(self) -> float:
        return 1.0

    def get_staleness_weight_coeff_fedasync_poly(self, t_diff: float) -> float:
        return math.pow(t_diff + 1, -self.fedasync_a)

    def get_staleness_weight_coeff_fedasync_hinge(self, t_diff: float, a: float = 10, b: float = 4) -> float:
        return 1 if t_diff <= b else 1 / (a * (t_diff - b) + 1)
