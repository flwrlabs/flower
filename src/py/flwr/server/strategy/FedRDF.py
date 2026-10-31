# Copyright 2021 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FedRDF: A Robust and Dynamic Aggregation Function against Poioning Attacks in Federated Learning. Enrique Marmol Campos, Aurora Gonzalez-Vidal José L. Hernández-Ramos and Antonio Skarmeta

Paper: arxiv.org/abs/2402.10082
"""


import numpy as np
from logging import WARNING
from flwr.server.strategy import Strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar, MetricsAggregationFn,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArray, NDArrays
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from scipy.stats import ks_2samp
from flwr.server.strategy import FedAvg



class FedRDF(FedAvg):
    """Federated Optim strategy.

        Implementation based on https://arxiv.org/abs/2003.00295v5

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        threshold: float
            Threshold that indicate when apply FedAvg or the Fast Fourier Transform (FFT)
            depending on the skewness of the weights. A value of 0 means applying always
            the FFT. Defaults to 0.
        """
    def __init__(self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        threshold: float = 0.0,
) -> None:
        super().__init__(
            fraction_fit = fraction_fit,
            fraction_evaluate = fraction_evaluate,
            min_fit_clients = min_fit_clients,
            min_evaluate_clients = min_evaluate_clients,
            min_available_clients = min_available_clients,
            evaluate_fn = evaluate_fn,
            on_fit_config_fn = on_fit_config_fn,
            on_evaluate_config_fn = on_evaluate_config_fn,
            accept_failures = accept_failures,
            initial_parameters = initial_parameters,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        )
        self.threshold = threshold
    def __repr__(self) -> str:
        rep = f"FedRDF(accept_failures={self.accept_failures})"
        return rep


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self.aggregate_FedRDF(weights_results,self.threshold))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def ks_proportion(self, sample):
        """Applies the Kolmogórov-Smirnov test"""
        total = []
        for i in range(1000):
            n_random = int(len(sample)*0.3)
            random_indices = np.random.choice(len(sample), size=n_random, replace=False)
            random_points = sample[random_indices]
            generated_points = np.delete(sample, random_indices)

            ks_stat, p_value = ks_2samp(generated_points, random_points)
            # Set a significance level
            significance_level = 0.05

            # Identify the random points
            if p_value < significance_level:
                total.append(1)
            else:
                total.append(0)
            rtn = sum(total) / len(total)
            return rtn

    def skeweness(self, arrays):
        """Measure the skeweness proportion in the layes"""
        means_array = []
        iteraciones_array = [np.nditer(a) for a in arrays]
        np_ite = np.array(iteraciones_array)
        idx = arrays[0].size
        for i in range(idx):
            values = [np_ite[j][i] for j in range(len(iteraciones_array))]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            values = (values - np.mean(values)) / np.std(values)
            prop = self.ks_proportion(values)
            means_array.append(prop)

        return np.mean(means_array)

    def fourier_abs(self, arrays):
        fourier_array = []
        iteraciones_array = [np.nditer(a) for a in arrays]
        np_ite = np.array(iteraciones_array)
        idx = arrays[0].size
        for i in range(idx):
            values = [np_ite[j][i] for j in range(len(iteraciones_array))]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            # Frequencies
            freq = np.fft.fftfreq(values.shape[-1])

            # Index of highest frequency
            index = np.argmax(np.abs(freq))

            # Value of highest frequency in original data
            result = values[index]

            fourier_array.append(result)
        return np.array(fourier_array).reshape(arrays[0].shape)

    def aggregate_FedRDF(self, results: List[Tuple[NDArrays, int]], threshold) -> NDArrays:
        """Compute FedRDF, depending on the threshold, it takes the mean or the FFT."""
        num_examples_total = sum([num_examples for _, num_examples in results])
        global_layers = [weights for weights, _ in results]

        sks = [np.mean(self.skeweness(layers)) for layers in zip(*global_layers)]
        prop = np.mean(sks)

        if prop < threshold:
            weighted_weights = [
                [layer * num_examples for layer in weights] for weights, num_examples in results
            ]
            # Compute average weights of each layer
            weights_prime: NDArrays = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in
                                       zip(*weighted_weights)]
        else:
            weights_prime: NDArrays = [self.fourier_abs(layers) for layers in zip(*global_layers)]

        return weights_prime


