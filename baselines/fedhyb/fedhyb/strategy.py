


from typing import List, Tuple
import flwr as fl
import numpy as np
from flwr.common import (
 
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from logging import WARNING
from typing import Callable, Dict,  Optional,  Union
from functools import reduce
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class WeightedStrategy(fl.server.strategy.FedAvg):
    
      

     def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        grad_list=[],
        ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn 
        self.grad_list=[]
         
         
     def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
            # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
                return None, {}
         # print the coefficienent of each client in each round
       # for _, fit_res in results:
       #   
          #print(fit_res.num_examples)
                # Convert results
        weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples,fit_res.metrics.get('coef') )                  
                    for _, fit_res in results
                    
            ]
                
                
                # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples*coef for (_, num_examples,coef) in weights_results)
               
      # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
                 [layer * num_examples*coef for layer in weights] for (weights, num_examples,coef) in weights_results
         ]

  # Compute average weights of each layer
        weights_prime: NDArrays = [
             reduce(np.add, layer_updates) / num_examples_total
             for layer_updates in zip(*weighted_weights)
            ]

        parameters_aggregated = ndarrays_to_parameters(weights_prime)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
