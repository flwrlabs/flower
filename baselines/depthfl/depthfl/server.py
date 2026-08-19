"""Server for DepthFL baseline."""

import copy
from collections import OrderedDict
from logging import DEBUG, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server, fit_clients
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from depthfl.client import prune
from depthfl.models import test, test_sbn
from depthfl.strategy import aggregate_fit_depthfl
from depthfl.strategy_hetero import aggregate_fit_hetero

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]],
    Tuple[float, Dict[str, Union[Scalar, List[float]]]],
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    model : DictConfig
        model configuration for instantiating

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
        Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, Dict[str, Union[Scalar, List[float]]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-100 test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy, accuracy_single = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy, "accuracy_single": accuracy_single}

    return evaluate


def gen_evaluate_fn_hetero(
    trainloaders: List[DataLoader],
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]],
    Tuple[float, Dict[str, Union[Scalar, List[float]]]],
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    trainloaders : List[DataLoader]
        The list of dataloaders to calculate statistics for BN
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    model_cfg : DictConfig
        model configuration for instantiating

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
        Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(  # pylint: disable=too-many-locals
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, Dict[str, Union[Scalar, List[float]]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-100 test set for evaluation."""
        # test per 50 rounds (sbn takes a long time)
        if server_round % 50 != 0:
            return 0.0, {"accuracy": 0.0, "accuracy_single": [0] * 4}

        # models with different width
        models = []
        for i in range(4):
            model_tmp = copy.deepcopy(model_cfg)
            model_tmp.n_blocks = i + 1
            models.append(model_tmp)

        # load global parameters
        param_idx_lst = []
        nets = []
        net_tmp = instantiate(models[-1], track=False)
        for model in models:
            net = instantiate(model, track=True, scale=False)
            nets.append(net)
            param_idx = {}
            for k in net_tmp.state_dict().keys():
                param_idx[k] = [
                    torch.arange(size) for size in net.state_dict()[k].shape
                ]
            param_idx_lst.append(param_idx)

        params_dict = zip(net_tmp.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        for net, param_idx in zip(nets, param_idx_lst):
            net.load_state_dict(prune(state_dict, param_idx), strict=False)
            net.to(device)
            net.train()

        loss, accuracy, accuracy_single = test_sbn(
            nets, trainloaders, testloader, device=device
        )
        # return statistics
        return loss, {"accuracy": accuracy, "accuracy_single": accuracy_single}

    return evaluate


class ServerFedDyn(Server):
    """Sever for FedDyn."""

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        if "HeteroFL" in str(type(self.strategy)):
            aggregate_fit = aggregate_fit_hetero
        else:
            aggregate_fit = aggregate_fit_depthfl

        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = aggregate_fit(
            self.strategy,
            server_round,
            results,
            failures,
            parameters_to_ndarrays(self.parameters),
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
