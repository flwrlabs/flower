"""Handles clients that are sampled every round.

In a FedMeta experiment, there is a train and a test client. So we modified the manager
to sample from each list each round.
"""

import random
from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class FedmetaClientManager(SimpleClientManager):
    """In the fit phase, clients must be sampled from the training client list.

    And in the evaluate stage, clients must be sampled from the validation client list.
    So we modify 'fedmeta_client_manager' to sample clients from [cid: List] for each
    list.
    """

    def __init__(self, valid_client, **kwargs):
        super().__init__(**kwargs)
        self.valid_client = valid_client

    # pylint: disable=too-many-arguments
    def sample(  # pylint: disable=arguments-differ
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        server_round: Optional[int] = None,
        step: Optional[str] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        if step == "evaluate":
            available_cids = [str(result) for result in range(0, self.valid_client)]
        else:
            available_cids = list(self.clients)

        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        if server_round is not None:
            random.seed(server_round)
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
