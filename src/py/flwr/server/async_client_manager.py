
from typing import List, Dict, Optional
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import threading
from flwr.common.logger import log
from logging import INFO, ERROR
import random
import time

class AsyncClientManager(SimpleClientManager):

    def __init__(self) -> None:
        super().__init__()
        self.free_clients = {}
        self._cv_free = threading.Condition()
    
    def set_client_to_busy(self, client_id: str):
        if client_id not in self.free_clients.keys() or client_id not in self.clients.keys():
            log(ERROR, "Client not found in free_clients")
            return False
        else:
            with self._cv_free:
                self.free_clients.pop(client_id)
                self._cv_free.notify_all()
            return True
    
    def set_client_to_free(self, client_id):
        if client_id not in self.clients.keys():
            log(ERROR, "Client not found in clients")
            return False
        else:
            with self._cv_free:
                self.free_clients[client_id] = self.clients[client_id]
                self._cv_free.notify_all()
            return True

    # waits for `num_free_clients` to be free
    def wait_for_free(self, num_free_clients: int, timeout: int = 86400) -> bool:
        with self._cv_free:
            return self._cv_free.wait_for(
                lambda: len(self.free_clients) >= num_free_clients, timeout=5
            )
    
    def register(self, client: ClientProxy) -> bool:
        log(INFO, "Registering client with id: %s", client.cid)
        if super().register(client):
            self.set_client_to_free(client.cid)
            return True
        else:
            return False
        
    def unregister(self, client: ClientProxy) -> None:
        log(INFO, "Unregistering client with id: %s", client.cid)
        if client.cid in self.free_clients:
            self.set_client_to_busy(client.cid)
        else:
            return super().unregister(client)
        

    def num_free(self) -> int:
        return len(self.free_clients)

    def all_free(self) -> Dict[str, ClientProxy]:
        return list(self.free_clients)

    def sample_free(
        self,
        num_free_clients: int,
        min_num_free_clients: int = 0,
        criterion: Criterion = None,
    ) -> List[ClientProxy]:
        log(INFO, "Sampling %s clients, min %s", num_free_clients, min_num_free_clients)
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are free. # It always samples from all clients.
        if min_num_free_clients is None:
            min_num_free_clients = num_free_clients
        self.wait_for_free(min_num_free_clients)
            
        # Sample clients which meet the criterion
        available_cids = list(self.free_clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.free_clients[cid])
            ]

        if num_free_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available free clients"
                " (%s) is less than number of requested free clients (%s).",
                len(available_cids),
                num_free_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_free_clients)
        ret_list = [self.free_clients[cid] for cid in sampled_cids]
        for cid in sampled_cids:
            self.set_client_to_busy(cid)
        return ret_list

        
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        return self.sample_free(num_clients, min_num_clients, criterion)