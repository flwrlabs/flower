import logging
import torch
from collections import OrderedDict
from flwr.common import Context
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from .model import MulticlassClassification, train, validation
from .dataset import prepare_dataset
import os



def get_client_logger(partition_id):
    logger = logging.getLogger(f"Client_{partition_id}")
    if not logger.hasHandlers():
        os.makedirs("logs", exist_ok=True)
        handler = logging.FileHandler(f"logs/client_{partition_id}.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
    
    
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model,cid, trainloader, valloader,num_features, total_classes, num_classes_cl, epoch,  server_round) -> None:
        super().__init__()

        self.id=cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.epoch=epoch
        
        # a model that is randomly initialised at first
        self.model = model
        self.num_classes_cl=num_classes_cl
        self.total_classes=total_classes
        self.server_round=server_round
        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger =  get_client_logger(self.id)
    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
       
        self.set_parameters(parameters)
        train(self.model, self.trainloader,self.valloader, self.epoch)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"coef":self.num_classes_cl/self.total_classes,"cid":self.id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        
        loss, accuracy = validation(self.model, self.valloader, config["current_round"], self.server_round, self.logger)
        self.logger.info(f"Client {self.id} - Round {config['current_round']} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "cid":self.id}

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    max_drop = context.run_config["max-drop"]
    total_classes = context.run_config["total-classes"]    
    local_epochs = context.run_config["local-epochs"]
    num_server_rounds = context.run_config["num-server-rounds"]   
     
    trainloader, valloader, num_classes_client, num_features = prepare_dataset(partition_id, num_partitions,max_drop,total_classes )
    model=MulticlassClassification(num_features,total_classes)
    # Return Client instance
    return FlowerClient(model, partition_id, trainloader, valloader, num_features, total_classes, num_classes_client, local_epochs, num_server_rounds).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
