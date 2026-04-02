from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import torch

from fedomop.task_utils import (seed_all, 
                                create_instantiate_parameters, 
                                get_train_and_test_modules, 
                                get_dataloaders)


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    seed_all(context.run_config["seed"])
    
    model_weights, metric_dict = train_fedavg(msg, context)

    # Construct and return reply Message
    model_record = ArrayRecord(model_weights)
    metric_record = MetricRecord(metric_dict)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    #print(f"id {context.node_config["partition-id"]}")
    return Message(content=content, reply_to=msg)



@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    seed_all(context.run_config["seed"])
    
    metrics = eval_fedavg(msg,context)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


def train_fedavg(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the data
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    model_cls = context.run_config["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_fn, _, _, _ = get_train_and_test_modules(dataset)

    trainloader, _ = get_dataloaders(dataset, 
                                     partition_id, 
                                     num_partitions, 
                                     batch_size, 
                                     seed, 
                                     context.run_config["partitioner"], 
                                     context.run_config["dirichlet_alpha"])

    # Load the model and initialize it with the received weights
    model = create_instantiate_parameters(dataset, model_cls)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    
    # Train model
    train_metrics = train_fn(
            model,
            trainloader,
            msg.content["config"]["epochs"],
            msg.content["config"]["lr"],
            msg.content["config"]["weight_decay"],
            device,
        )
    return model.state_dict(), train_metrics 


def eval_fedavg(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cls = context.run_config["model"]
    
    _,test_fn,_, _ = get_train_and_test_modules(dataset)
    
    #Load Model 
    model = create_instantiate_parameters(dataset, model_cls) 
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    _, valloader = get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partitioner"], 
                                    context.run_config["dirichlet_alpha"])

    return test_fn(model, valloader, device)
