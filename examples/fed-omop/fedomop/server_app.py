

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch
from functools import partial

from fedomop.task_utils import (create_instantiate_parameters, 
                                get_train_and_test_modules, 
                                custom_aggregate_metricrecords,
                                load_centralized_data,
                                seed_all,)

from fedomop.log_utils import config_json_file, save_metrics_as_json

from fedomop.result_visualization import plot_metrics

# Create Flower ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    seed_all(context.run_config["seed"])
    
    res_save_path = config_json_file(len(grid.get_node_ids()), 
                                     context.run_config)
    
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    dataset = context.run_config["dataset"]
    model_cls = context.run_config["model"]

    # Load the model and initialize it with the received weights

    global_model = create_instantiate_parameters(dataset, model_cls)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy

    strategy = FedAvg(fraction_evaluate=fraction_evaluate, 
                      min_available_nodes=2,
                      evaluate_metrics_aggr_fn=custom_aggregate_metricrecords)


    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": context.run_config["lr"],
                                   "epochs": context.run_config["local-epochs"],
                                   "weight_decay": context.run_config["weight_decay"]}),
        num_rounds=num_rounds,
        evaluate_fn=partial(global_evaluate, dataset=dataset, model_cls=model_cls),
    )

    save_metrics_as_json(res_save_path, result)
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("\n Plotting results")
    plot_metrics(res_save_path)

def global_evaluate(server_round: int, arrays: ArrayRecord, dataset: str, model_cls: str) -> MetricRecord:
    """Evaluate model on central data."""
        
    # Load the model and initialize it with the received weights
    model = create_instantiate_parameters(dataset, model_cls)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_data(dataset)
    
    _, eval_fn, _, _ =  get_train_and_test_modules(dataset)
    # Evaluate the global model on the test set
    metrics = eval_fn(model, test_dataloader, device)

    #Return the evaluation metrics
    return MetricRecord(metrics)
  