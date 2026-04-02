import json
import os
from pathlib import Path
import time

from flwr.serverapp.strategy import Result

RESULTS_FILE_TEMPLATE = "result-{}.json"

def config_json_file(n_nodes: int, run_config: dict):
    """Initialize the json file and write the run configurations."""
    # Initialize the execution results directory.
    res_save_path = Path("results") / run_config['dataset'] / f"{n_nodes}_clients" / f"{run_config['num-server-rounds']}_rounds" #Path(f"./results/{run_config['dataset']}/{str(n_nodes)}_clients/{run_config['num-server-rounds']}_rounds")
    res_save_path.mkdir(parents=True, exist_ok=True)

    # Set the date and full path of the file to save the results.
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = RESULTS_FILE_TEMPLATE.format(timestamp)
    result_file = res_save_path /filename

    data = {
        "number_of_nodes": n_nodes,
        "run_config": run_config,
        "round_res": [],
    }
    with open(result_file, "w", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)
    return Path(result_file)

def save_metrics_as_json(save_path: str, result: Result) -> None:
    """Append per-round metrics into payload['round_res'] in save_path (JSON file)."""

    # save_path is the JSON file path
    with open(save_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)  # dict with keys: run_config, round_res

    # Ensure round_res exists and is a list
    round_res = payload.setdefault("round_res", [])
    round_ids = sorted(result.evaluate_metrics_clientapp.keys())
    
    for r in round_ids:
        round_res.append({
            "round": r,
            "evaluate_metrics_clientapp": dict(result.evaluate_metrics_clientapp[r]),
        })

    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)