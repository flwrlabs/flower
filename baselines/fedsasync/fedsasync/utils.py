"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""
import csv
import torch
import json
import os
from logging import INFO

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr.common import MetricRecord, RecordDict, log
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from sklearn.model_selection import train_test_split

def load_centralized_testloader(
    dataset_name: str = "uoft-cs/cifar10", batch_size: int = 64, max_samples: int = 2000
):
    """Carga el dataset de test oficial/global directamente para el ServerApp."""
    test_ds = load_dataset(dataset_name, split="test")

    # Data stratification for non-iid scenarios: select max_samples while maintaining class proportions
    if max_samples and max_samples < len(test_ds):
        labels = test_ds["label"]
        indices = list(range(len(test_ds)))
        
        # Selecciona max_samples manteniendo la proporción exacta de clases de CIFAR-10
        sub_indices, _ = train_test_split(
            indices, 
            train_size=max_samples, 
            stratify=labels, 
            random_state=42
        )
        test_ds = test_ds.select(sub_indices)

    if dataset_name == "uoft-cs/cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        img_key = "img"
    elif dataset_name == "ylecun/mnist":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.1307,), (0.3081,))]
        )
        img_key = "image"
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")

    def apply_transforms(batch):
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    test_ds = test_ds.with_transform(apply_transforms)

    def collate_fn(batch):
        images = torch.stack([item[img_key] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return images, labels

    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )


def get_evaluate_fn(model, testloader, device):

    def evaluate_fn(server_round: int, arrays):
        if arrays:
            state_dict = {}

            # 1. Si es un ArrayRecord de Flower o un diccionario de parámetros
            if hasattr(arrays, "items") or isinstance(arrays, dict):
                for k, v in arrays.items():
                    # Si el valor es un objeto Array de Flower, extraemos el numpy array
                    if hasattr(v, "numpy"):
                        val = v.numpy()
                    elif hasattr(v, "to_numpy"):
                        val = v.to_numpy()
                    else:
                        val = v

                    if isinstance(val, torch.Tensor):
                        state_dict[k] = val
                    elif not isinstance(val, str):
                        state_dict[k] = torch.from_numpy(val) if hasattr(val, "dtype") else torch.as_tensor(val)

            # 2. Si es una lista o tupla de arrays
            elif isinstance(arrays, (list, tuple)):
                model_keys = list(model.state_dict().keys())
                for k, v in zip(model_keys, arrays):
                    if hasattr(v, "numpy"):
                        val = v.numpy()
                    elif hasattr(v, "to_numpy"):
                        val = v.to_numpy()
                    else:
                        val = v

                    if isinstance(val, str):
                        continue
                    if isinstance(val, torch.Tensor):
                        state_dict[k] = val
                    else:
                        state_dict[k] = torch.from_numpy(val) if hasattr(val, "dtype") else torch.as_tensor(val)

            # Carga de los pesos extraídos
            if state_dict:
                model.load_state_dict(state_dict, strict=False)

        # Configuración para evaluación
        model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_loss = total_loss / total
        eval_acc = correct / total

        return MetricRecord(
            {
                "eval_loss": float(eval_loss),
                "eval_acc": float(eval_acc),
            }
        )

    return evaluate_fn

def train_metrics_aggr_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Personalized train_metrics_aggr_fn to delete client times."""
    train_times: list[float] = []
    for record in records:
        value = record.metric_records["metrics"]["train_time"]
        if isinstance(value, list):
            raise ValueError("train_time should never be a list")

        train_times.append(float(value))
        record.metric_records["metrics"].pop("train_time", None)
    train_time = max(train_times)

    # Call original default function
    aggregated_metrics = aggregate_metricrecords(
        records,
        weighting_metric_name,
    )
    aggregated_metrics["train_time"] = train_time
    aggregated_metrics["qtty_records"] = len(records)
    return aggregated_metrics

def save_logs(
    result,
    strategy_name: str,
    semiasync_deg: int,
    fraction_slow: float,
    dataset_name: str,
    total_rounds: int,
    run_id: int,
    client_staleness: dict,
    metrics: dict,
    data_distribution: str,
) -> None:
    """Save the federated result in a csv and summary statistics in a json."""
    dataset_map = {
        "uoft-cs/cifar10": "cifar10",
        "ylecun/mnist": "mnist",
    }
    dataset = dataset_map.get(dataset_name, dataset_name)

    if strategy_name == "FedAvg":
        config_dir = f"{strategy_name}_fs{fraction_slow}"
    else:
        config_dir = f"{strategy_name}_fs{fraction_slow}_m{semiasync_deg}"

    base_dir = f"_static/{dataset}_{data_distribution}/{config_dir}"
    os.makedirs(base_dir, exist_ok=True)

    csv_path = f"{base_dir}/ex{run_id}.csv"
    summary_path = f"{base_dir}/summary_ex{run_id}.json"

    # --- 1. Export round-level metrics to CSV ---
    rounds = sorted(result.evaluate_metrics_clientapp.keys())
    
    header = None
    for rnd in rounds:
        ev_cl = result.evaluate_metrics_clientapp.get(rnd, {}) or {}
        ev_sr = result.evaluate_metrics_serverapp.get(rnd, {}) or {}
        tr = result.train_metrics_clientapp.get(rnd, {}) or {}
        if ev_cl or ev_sr or tr:
            header = list(dict.fromkeys(list(ev_cl.keys()) + list(ev_sr.keys()) + list(tr.keys())))
            break

    if header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for rnd in rounds:
                eval_metrics_cl = result.evaluate_metrics_clientapp.get(rnd, {}) or {}
                eval_metrics_sr = result.evaluate_metrics_serverapp.get(rnd, {}) or {}
                train_metrics = result.train_metrics_clientapp.get(rnd, {}) or {}

                if not eval_metrics_cl and not eval_metrics_sr and not train_metrics:
                    continue

                row = []
                for key in header:
                    if eval_metrics_cl and key in eval_metrics_cl:
                        row.append(eval_metrics_cl[key])
                    elif eval_metrics_sr and key in eval_metrics_sr:
                        row.append(eval_metrics_sr[key])
                    elif train_metrics and key in train_metrics:
                        row.append(train_metrics[key])
                    else:
                        row.append("")

                writer.writerow(row)
        log(INFO, "CSV saved: %s", csv_path)
    else:
        log(INFO, "No valid metrics found to save CSV for %s", csv_path)

    # --- 2. Compute and export global summary statistics to JSON ---
    formatted_staleness_lists = {str(k): v for k, v in client_staleness.items()}

    total_staleness_per_client = {
        k: sum(values) for k, values in formatted_staleness_lists.items()
    }

    formatted_participation = {
        k: total_rounds - total_staleness
        for k, total_staleness in total_staleness_per_client.items()
    }

    aggregated_staleness = (
        [sum(round_values) for round_values in zip(*client_staleness.values())]
        if client_staleness
        else []
    )

    summary_data = {
        "client_participation": formatted_participation,
        "round_staleness": aggregated_staleness,
        "client_staleness": total_staleness_per_client,
        "metrics": metrics,
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    
    log(INFO, "Summary JSON saved: %s", summary_path)
