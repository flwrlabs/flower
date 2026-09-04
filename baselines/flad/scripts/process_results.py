"""Provide F1 score plots.

Assumes CSV files are located at:
`<log_dir>/<experiment_dir>/training_history_<rn_seed>.csv`
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


def _parse_round(value: str) -> int:
    """Parse `Round` column, stripping the '*' best-round marker."""
    return int(value.lstrip("*"))


def _find_experiment_dir(rn_seed: int, log_dir: str) -> str:
    """Find the experiment directory for a given `rn_seed`."""
    matches = glob.glob(os.path.join(log_dir, f"federated_training_flad_{rn_seed}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No experiment directory found for rn_seed={rn_seed} under {log_dir}"
        )
    return matches[0]


def plot_f1_over_rounds(
    rn_seed: int,
    log_dir: str,
    save_path: str,
    client_name: str | None = None,
) -> None:
    """Given the experiment identified by `rn_seed`, plot the F1 score over rounds.

    If `client_name` is given, plots `<client_name>_f1_score` column.
    Otherwise plots `avg_f1_score_best`.
    """
    experiment_dir = _find_experiment_dir(rn_seed, log_dir)
    csv_path = os.path.join(experiment_dir, f"training_history_{rn_seed}.csv")

    with open(csv_path, newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))

    column = f"{client_name}_f1_score" if client_name else "avg_f1_score_best"
    rounds = [_parse_round(row["Round"]) for row in rows]
    values = [float(row[column]) for row in rows]

    plt.figure()
    plt.plot(rounds, values)
    plt.xlabel("Round")
    plt.ylabel("F1 Score")
    if client_name is None:
        plt.title(f"Overall F1 Score over rounds (rn_seed={rn_seed})")
    else:
        plt.title(f"{client_name} F1 Score over rounds (rn_seed={rn_seed})")
    plt.grid(True)

    print(f"Saving plot to {save_path}/{column}_{rn_seed}.png")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{column}_{rn_seed}.png")


def main() -> None:
    """Parse cli args and process results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rn-seed",
        type=int,
        required=True,
        help="rn_seed identifying the experiment to plot",
    )
    parser.add_argument(
        "--client-name",
        default=None,
        help="Client to plot (e.g. '00-WebDDoS'). If omitted, plots avg_f1_score_best",
    )
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Root logs directory"
    )
    parser.add_argument(
        "--save-path", type=str, default="./_static", help="Save the plot to this path"
    )
    args = parser.parse_args()

    plot_f1_over_rounds(
        rn_seed=args.rn_seed,
        client_name=args.client_name,
        log_dir=args.log_dir,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
