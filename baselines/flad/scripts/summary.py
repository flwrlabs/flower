"""Produce a summary table by FLAD experiments.

Assumes CSV files are located at:
`<log_dir>/<experiment_dir>/training_history_<rn_seed>.csv`
"""

import argparse
import csv
import glob
import os
import statistics


def _parse_round(value: str) -> int:
    """Parse `Round` column, stripping the '*' best-round marker."""
    return int(value.lstrip("*"))


def summarize_experiments(log_dir: str) -> dict[str, float]:
    """Average the round number and `avg_f1_score_best` across all experiments.

    Also computes the standard deviation of `avg_f1_score_best` values.
    """
    final_rounds = []
    final_f1_bests = []

    for csv_path in glob.glob(os.path.join(log_dir, "*", "training_history_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as history_file:
            rows = list(csv.DictReader(history_file))
        if not rows:
            continue
        last_row = rows[-1]
        final_rounds.append(_parse_round(last_row["Round"]))
        final_f1_bests.append(float(last_row["avg_f1_score_best"]))

    if not final_rounds:
        raise ValueError(f"No training_history CSV files found under {log_dir}")

    return {
        "avg_round_number": statistics.mean(final_rounds),
        "avg_f1_score_best": statistics.mean(final_f1_bests),
        "stddev_f1_score_best": (
            statistics.stdev(final_f1_bests) if len(final_f1_bests) > 1 else 0.0
        ),
    }


def print_summary_table(summary: dict[str, float]) -> None:
    """Print a summary table given summarize_experiments() results."""
    print("\nExperiment Summary:")
    print(f"Average round number: {summary['avg_round_number']:.2f}")
    print(f"Average f1 score best: {summary['avg_f1_score_best']:.4f}")
    print(f"Stddev of avg_f1_score_best: {summary['stddev_f1_score_best']:.4f}\n")


def main() -> None:
    """Parse args and process results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Root logs directory"
    )

    args = parser.parse_args()

    summary = summarize_experiments(args.log_dir)
    print_summary_table(summary)


if __name__ == "__main__":
    main()
