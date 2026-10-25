#
# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Produce a summary table of FLAD experiments.

Assumes CSV files are located at:
`<log_dir>/<experiment_dir>/training_history_<rn_seed>.csv`
"""

import argparse
import csv
import glob
import os
import re
import statistics

# Directory names embed a `%Y%m%d-%H%M%S` timestamp, so sorting them
# lexicographically is equivalent to sorting chronologically (see
# process_results.py, which uses the same convention).
_RUN_DIR_PATTERN = re.compile(r"^federated_training_flad_(?P<seed>\d+)-\d{8}-\d{6}$")


def _parse_round(value: str) -> int:
    """Parse `Round` column, stripping the '*' best-round marker."""
    return int(value.lstrip("*"))


def _has_valid_history(run_dir: str) -> bool:
    """Return True if `run_dir` contains a non-empty training_history_*.csv."""
    for csv_path in glob.glob(os.path.join(run_dir, "training_history_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as history_file:
            if list(csv.DictReader(history_file)):
                return True
    return False


def _latest_valid_run_dir(log_dir: str, seed: str, entries: list[str]) -> str:
    """Return the most recent run directory for `seed` with a valid history.

    `entries` must be sorted ascending (oldest first). A run whose directory
    was created but that crashed before writing a training history (e.g.
    server_app.py creates the timestamped directory before starting the
    strategy) is skipped in favor of an older, complete run, rather than
    silently dropping the seed from the summary.
    """
    for entry in reversed(entries):
        run_dir = os.path.join(log_dir, entry)
        if _has_valid_history(run_dir):
            if entry != entries[-1]:
                print(
                    f"Newest run for rn_seed={seed} ({entries[-1]}) has no valid "
                    f"training history; falling back to {entry}."
                )
            elif len(entries) > 1:
                print(
                    f"Found {len(entries)} runs for rn_seed={seed}; "
                    f"using the most recent one: {entry}"
                )
            return run_dir
    raise ValueError(
        f"No run with a valid training history found for rn_seed={seed} "
        f"under {log_dir} (checked: {', '.join(entries)})."
    )


def _latest_run_dirs(log_dir: str) -> list[str]:
    """Return one run directory per rn_seed: the most recent one with a valid history.

    If multiple runs exist for the same seed (e.g. a rerun), only one run is
    kept per seed.
    """
    entries_by_seed: dict[str, list[str]] = {}
    for entry in sorted(os.listdir(log_dir)):
        match = _RUN_DIR_PATTERN.match(entry)
        if not match:
            continue
        entries_by_seed.setdefault(match.group("seed"), []).append(entry)

    return [
        _latest_valid_run_dir(log_dir, seed, entries)
        for seed, entries in entries_by_seed.items()
    ]


def summarize_experiments(log_dir: str) -> dict[str, float]:
    """Average the round number and `avg_f1_score_best` across all experiments.

    Also computes the standard deviation of `avg_f1_score_best` values.
    """
    final_rounds = []
    final_f1_bests = []

    for run_dir in _latest_run_dirs(log_dir):
        for csv_path in glob.glob(os.path.join(run_dir, "training_history_*.csv")):
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
