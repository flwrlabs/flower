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
"""flad: A Flower Baseline."""

import csv
import os
import time

from flwr.app import MetricRecord


def make_run_output_folder(output_folder: str, rn_seed: int) -> str:
    """Create and return a timestamped output folder."""
    os.makedirs(output_folder, exist_ok=True)
    run_folder = (
        os.path.join(
            output_folder,
            f"federated_training_flad_{rn_seed}-{time.strftime('%Y%m%d-%H%M%S')}",
        )
        + os.sep
    )
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def save_training_history(
    output_folder: str,
    rn_seed: int,
    evaluate_metrics_clientapp: dict[int, MetricRecord],
) -> None:
    """Save per-round evaluation metrics to a CSV file."""
    history_filename = f"training_history_{rn_seed}.csv"
    with open(
        output_folder + "/" + history_filename, "w", newline="", encoding="utf-8"
    ) as history_file:
        round_fieldnames = ["Round"]
        for round_metrics in evaluate_metrics_clientapp.values():
            for key in round_metrics.keys():
                if key not in round_fieldnames:
                    round_fieldnames.append(key)
        writer = csv.DictWriter(history_file, fieldnames=round_fieldnames)
        writer.writeheader()

        for r, round_metrics in evaluate_metrics_clientapp.items():
            avg_f1_value = round_metrics["avg_f1_score"]
            avg_f1_best_value = round_metrics["avg_f1_score_best"]
            assert isinstance(avg_f1_value, int | float)
            assert isinstance(avg_f1_best_value, int | float)
            avg_f1 = float(avg_f1_value)
            avg_f1_best = float(avg_f1_best_value)
            evaluate_metrics = {
                key: (
                    f"{value:.5f}"
                    if isinstance(value, float)
                    else f"{value:d}" if isinstance(value, int) else str(value)
                )
                for key, value in round_metrics.items()
            }
            row = {
                "Round": r if avg_f1 < avg_f1_best else f"*{r}",
                **evaluate_metrics,
            }
            writer.writerow(row)
