# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Strategy results."""


import pprint
from dataclasses import dataclass, field

from flwr.app.typing import MetricRecordValues
from flwr.common import ArrayRecord, MetricRecord


@dataclass
class Result:
    """Data class carrying records generated during the execution of a strategy.

    This class encapsulates the results of a federated learning strategy execution,
    including the final global model parameters and metrics collected throughout
    the federated training and evaluation (both federated and centralized) stages.

    Attributes
    ----------
    arrays : ArrayRecord
        The final global model parameters. Contains the
        aggregated model weights/parameters that resulted from the federated
        learning process.
    train_metrics_clientapp : dict[int, MetricRecord]
        Training metrics collected from ClientApps, indexed by round number.
        Contains aggregated metrics (e.g., loss, accuracy) from the training
        phase of each federated learning round.
    evaluate_metrics_clientapp : dict[int, MetricRecord]
        Evaluation metrics collected from ClientApps, indexed by round number.
        Contains aggregated metrics  (e.g. validation loss) from the evaluation
        phase where ClientApps evaluate the global model on their local
        validation/test data.
    evaluate_metrics_serverapp : dict[int, MetricRecord]
        Evaluation metrics generated at the ServerApp, indexed by round number.
        Contains metrics from centralized evaluation performed by the ServerApp
        (e.g., when the server evaluates the global model on a held-out dataset).
    """

    arrays: ArrayRecord = field(default_factory=ArrayRecord)
    train_metrics_clientapp: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics_clientapp: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics_serverapp: dict[int, MetricRecord] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Create a representation of the Result instance."""
        rep = ""
        arr_size = sum(len(array.data) for array in self.arrays.values()) / (1024**2)
        rep += "Global Arrays:\n" + f"\tArrayRecord ({arr_size:.3f} MB)\n" + "\n"
        rep += (
            "Aggregated ClientApp-side Train Metrics:\n"
            + pprint.pformat(stringify_dict(self.train_metrics_clientapp), indent=2)
            + "\n\n"
        )

        rep += (
            "Aggregated ClientApp-side Evaluate Metrics:\n"
            + pprint.pformat(stringify_dict(self.evaluate_metrics_clientapp), indent=2)
            + "\n\n"
        )

        rep += (
            "ServerApp-side Evaluate Metrics:\n"
            + pprint.pformat(stringify_dict(self.evaluate_metrics_serverapp), indent=2)
            + "\n"
        )

        return rep


def format_value(val: MetricRecordValues) -> str:
    """Format a value as string, applying scientific notation for floats."""
    if isinstance(val, float):
        return f"{val:.4e}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, list):
        return str([f"{x:.4e}" if isinstance(x, float) else str(x) for x in val])
    return str(val)


def stringify_dict(d: dict[int, MetricRecord]) -> dict[int, dict[str, str]]:
    """Return a copy results metrics but with values converted to string and formatted
    accordingtly."""
    new_metrics_dict = {}
    for k, inner in d.items():
        new_inner = {}
        for ik, iv in inner.items():
            new_inner[ik] = format_value(iv)
        new_metrics_dict[k] = new_inner
    return new_metrics_dict
