"""Utility helpers for persisting TraceFL provenance results.

This module mirrors the bookkeeping performed in the original TraceFL project
so that experiments executed through the Flower baseline can be analysed with
the same plotting scripts.  Every provenance run is appended to a structured
CSV file inside ``results_csvs/``; these CSVs are then consumed by
``tracefl.plotting`` and the shell utilities under ``scripts/`` to
generate paper-style figures and tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _sanitize_component(component: str) -> str:
    """Return a filesystem-safe representation for a filename component.

    ``pathvalidate`` is used in the upstream project, but we avoid the extra
    dependency here by performing a conservative replacement which retains
    alphanumeric characters, ``-`` and ``_``.
    """
    allowed = {"-", "_"}
    sanitized_chars: list[str] = []
    for ch in component:
        if ch.isalnum() or ch in allowed:
            sanitized_chars.append(ch)
        else:
            sanitized_chars.append("-")
    sanitized = "".join(sanitized_chars)
    # collapse duplicated separators introduced by replacement
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "value"


def _format_label_map(label_map: dict[int, int]) -> str:
    if not label_map:
        return "none"
    pairs = [f"{int(k)}to{int(v)}" for k, v in sorted(label_map.items())]
    return "_".join(pairs)


def _format_faulty_clients(clients: Iterable[int]) -> str:
    clients = sorted(int(c) for c in clients)
    if not clients:
        return "none"
    return "_".join(str(c) for c in clients)


def _build_experiment_key(cfg: Any) -> str:
    """Construct a descriptive experiment key based on the configuration."""
    dist = getattr(cfg, "data_dist", None)
    if dist is None:
        return "tracefl_experiment"

    prov_rounds = getattr(cfg, "provenance_rounds", [])
    rounds_str = "-".join(str(r) for r in sorted(prov_rounds))
    parts = [
        f"dataset-{getattr(dist, 'dname', 'unknown')}",
        f"model-{getattr(dist, 'model_name', 'unknown')}",
        f"clients-{getattr(dist, 'num_clients', 'n')}",
        f"alpha-{getattr(dist, 'dirichlet_alpha', 'a')}",
        f"rounds-{rounds_str}",
    ]

    noise = getattr(cfg, "noise_multiplier", None)
    if noise is not None and noise > 0:
        parts.append(f"noise-{noise}")

    clip = getattr(cfg, "clipping_norm", None)
    if clip is not None and clip > 0:
        parts.append(f"clip-{clip}")

    faulty = getattr(cfg, "faulty_clients_ids", [])
    if faulty:
        parts.append(f"faulty-{_format_faulty_clients(faulty)}")

    flip = getattr(cfg, "label2flip", {})
    if flip:
        parts.append(f"flip-{_format_label_map(flip)}")

    sanitized_parts = [_sanitize_component(str(p)) for p in parts if p]
    return "_".join(filter(None, sanitized_parts)) or "tracefl_experiment"


@dataclass
class ExperimentResultLogger:
    """Persist per-round provenance results to CSV files."""

    cfg: Any
    results_dir: Path = Path("results_csvs")
    output_dir: Path | None = None
    _records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize the results logger after object creation."""
        # Use output_dir if provided, otherwise fall back to results_dir
        target_dir = self.output_dir if self.output_dir else self.results_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = target_dir / f"prov_{_build_experiment_key(self.cfg)}.csv"
        if self._file_path.exists():
            try:
                existing = pd.read_csv(self._file_path)
                self._records = existing.to_dict("records")
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                FileNotFoundError,
                PermissionError,
            ):
                # If the file is corrupted we start from scratch but keep a
                # backup for manual inspection.
                backup = self._file_path.with_suffix(".backup")
                self._file_path.replace(backup)
                self._records = []

    @property
    def file_path(self) -> Path:
        """Get the file path for the results CSV."""
        return self._file_path

    def record_round(self, round_id: int, results: dict[str, Any]) -> None:
        """Record results for a specific round."""
        eval_metrics = results.get("eval_metrics", {})
        accuracy = eval_metrics.get("Accuracy")
        contribution = results.get("client_contributions", {})

        record: dict[str, Any] = {
            "round": int(round_id),
            "samples_analyzed": int(results.get("samples_analyzed", 0)),
            "Accuracy": float(accuracy) if accuracy is not None else None,
            "test_data_acc": results.get("test_data_acc"),
            "test_data_loss": results.get("test_data_loss"),
            "avg_prov_time_per_input": results.get("avg_prov_time_per_input"),
            "top_contributor": results.get("top_contributor"),
        }

        # Preserve deterministic ordering of client contribution columns.
        for client_id in sorted(contribution):
            record[f"client_{client_id}_contribution"] = contribution[client_id]

        # Deduplicate the record for the same round to support re-runs.
        self._records = [
            r for r in self._records if int(r.get("round", -1)) != round_id
        ]
        self._records.append(record)
        self._records.sort(key=lambda r: int(r.get("round", 0)))

        df = pd.DataFrame(self._records)
        df.to_csv(self._file_path, index=False)
