"""Plotting helpers for TraceFL baseline experiments.

The upstream TraceFL project ships a comprehensive plotting pipeline which is
driven by cached experiment artefacts.  The baseline implementation now exports
per-round provenance results to CSV files; this module consumes those CSVs and
recreates the most relevant figures (accuracy curves and summary tables) using
``matplotlib``/``seaborn``.  The functions are intentionally lightweight so that
they can be invoked both from Python and the convenience shell scripts located
under ``tracefl-baseline/scripts``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class PlotArtefacts:
    """Container describing where generated plots and tables are stored."""

    base_dir: Path

    @property
    def png_dir(self) -> Path:
        """Get the PNG directory path."""
        path = self.base_dir / "png"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def pdf_dir(self) -> Path:
        """Get the PDF directory path."""
        path = self.base_dir / "pdf"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tables_dir(self) -> Path:
        """Get the tables directory path."""
        path = self.base_dir / "tables"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _scale_to_percentage(series: pd.Series) -> pd.Series:
    """Return a percentage representation while guarding against NaNs."""
    if series is None:
        return pd.Series(dtype=float)

    series = series.astype(float)
    if series.empty:
        return series

    max_value = series.max()
    if pd.isna(max_value):
        return series
    if max_value <= 1.0:
        return series * 100.0
    return series


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def _derive_run_label(path: Path) -> str:
    """Create a clean, readable run label from the CSV filename."""
    name = path.stem
    if name.startswith("prov_"):
        name = name[5:]

    # Create cleaner labels for common experiment patterns
    if "experiment_a" in name:
        return "Localization Accuracy (α=0.3)"
    if "experiment_b" in name:
        return "Data Distribution Analysis"
    if "experiment_c" in name:
        return "Faulty Client Detection"
    if "experiment_d" in name:
        return "Differential Privacy"
    # For other cases, create a shorter label
    parts = name.split("_")
    if len(parts) > 3:
        # Take first few meaningful parts
        return " ".join(parts[:3]).replace("-", "=")
    return name.replace("_", " ")


def load_results(paths: Iterable[Path]) -> list[pd.DataFrame]:
    """Load results from CSV files."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if "round" not in df:
            df["round"] = range(1, len(df) + 1)
        df["TraceFL Accuracy (%)"] = _scale_to_percentage(df.get("Accuracy"))
        if "test_data_acc" in df:
            df["FL Accuracy (%)"] = _scale_to_percentage(df["test_data_acc"])
        df["Run"] = _derive_run_label(path)
        frames.append(df)
    return frames


def plot_accuracy_curves(
    csv_paths: Sequence[Path],
    output_dir: Path,
    smoothing_window: int = 5,
    title: str | None = None,
    include_test_accuracy: bool = True,
) -> list[Path]:
    """Generate per-run and aggregated accuracy plots.

    Parameters
    ----------
    csv_paths:
        Collection of CSV files produced by :class:`ExperimentResultLogger`.
    output_dir:
        Directory where PNG/PDF artefacts should be stored.
    smoothing_window:
        Window size for the moving-average smoothing applied to the curves.
    title:
        Optional title for the combined figure.
    include_test_accuracy:
        Whether to plot the aggregated FL test accuracy alongside TraceFL's
        localization accuracy.
    """
    if not csv_paths:
        raise ValueError("No provenance CSV files matched the provided pattern")

    artefacts = PlotArtefacts(output_dir)
    frames = load_results(csv_paths)

    combined = []
    for frame, path in zip(frames, csv_paths, strict=False):
        frame = frame.sort_values("round")
        frame["TraceFL Accuracy (Smoothed)"] = _smooth(
            frame["TraceFL Accuracy (%)"], smoothing_window
        )
        if include_test_accuracy and "FL Accuracy (%)" in frame:
            frame["FL Accuracy (Smoothed)"] = _smooth(
                frame["FL Accuracy (%)"], smoothing_window
            )

        _plot_single_run(frame, path, artefacts, include_test_accuracy)
        combined.append(frame)

    combined_df = pd.concat(combined, ignore_index=True)
    _plot_combined_runs(
        combined_df,
        artefacts,
        title=title,
        include_test_accuracy=include_test_accuracy,
    )

    summary_path = export_summary_table(combined_df, artefacts)
    return [artefacts.png_dir, artefacts.pdf_dir, summary_path]


def _plot_single_run(
    frame: pd.DataFrame,
    path: Path,
    artefacts: PlotArtefacts,
    include_test_accuracy: bool,
) -> None:
    run_label = frame["Run"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))  # Increased figure size
    ax.plot(
        frame["round"],
        frame["TraceFL Accuracy (Smoothed)"],
        label="TraceFL",
        linewidth=2,
    )
    if include_test_accuracy and "FL Accuracy (Smoothed)" in frame:
        ax.plot(
            frame["round"],
            frame["FL Accuracy (Smoothed)"],
            label="Global Model",
            linestyle="--",
            linewidth=2,
        )
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)

    # Create a cleaner title by truncating long run labels
    if len(run_label) > 50:
        title = run_label[:47] + "..."
    else:
        title = run_label
    ax.set_title(title, fontsize=12, pad=20)
    ax.legend()
    fig.tight_layout()

    file_slug = path.stem
    fig.savefig(artefacts.png_dir / f"{file_slug}.png", dpi=300)
    fig.savefig(artefacts.pdf_dir / f"{file_slug}.pdf")
    plt.close(fig)


def _plot_combined_runs(
    combined_df: pd.DataFrame,
    artefacts: PlotArtefacts,
    title: str | None,
    include_test_accuracy: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))  # Increased figure size
    sns.lineplot(
        data=combined_df,
        x="round",
        y="TraceFL Accuracy (Smoothed)",
        hue="Run",
        ax=ax,
        linewidth=2,
    )

    if include_test_accuracy and "FL Accuracy (Smoothed)" in combined_df:
        sns.lineplot(
            data=combined_df,
            x="round",
            y="FL Accuracy (Smoothed)",
            hue="Run",
            style="Run",
            linestyle="--",
            legend=False,
            ax=ax,
            linewidth=1.5,
            alpha=0.6,
        )

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.legend(title="Run", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    fig.savefig(artefacts.png_dir / "combined_accuracy.png", dpi=300)
    fig.savefig(artefacts.pdf_dir / "combined_accuracy.pdf")
    plt.close(fig)


def export_summary_table(combined_df: pd.DataFrame, artefacts: PlotArtefacts) -> Path:
    """Write a summary table aggregating per-run statistics."""
    summary = (
        combined_df.groupby("Run")
        .agg(
            rounds=("round", "max"),
            tracefl_avg=("TraceFL Accuracy (%)", "mean"),
            tracefl_max=("TraceFL Accuracy (%)", "max"),
            fl_max=("FL Accuracy (%)", "max"),
        )
        .reset_index()
    )

    summary.rename(
        columns={
            "rounds": "Total Rounds",
            "tracefl_avg": "Avg. TraceFL Accuracy (%)",
            "tracefl_max": "Best TraceFL Accuracy (%)",
            "fl_max": "Best FL Accuracy (%)",
        },
        inplace=True,
    )

    summary_path = artefacts.tables_dir / "accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    latex_path = artefacts.tables_dir / "accuracy_summary.tex"
    summary.to_latex(latex_path, index=False, float_format="%.2f")

    return summary_path
