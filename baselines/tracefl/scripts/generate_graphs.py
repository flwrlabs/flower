"""Command-line entry point for generating TraceFL experiment figures."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tracefl.plotting import plot_accuracy_curves


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TraceFL accuracy plots from provenance CSV files",
    )
    parser.add_argument(
        "--results-dir",
        default="results_csvs",
        type=Path,
        help="Directory containing provenance CSV files (default: results_csvs)",
    )
    parser.add_argument(
        "--pattern",
        default="prov_*.csv",
        help="Glob pattern used to locate CSV files inside --results-dir",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("graphs"),
        type=Path,
        help="Directory where plots/tables should be written (default: graphs)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling window used for curve smoothing (default: 5 rounds)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the combined accuracy plot",
    )
    parser.add_argument(
        "--skip-test-accuracy",
        action="store_true",
        help="Disable plotting of the aggregated FL test accuracy curve",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate graphs from experiment results."""
    args = _parse_args(argv)

    # Check if pattern includes a path (e.g., "results/experiment_a/prov_*.csv")
    pattern_path = Path(args.pattern)
    if pattern_path.parent != Path("."):
        # Pattern includes directory, use it directly
        csv_paths = sorted(Path(".").glob(args.pattern))
        if not csv_paths:
            raise SystemExit(f"No provenance CSVs matching '{args.pattern}' found")
    else:
        # Pattern is just a filename pattern, use results_dir
        results_dir: Path = args.results_dir
        if not results_dir.exists():
            raise SystemExit(f"Results directory '{results_dir}' does not exist")

        csv_paths = sorted(results_dir.glob(args.pattern))
        if not csv_paths:
            raise SystemExit(
                f"No provenance CSVs matching '{args.pattern}' found in {results_dir}"
            )

    artefacts = plot_accuracy_curves(
        csv_paths,
        Path(args.output_dir),
        smoothing_window=max(args.smooth_window, 1),
        title=args.title,
        include_test_accuracy=not args.skip_test_accuracy,
    )

    print("Generated artefacts:")
    for artefact in artefacts:
        print(f" - {artefact}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
