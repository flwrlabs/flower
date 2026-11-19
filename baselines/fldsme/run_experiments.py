#!/usr/bin/env python3
"""Run the fldsme ablation sweep: control vs DSME vs energy-aware DSME.

Usage
-----
    python run_experiments.py --baseline-dir /path/to/baselines/fldsme \
                              --seeds 0 1 2 3 4 \
                              --rounds 20

Each run is a separate ``flwr run`` subprocess. The server is expected to write
a JSON file to the path given by the ``results-path`` run-config key (see the
``save_result_json`` helper in README_ADDONS.md). Console output is also kept so
that a failed run can be inspected.

Output layout::

    results/
      raw/<arm>__seed<k>.json      machine-readable metrics per round
      raw/<arm>__seed<k>.log       captured stdout+stderr
      index.json                   manifest consumed by plot_results.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Each arm is a name plus the run-config overrides that define it.
# Keys must match the ones declared in pyproject.toml under
# [tool.flwr.app.config]. Add any missing keys there first.
ARMS: dict[str, dict[str, object]] = {
    # Control: no MAC-layer gating at all. This is the curve that lets the cost
    # of the DSME constraints be measured rather than assumed.
    "fedavg_no_dsme": {
        "dsme-enabled": False,
    },
    # MAC gate and GTS bandwidth limit active.
    "dsme_random": {
        "dsme-enabled": True,
    },
}

# NOTE ON REMOVED ARMS
# Earlier revisions of this file declared arms differing only by a
# `selection-policy` key (eligible / greedy / proportional) plus a harvesting
# variant. Those settings are not read anywhere: DSMEFedAvg inherits FedAvg's
# uniform `configure_train` sampling, so every one of those arms ran the
# identical code path as `dsme_random` and could not have measured the policy
# it named. They have been removed rather than left as dead configuration.
#
# Running them anyway was not wasted: five seeds of `dsme_eligible` against five
# of `dsme_random` is the same code under a different RNG stream, which gives a
# 0.7-point accuracy spread. That is the run-to-run noise floor quoted in the
# README, and the bar any real selection policy has to clear.
#
# When energy-aware selection is implemented, it belongs in `strategy.py` as an
# override of `configure_train`, with the arm re-added here at that point.


def as_run_config(overrides: dict[str, object]) -> str:
    """Render a dict as a ``flwr run --run-config`` string."""
    parts = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, str):
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def run_one(
    baseline_dir: Path,
    arm: str,
    overrides: dict[str, object],
    seed: int,
    rounds: int,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, object]:
    tag = f"{arm}__seed{seed}"
    json_path = out_dir / "raw" / f"{tag}.json"
    log_path = out_dir / "raw" / f"{tag}.log"

    cfg = dict(overrides)
    cfg["seed"] = seed
    cfg["num-server-rounds"] = rounds
    cfg["results-path"] = str(json_path)

    # --stream is required: without it `flwr run` submits the run to the
    # SuperLink and returns immediately, so the subprocess would "succeed"
    # before any training happened and the results file would never appear.
    cmd = [
        "flwr", "run", str(baseline_dir),
        "--run-config", as_run_config(cfg),
        "--stream",
    ]
    print(f"  -> {tag}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"arm": arm, "seed": seed, "json": str(json_path), "status": "dry-run"}

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n===== STDERR =====\n" + proc.stderr)
    elapsed = time.time() - started

    status = "ok" if proc.returncode == 0 and json_path.exists() else "failed"
    if status == "failed":
        reason = (
            f"rc={proc.returncode}"
            if proc.returncode != 0
            else "run finished but no results JSON was written"
        )
        print(f"     FAILED ({reason}) — see {log_path}", flush=True)
    else:
        print(f"     done in {elapsed:.0f}s", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "json": str(json_path),
        "log": str(log_path),
        "status": status,
        "returncode": proc.returncode,
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=sorted(ARMS),
        choices=sorted(ARMS),
        help="Subset of arms to run. Defaults to all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    (args.out / "raw").mkdir(parents=True, exist_ok=True)

    manifest = []
    total = len(args.arms) * len(args.seeds)
    n = 0
    for arm in args.arms:
        print(f"\n[{arm}]")
        for seed in args.seeds:
            n += 1
            print(f"({n}/{total})", end=" ")
            manifest.append(
                run_one(
                    args.baseline_dir,
                    arm,
                    ARMS[arm],
                    seed,
                    args.rounds,
                    args.out,
                    args.dry_run,
                )
            )

    index = {
        "rounds": args.rounds,
        "seeds": args.seeds,
        "arms": {a: ARMS[a] for a in args.arms},
        "runs": manifest,
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=2))

    failed = [r for r in manifest if r["status"] == "failed"]
    print(f"\n{len(manifest) - len(failed)}/{len(manifest)} runs ok")
    if failed:
        print("failed:", ", ".join(f"{r['arm']}/seed{r['seed']}" for r in failed))
    print(f"manifest: {args.out / 'index.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
