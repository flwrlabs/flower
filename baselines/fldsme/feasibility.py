"""Feasibility horizon for federated learning under a finite energy budget.

Standalone: imports only numpy and your ``dsme_model``. Runs no training and
touches nothing in the baseline, so it is safe to use while a sweep is running.

------------------------------------------------------------------------------
The metric
------------------------------------------------------------------------------
FL papers report accuracy at round N. On battery-powered hardware that question
is incomplete, because round N may not be reachable: the client runs out of
energy first. The missing quantity is how long the federation stays trainable.

    Feasibility horizon R*
        The largest round index r at which at least one client still has
        enough residual energy to complete a local training round.

    Contiguous horizon R*_cont
        The last round before the first unreachable one. Training up to here is
        uninterrupted. This is the conservative, and usually the more useful,
        number: it is the schedule you can actually promise.

    Accuracy at the horizon  A(R*)
        Global accuracy at the horizon, rather than at an arbitrary round budget.

Harvesting makes these two differ. Between harvest events a client can dip
below the cost and recover afterwards, so participation past R*_cont is bursty:
the federation is alive but only trains on the rounds following a recharge. A
large gap between R*_cont and R* is therefore a design signal in itself - it
says the schedule is dictated by the harvest period, not by the budget.

A(R*) is the honest headline number for an energy-constrained federation.
Reporting accuracy at round 20 when R* = 12 describes eight rounds that the
hardware cannot pay for.

------------------------------------------------------------------------------
Closed form
------------------------------------------------------------------------------
For a client in cluster c, with initial budget B0, per-round drain d_c,
harvest H every P rounds, the residual at round r is

    B_c(r) = max(B_floor, B0 - r * d_c + H * floor(r / P))

and the client can train iff B_c(r) >= E(m, epochs, size), the per-round cost
from the MAC model. The horizon for the federation is

    R* = max over c of { largest r <= R_budget with B_c(r) >= E }

Ignoring the sawtooth (averaging the harvest to H/P per round) gives the smooth
envelope and a closed-form estimate:

    R*_approx = (B0 - E) / (d_c - H/P)          maximised over c

The approximation is an upper bound on the exact value: it misses the mid-cycle
dips between harvest events, during which a client can be temporarily
ineligible. Both are reported below, and the gap between them is itself
informative - a large gap means participation is bursty rather than a clean
cutoff.

------------------------------------------------------------------------------
Why the two modes matter
------------------------------------------------------------------------------
E depends on the CAP mode. Under the SeCAP model NCR costs roughly 39% of CR,
so R*(NCR) is far beyond R*(CR). Once r > R*(CR), *every* round that trains at
all is an NCR round. NCR mode stops being an efficiency optimisation and
becomes the necessary condition for the federation to continue.

Usage
-----
    python feasibility.py                       # defaults from the baseline
    python feasibility.py --rounds 40 --epochs 1
    python feasibility.py --sweep-size          # horizon vs model size
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

try:
    from fldsme.dsme_model import DSMEMACModel
except ImportError:  # allow running from inside the package directory
    from dsme_model import DSMEMACModel  # type: ignore

# Matches MODEL_SIZE_KB in client_app.py: 61,770 params * 4 bytes / 1024.
DEFAULT_MODEL_KB = 61770 * 4 / 1024


@dataclass
class Horizon:
    """Result of a horizon computation for one CAP mode."""

    mode: str
    cost_mj: float
    exact: int              # last reachable round, sawtooth included
    approx: float           # smooth-envelope estimate
    best_cluster: int
    trainable_rounds: int   # how many rounds in 1..R_budget are reachable
    first_gap: int | None   # first round that is NOT reachable, if any
    contiguous: int         # last round before the first gap (uninterrupted)


def cluster_drain(mac: DSMEMACModel, cluster_id: int) -> float:
    """Per-round depletion for a cluster, mirroring get_client_profile."""
    return 1.5 + (cluster_id % 3) * 0.8


def residual_mj(mac: DSMEMACModel, cluster_id: int, fl_round: int) -> float:
    """Residual budget at the start of a round, mirroring get_client_profile."""
    depletion = fl_round * cluster_drain(mac, cluster_id)
    recharge = 5.0 * (fl_round // 5)
    return max(5.0, mac.energy_budget - depletion + recharge)


def compute_horizon(
    mac: DSMEMACModel,
    model_kb: float,
    epochs: int,
    mode: str,
    max_rounds: int,
) -> Horizon:
    """Exact and approximate feasibility horizon for one CAP mode."""
    cost = mac.energy_per_round_mj(model_kb, epochs, mode)

    reachable: set[int] = set()
    best_round, best_cluster = 0, -1
    for cluster in range(mac.num_clusters):
        for r in range(1, max_rounds + 1):
            if residual_mj(mac, cluster, r) >= cost:
                reachable.add(r)
                if r > best_round:
                    best_round, best_cluster = r, cluster

    # smooth envelope: harvest averaged to 1.0 mJ/round (5.0 every 5 rounds)
    approx = 0.0
    for cluster in range(mac.num_clusters):
        net_drain = cluster_drain(mac, cluster) - 1.0
        if net_drain > 0:
            approx = max(approx, (mac.energy_budget - cost) / net_drain)
        else:
            approx = float("inf")

    gap = next((r for r in range(1, best_round + 1) if r not in reachable), None)

    return Horizon(
        mode=mode,
        cost_mj=cost,
        exact=best_round,
        approx=approx,
        best_cluster=best_cluster,
        trainable_rounds=len(reachable),
        first_gap=gap,
        contiguous=(gap - 1) if gap else best_round,
    )


def report(mac: DSMEMACModel, model_kb: float, epochs: int, max_rounds: int) -> None:
    print(f"\nmodel {model_kb:.1f} KB | {epochs} local epoch(s) | "
          f"budget {mac.energy_budget:.0f} mJ | horizon capped at {max_rounds}\n")
    print(f"{'mode':>5} | {'cost/round':>10} | {'R* cont':>7} | {'R* last':>7} "
          f"| {'R* approx':>9} | {'reachable':>9} | {'first gap':>9}")
    print("-" * 78)
    for mode in ("CR", "NCR"):
        h = compute_horizon(mac, model_kb, epochs, mode, max_rounds)
        approx = "inf" if h.approx == float("inf") else f"{h.approx:.1f}"
        gap = str(h.first_gap) if h.first_gap else "-"
        print(f"{h.mode:>5} | {h.cost_mj:>7.2f} mJ | {h.contiguous:>7} | {h.exact:>7} "
              f"| {approx:>9} | {h.trainable_rounds:>9} | {gap:>9}")

    cr = compute_horizon(mac, model_kb, epochs, "CR", max_rounds)
    ncr = compute_horizon(mac, model_kb, epochs, "NCR", max_rounds)
    print()
    if ncr.exact > cr.exact:
        print(f"Past round {cr.exact}, no client can afford a CR-mode round.")
        print(f"Every round that trains from {cr.exact + 1} to {ncr.exact} is an NCR round:")
        print("NCR is not an optimisation in that regime, it is the enabling condition.")
    else:
        print("CR and NCR horizons coincide - the budget is not the binding constraint here.")


def sweep_size(mac: DSMEMACModel, epochs: int, max_rounds: int) -> None:
    """Horizon as a function of model size.

    Cost is linear in model size, so the horizon falls monotonically. Combined
    with measured accuracy per round, this is what makes an optimum exist:
    a smaller model converges more slowly per round but survives more rounds.
    """
    print(f"\nhorizon vs model size | {epochs} epoch(s) | capped at {max_rounds} rounds\n")
    print(f"{'size (KB)':>9} | {'CR cost':>8} | {'R* (CR)':>7} | "
          f"{'NCR cost':>8} | {'R* (NCR)':>8} | {'reachable (NCR)':>15}")
    print("-" * 74)
    for kb in (15, 30, 60, 120, 241, 480, 960):
        cr = compute_horizon(mac, kb, epochs, "CR", max_rounds)
        ncr = compute_horizon(mac, kb, epochs, "NCR", max_rounds)
        print(f"{kb:>9} | {cr.cost_mj:>5.1f} mJ | {cr.exact:>7} | "
              f"{ncr.cost_mj:>5.1f} mJ | {ncr.exact:>8} | {ncr.trainable_rounds:>15}")
    print("\nThe horizon is the budget side of the tradeoff. Pair it with measured")
    print("accuracy per round to find the model size that maximises A(R*).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-kb", type=float, default=DEFAULT_MODEL_KB)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=100,
                    help="Cap on the search. Use 20 to match the baseline run.")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--sweep-size", action="store_true")
    args = ap.parse_args()

    mac = DSMEMACModel(bo=6, mo=5, so=3, num_clients=10, num_clusters=4,
                       energy_budget=args.budget, bandwidth_frac=0.8)

    if args.sweep_size:
        sweep_size(mac, args.epochs, args.rounds)
    else:
        report(mac, args.model_kb, args.epochs, args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
