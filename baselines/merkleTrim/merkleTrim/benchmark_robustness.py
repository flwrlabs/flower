"""
Robustness under attack — Benchmark results plotting.

Plots accuracy curves across malicious fractions alpha in {0.0, 0.20, 0.30, 0.40, 0.50}
for VerifiableRobustStrategy (beta=0.2) against plain FedAvg (beta=0.0).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── VerifiableRobustStrategy (beta=0.2) ─────────────────────────────────────
robust_accuracy = {
    0.0:  [0.6916, 0.7343, 0.7725],
    0.20: [0.7107, 0.7598, 0.8048],
    0.30: [0.7391, 0.8187, 0.8099],
    0.50: [0.7449, 0.3313, 0.5510],
}

# ── Plain FedAvg, no defense (beta=0.0) — full sweep, matches REPORT.md 3.3 ─
fedavg_accuracy = {
    0.0:  [0.6916, 0.7343, 0.7725],
    0.20: [0.7517, 0.3311, 0.5988],
    0.30: [0.7742, 0.7205, 0.7995],
    0.40: [0.7693, 0.0988, 0.0633],
    0.50: [0.7241, 0.7335, 0.1946],
}

# ── Chart 1: final-round accuracy vs malicious fraction ────────────────────

fractions_robust = sorted(robust_accuracy.keys())
fractions_fedavg = sorted(fedavg_accuracy.keys())

robust_final = [robust_accuracy[f][-1] for f in fractions_robust]
fedavg_final = [fedavg_accuracy[f][-1] for f in fractions_fedavg]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(fractions_fedavg, fedavg_final, marker='o', color='#E8703C',
         label='plain FedAvg (no defense)')
ax1.plot(fractions_robust, robust_final, marker='o', color='#2FBF8F',
         label='VerifiableRobustStrategy (beta=0.2)')
ax1.axvline(x=0.2, color='#D9A441', linestyle='--', alpha=0.6, label='beta threshold (0.2)')
ax1.set_xlabel('malicious fraction')
ax1.set_ylabel('final-round accuracy')
ax1.set_title('Damage vs attacker fraction')
ax1.legend()
ax1.grid(alpha=0.3)

# ── Chart 2: accuracy curve across rounds, one line per fraction ───────────

rounds = [1, 2, 3]
colors = {0.0: '#2FBF8F', 0.20: '#D9A441', 0.30: '#9B8EC4', 0.50: '#E8703C'}

for frac in fractions_robust:
    ax2.plot(rounds, robust_accuracy[frac], marker='o', color=colors[frac],
              label=f'{int(frac*100)}% malicious')

ax2.set_xlabel('round')
ax2.set_ylabel('accuracy')
ax2.set_title('Accuracy across rounds by attacker fraction (VerifiableRobustStrategy)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_robustness.png')
plt.savefig(output_path, dpi=150)
print(f"Chart saved to {output_path}")
plt.close()

# ── Breaking point check ────────────────────────────────────────────────────

print("\n── Breaking point check (defended strategy only) ──")
for f in fractions_robust:
    gap = robust_accuracy[f][-1] - robust_accuracy[0.0][-1]
    status = "holding" if gap > -0.05 else "COLLAPSED — past beta threshold"
    print(f"  {int(f*100)}% malicious: {status} (final accuracy vs clean baseline: {gap:+.3f})")

print("\n── Undefended baseline — non-monotonicity check ──")
for f in fractions_fedavg:
    print(f"  {int(f*100)}% malicious, no defense: final accuracy {fedavg_accuracy[f][-1]:.4f}")
print("  Note: non-monotonic across 20/30/40/50% — see REPORT.md Section 3.3, point 3.")
