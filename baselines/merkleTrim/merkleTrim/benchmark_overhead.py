"""
Benchmark — Aggregation Compute Time & On-Chain Payload Overhead.

Measures:
  1. Aggregation compute time: plain mean (FedAvg) vs coordinate-wise trimmed mean
  2. On-chain payload size: full update logging O(n) vs Merkle checkpoint O(1)
"""

import os
import sys
import time
import hashlib
import numpy as np

# Add parent merkleTrim directory to sys.path for fallback resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from merkleTrim.custom_strategy import VerifiableRobustStrategy


def make_synthetic_clients(n_clients, n_params=15000):
    """
    n_params=15000 float32 values ≈ 60,000 bytes ≈ 0.06 MB per client,
    matching standard ArrayRecord size per client.
    """
    return [np.random.randn(n_params).astype(np.float32) for _ in range(n_clients)]


def naive_full_logging_bytes(clients_arrays):
    """Payload size required to log every client's complete update vector on-chain."""
    return sum(arr.nbytes for arr in clients_arrays)


def merkle_checkpoint_bytes():
    """Succinct state commitment design: single 32-byte SHA-256 root."""
    return 32


def time_plain_mean(clients_arrays, repeats=20):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = np.mean(clients_arrays, axis=0)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000  # ms


def time_trimmed_mean(clients_arrays, beta=0.2, repeats=20):
    strategy = VerifiableRobustStrategy(beta=beta)
    arrays_list = [[a] for a in clients_arrays]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = strategy._trimmed_mean(arrays_list, beta)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000


def time_merkle(clients_arrays, repeats=20):
    strategy = VerifiableRobustStrategy()
    commitments = [hashlib.sha256(a.tobytes()).digest() for a in clients_arrays]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = strategy._merkle_root(commitments)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000


# ── Run evaluation across increasing client counts ─────────────────────────

client_counts = [5, 20, 50, 100, 500]

naive_bytes_list   = []
merkle_bytes_list  = []
plain_times_list   = []
trimmed_times_list = []
merkle_times_list  = []

print(f"{'n_clients':>10} | {'plain mean (ms)':>16} | {'trimmed mean (ms)':>18} | "
      f"{'merkle (ms)':>12} | {'naive bytes':>14} | {'merkle bytes':>12}")
print("-" * 100)

for n in client_counts:
    clients = make_synthetic_clients(n)

    t_plain   = time_plain_mean(clients)
    t_trimmed = time_trimmed_mean(clients)
    t_merkle  = time_merkle(clients)

    naive_b  = naive_full_logging_bytes(clients)
    merkle_b = merkle_checkpoint_bytes()

    naive_bytes_list.append(naive_b)
    merkle_bytes_list.append(merkle_b)
    plain_times_list.append(t_plain)
    trimmed_times_list.append(t_trimmed)
    merkle_times_list.append(t_merkle)

    print(f"{n:>10} | {t_plain:>16.4f} | {t_trimmed:>18.4f} | {t_merkle:>12.4f} | "
          f"{naive_b:>14,} | {merkle_b:>12}")

print("\nBenchmark table complete.")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(client_counts, naive_bytes_list, marker='o', color='#E8703C',
         label='naive full logging — O(n)')
ax1.plot(client_counts, merkle_bytes_list, marker='o', color='#2FBF8F',
         label='Merkle checkpoint — O(1)')
ax1.set_yscale('log')
ax1.set_xlabel('number of clients')
ax1.set_ylabel('bytes posted on-chain per round (log scale)')
ax1.set_title('On-chain payload: naive logging vs Merkle checkpoint')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(client_counts, plain_times_list, marker='o', color='#8B8F9C',
         label='plain mean (FedAvg)')
ax2.plot(client_counts, trimmed_times_list, marker='o', color='#D9A441',
         label='trimmed mean')
ax2.plot(client_counts, merkle_times_list, marker='o', color='#9B8EC4',
         label='Merkle root build')
ax2.set_xlabel('number of clients')
ax2.set_ylabel('compute time (ms, median of 20 runs)')
ax2.set_title('Aggregation compute time')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_overhead.png')
plt.savefig(output_path, dpi=150)
print(f"Chart saved to {output_path}")
plt.close()
