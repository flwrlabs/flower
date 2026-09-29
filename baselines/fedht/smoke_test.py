"""Smoke test for the FedHT baseline.

Validates core components without running a full federated simulation:
  1. The hard thresholding operator
  2. Simulation data generation
  3. One round of local training for each task type (linear, logistic, softmax)
  4. Weight serialization round-trip

Run with:
    cd ~/Projects/fedht
    python3 smoke_test.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from fedht.dataset import generate_simulation_I, generate_simulation_II, make_loaders
from fedht.model import (
    SparseLinearRegression,
    SparseLogisticRegression,
    SparseSoftmaxRegression,
    eval_linear,
    eval_logistic,
    eval_softmax,
    get_weights,
    set_weights,
    train_linear,
    train_logistic,
    train_softmax,
)
from fedht.strategy import hard_threshold

# ---------------------------------------------------------------------------
# 1. Hard thresholding
# ---------------------------------------------------------------------------

print("=== Testing hard_threshold ===")

rng = np.random.default_rng(0)
arrays = [rng.standard_normal(100), rng.standard_normal(50)]
tau = 30

thresholded = hard_threshold(arrays, tau)
total_nonzero = sum((a != 0).sum() for a in thresholded)
assert total_nonzero == tau, f"Expected {tau} nonzero, got {total_nonzero}"

flat_original = np.concatenate([a.flatten() for a in arrays])
flat_result = np.concatenate([a.flatten() for a in thresholded])
kept_magnitudes = np.abs(flat_original[flat_result != 0])
dropped_magnitudes = np.abs(flat_original[flat_result == 0])
assert (
    kept_magnitudes.min() >= dropped_magnitudes.max() - 1e-9
), "Hard threshold kept a smaller value than a dropped one"

print(f"  Total nonzero after tau={tau}: {total_nonzero} (correct)")
print(f"  Minimum kept magnitude: {kept_magnitudes.min():.6f}")
print(f"  Maximum dropped magnitude: {dropped_magnitudes.max():.6f}")
print("  PASS")


# ---------------------------------------------------------------------------
# 2. Simulation data generation
# ---------------------------------------------------------------------------

print("\n=== Testing generate_simulation_I ===")

sim1 = generate_simulation_I(num_clients=10, num_samples=100, d=1000, seed=42)
assert len(sim1) == 10, f"Expected 10 clients, got {len(sim1)}"
X0, y0 = sim1[0]
assert X0.shape == (100, 1000), f"Unexpected X shape: {X0.shape}"
assert y0.shape == (100,), f"Unexpected y shape: {y0.shape}"

X1, _ = sim1[1]
assert not np.allclose(
    X0, X1
), "Client data looks identical -- non-IID generation may be broken"

print(f"  Client 0 X shape: {X0.shape}, y shape: {y0.shape}")
print(f"  y range: [{y0.min():.3f}, {y0.max():.3f}]")
print("  PASS")

print("\n=== Testing generate_simulation_II ===")

sim2 = generate_simulation_II(num_clients=10, num_samples=200, d=1000, seed=42)
assert len(sim2) == 10
X0, y0 = sim2[0]
assert set(np.unique(y0)).issubset({0.0, 1.0}), "Labels should be binary"
positive_count = int(y0.sum())
assert positive_count == 100, f"Expected 100 positive labels, got {positive_count}"

print(f"  Client 0 X shape: {X0.shape}, positive labels: {positive_count}/200")
print("  PASS")


# ---------------------------------------------------------------------------
# 3. DataLoaders
# ---------------------------------------------------------------------------

print("\n=== Testing make_loaders ===")

X, y = sim1[0]
train_loader, val_loader = make_loaders(X, y, batch_size=20)
train_batches = list(train_loader)
assert len(train_batches) > 0, "Train loader returned no batches"
Xb, yb = train_batches[0]
assert Xb.shape[1] == 1000, f"Unexpected batch feature dim: {Xb.shape}"
print(f"  Train batches: {len(train_batches)}, first batch X: {tuple(Xb.shape)}")
print("  PASS")


# ---------------------------------------------------------------------------
# 4. Local training -- linear regression (Simulation I)
# ---------------------------------------------------------------------------

print("\n=== Testing train_linear (Simulation I) ===")

net = SparseLinearRegression(input_dim=1000)
torch.nn.init.zeros_(net.linear.weight)

device = torch.device("cpu")
loss_before, _ = eval_linear(net, val_loader, device)
train_linear(net, train_loader, device, lr=0.1, local_steps=5)
loss_after, _ = eval_linear(net, val_loader, device)

print(f"  MSE before training: {loss_before:.6f}")
print(f"  MSE after 5 steps:   {loss_after:.6f}")

weights = get_weights(net)
nonzero_total = sum((w != 0).sum() for w in weights)
print(f"  Nonzero parameters after plain SGD: {nonzero_total}")
print("  PASS")


# ---------------------------------------------------------------------------
# 5. Local training with hard thresholding (FedIter-HT client side)
# ---------------------------------------------------------------------------

print("\n=== Testing train_linear with local HT (FedIter-HT) ===")

net_ht = SparseLinearRegression(input_dim=1000)
torch.nn.init.zeros_(net_ht.linear.weight)

train_linear(
    net_ht, train_loader, device, lr=0.1, local_steps=5, use_local_ht=True, tau=200
)
weights_ht = get_weights(net_ht)
nonzero_ht = sum((w != 0).sum() for w in weights_ht)

assert nonzero_ht <= 200, f"Expected at most 200 nonzero, got {nonzero_ht}"
print(f"  Nonzero parameters after local HT (tau=200): {nonzero_ht}")
print("  PASS")


# ---------------------------------------------------------------------------
# 6. Local training -- logistic regression (Simulation II)
# ---------------------------------------------------------------------------

print("\n=== Testing train_logistic (Simulation II) ===")

X2, y2 = sim2[0]
train_loader2, val_loader2 = make_loaders(X2, y2, batch_size=20)

net2 = SparseLogisticRegression(input_dim=1000)
torch.nn.init.zeros_(net2.linear.weight)

loss_before2, acc_before2, _ = eval_logistic(net2, val_loader2, device)
train_logistic(net2, train_loader2, device, lr=0.1, local_steps=5)
loss_after2, acc_after2, _ = eval_logistic(net2, val_loader2, device)

print(f"  BCE before training: {loss_before2:.6f}  accuracy: {acc_before2:.4f}")
print(f"  BCE after 5 steps:   {loss_after2:.6f}  accuracy: {acc_after2:.4f}")
print("  PASS")


# ---------------------------------------------------------------------------
# 7. Local training -- softmax regression (MNIST-style)
# ---------------------------------------------------------------------------

print("\n=== Testing train_softmax (MNIST-style synthetic) ===")

rng_sm = np.random.default_rng(7)
X_sm = rng_sm.standard_normal((200, 784)).astype(np.float32)
y_sm = rng_sm.integers(0, 10, size=200).astype(np.int64)
train_loader_sm, val_loader_sm = make_loaders(X_sm, y_sm, batch_size=32)

net_sm = SparseSoftmaxRegression(input_dim=784, num_classes=10)
torch.nn.init.zeros_(net_sm.linear.weight)

loss_sm_before, acc_sm_before, _ = eval_softmax(net_sm, val_loader_sm, device)
train_softmax(net_sm, train_loader_sm, device, lr=0.01, local_steps=5)
loss_sm_after, acc_sm_after, _ = eval_softmax(net_sm, val_loader_sm, device)

print(f"  CE before training: {loss_sm_before:.6f}  accuracy: {acc_sm_before:.4f}")
print(f"  CE after 5 steps:   {loss_sm_after:.6f}  accuracy: {acc_sm_after:.4f}")
print("  PASS")


# ---------------------------------------------------------------------------
# 8. get_weights / set_weights round-trip
# ---------------------------------------------------------------------------

print("\n=== Testing get_weights / set_weights round-trip ===")

original_weights = get_weights(net)
net_copy = SparseLinearRegression(input_dim=1000)
set_weights(net_copy, original_weights)
copied_weights = get_weights(net_copy)

for w1, w2 in zip(original_weights, copied_weights):
    assert np.allclose(w1, w2), "Weights differ after round-trip"

print("  Round-trip weights match exactly")
print("  PASS")


print("\nAll smoke tests passed.")
