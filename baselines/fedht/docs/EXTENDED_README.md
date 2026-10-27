# FedHT Extended Results

This document contains detailed per-experiment results for the Fed-HT and FedIter-HT baselines.
Results are structured to match Figures 3 and 5 from the paper.

---

## Simulation I: Sparse Linear Regression

**Setup:** 100 clients, 100 samples each, feature dimension d=1000, tau=200, alpha=0.1, beta=0.1.
The first 100 elements of each local coefficient vector are drawn from N(u_i, 1), the rest are zero.

**Objective function:** Mean squared error.

**Command:**
```bash
flwr run . local-simulation --run-config "algorithm.num-server-rounds=100 algorithm.tau=200"
```

**Results (objective value vs communication rounds):**

Best configurations: Distributed-IHT lr=0.01 K=1, Fed-HT lr=0.003 K=3, FedIter-HT lr=0.03 K=1.

| Rounds | Distributed-IHT | Fed-HT (K=3) | FedIter-HT (K=1) |
| :---: | :---: | :---: | :---: |
| 1 | 55.09 | 31.65 | 46.80 |
| 20 | 14.28 | 5.90 | 23.48 |
| 40 | 8.44 | 5.60 | 17.43 |
| 60 | 6.73 | 5.61 | 15.48 |
| 80 | 5.98 | 5.60 | 13.18 |
| 100 | 5.71 | 5.62 | 10.09 |

Fed-HT (K=3) reaches the same objective as Distributed-IHT in approximately 28 communication rounds (vs 100 for Distributed-IHT), a 3.5x improvement in communication efficiency.

FedIter-HT achieves its best result with K=1 (lr=0.03), reaching 10.09 at round 100. With K=3, the local hard thresholding after each step causes clients to develop misaligned sparse supports; averaging 100 such sparse vectors injects noise, producing oscillation around 14–17. With K=1 there is no drift accumulation across local steps, allowing a higher learning rate and steady (if slow) convergence.

Best hyperparameters found by grid search:

| Algorithm | K | Learning rate |
| :--- | :---: | :---: |
| Fed-HT | 3 | 0.003 |
| FedIter-HT | 1 | 0.03 |
| Distributed-IHT | 1 | 0.01 |

Plot: `../_static/simulation_I_comparison.png`

---

## Simulation II: Sparse Logistic Regression

**Setup:** 100 clients, 1000 samples each, feature dimension d=1000, tau=200, alpha=1.0, beta=1.0.
Binary labels: top-100 samples per client by sigmoid score are assigned label 1.

**Objective function:** Binary cross-entropy.

**Command:**
```bash
flwr run . local-simulation --run-config 'dataset.name="simulation_II" model.input-dim=1000 algorithm.tau=200'
```

**Results (objective value vs communication rounds):**

| Rounds | Distributed-IHT | Fed-HT (K=5) | FedIter-HT (K=5) |
| :---: | :---: | :---: | :---: |
| 0 | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD |
| 150 | TBD | TBD | TBD |
| 200 | TBD | TBD | TBD |

Best hyperparameters found by grid search:

| Algorithm | K | Learning rate |
| :--- | :---: | :---: |
| Fed-HT | TBD | TBD |
| FedIter-HT | TBD | TBD |
| Distributed-IHT | 1 | TBD |

Plot: `../_static/simulation_II_comparison.png`

---

## MNIST: Sparse Softmax Regression

**Setup:** 100 clients, 600 samples each, feature dimension d=784 (flattened images), tau=500, 10 classes.
Each client holds data from 2 digit classes (non-IID).

**Objective function:** Cross-entropy.

**Command:**
```bash
flwr run . local-simulation --run-config 'dataset.name="mnist" dataset.batch-size=64 algorithm.tau=500 model.input-dim=784 model.num-classes=10'
```

**Results (objective value vs communication rounds):**

| Rounds | Distributed-IHT | Fed-HT (K=5) | FedIter-HT (K=5) |
| :---: | :---: | :---: | :---: |
| 0 | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD |

Plot: `../_static/mnist_comparison.png`

---

## Notes on Reproducing the Paper

The paper does not report exact final hyperparameter values for the grid search, only the search ranges.
The best values for each experiment were obtained by the baseline authors via grid search:

- K searched over {3, 5, 8, 10}
- Learning rate searched over {10, 1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01, 0.001}
- All algorithms initialized with x_0 = 0

For Simulations I and II, the paper reports results on the objective value vs both communication rounds (Figure 3)
and total internal iterations (Figure 3 right panels). The communication round comparison is the primary metric
because it reflects the practical advantage of federated methods.

The paper also reports results on real datasets (E2006-tfidf, RCV1, MNIST) in Figure 5.
Implementation of E2006-tfidf and RCV1 loaders is in progress and requires downloading from the LibSVM website.
