 #!/usr/bin/env bash
 set -euo pipefail

# command to run FedAvg experiment with 50% participation.
flwr run . --run-config "algorithm='fedavg' fraction-train=0.5 dataset='cifar10' batch-size=32 lr=0.01 proximal_mu=0.0 save-dir='_static/cifar10/'" --stream

# command to run FedAvg experiment with 20% participation.
flwr run . --run-config "algorithm='fedavg' fraction-train=0.2 dataset='cifar10' batch-size=32 lr=0.01 proximal_mu=0.0 save-dir='_static/cifar10/'" --stream


# command to run FedAMP experiment with 50% participation.
flwr run . --run-config "algorithm='fedamp' fraction-train=0.5 dataset='cifar10' lr=0.01 batch-size=32 alphaK=1.0 fedamp-lambda=1.0 sigma=1.0 save-dir='_static/cifar10/'" --stream

# command to run FedAMP experiment with 20% participation.
flwr run . --run-config "algorithm='fedamp' fraction-train=0.2 dataset='cifar10' lr=0.01 batch-size=32 alphaK=1.0 fedamp-lambda=1.0 sigma=1.0 save-dir='_static/cifar10/'" --stream


# command to run FedProx experiment with 50% participation.
flwr run . --run-config "algorithm='fedprox' fraction-train=0.5 dataset='cifar10' lr=0.01 batch-size=32 proximal_mu=0.6 save-dir='_static/cifar10/'" --stream

# command to run FedProx experiment with 20% participation.
flwr run . --run-config "algorithm='fedprox' fraction-train=0.2 dataset='cifar10' lr=0.01 batch-size=32 proximal_mu=0.6 save-dir='_static/cifar10/'" --stream


# command to run FedAvg experiment with 50% participation.
flwr run . --run-config "algorithm='fedavg' fraction-train=0.5 dataset='fashion' batch-size=32 lr=0.01 proximal_mu=0.0 save-dir='_static/fashion/'" --stream

# command to run FedAvg experiment with 20% participation.
flwr run . --run-config "algorithm='fedavg' fraction-train=0.2 dataset='fashion' batch-size=32 lr=0.01 proximal_mu=0.0 save-dir='_static/fashion/'" --stream


# command to run FedAMP experiment with 50% participation.
flwr run . --run-config "algorithm='fedamp' fraction-train=0.5 dataset='fashion' lr=0.01 batch-size=32 alphaK=1.0 fedamp-lambda=1.0 sigma=1.0 save-dir='_static/fashion/'" --stream

# command to run FedAMP experiment with 20% participation.
flwr run . --run-config "algorithm='fedamp' fraction-train=0.2 dataset='fashion' lr=0.01 batch-size=32 alphaK=1.0 fedamp-lambda=1.0 sigma=1.0 save-dir='_static/fashion/'" --stream


# command to run FedProx experiment with 50% participation.
flwr run . --run-config "algorithm='fedprox' fraction-train=0.5 dataset='fashion' lr=0.01 batch-size=32 proximal_mu=0.6 save-dir='_static/fashion/'" --stream

# command to run FedProx experiment with 20% participation.
flwr run . --run-config "algorithm='fedprox' fraction-train=0.2 dataset='fashion' lr=0.01 batch-size=32 proximal_mu=0.6 save-dir='_static/fashion/'" --stream

