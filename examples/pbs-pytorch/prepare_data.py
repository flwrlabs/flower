"""Download and verify CIFAR-10 once, inside a compute allocation."""

import os

from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    for training in (True, False):
        dataset = CIFAR10(os.environ["DATA_ROOT"], train=training, download=True)
        print(f"CIFAR-10 train={training}: {len(dataset)} examples", flush=True)
