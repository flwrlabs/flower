from tqdm import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from fedSSL.model import SimClrTransform


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(transform_fn(True))

    return partition


def transform_fn(augment_data):
    simclr_transform = SimClrTransform(size=32)

    def apply_transforms(batch):
        batch["img"] = [simclr_transform(img, augment_data) for img in batch["img"]]
        return batch

    return apply_transforms


def train(net, cid, trainloader, optimizer, criterion, epochs, device):
    net.to(device)  # move model to GPU if available
    net.train()
    criterion.to(device)
    num_batches = len(trainloader)
    total_loss = 0

    with tqdm(total=num_batches * epochs, desc=f'Client {cid} Local Train', position=0, leave=True) as pbar:
        for epoch in range(epochs):
            for item in trainloader:
                x_i, x_j = item['img']
                x_i, x_j = x_i.to(device), x_j.to(device)
                optimizer.zero_grad()

                z_i = net(x_i)
                z_j = net(x_j)

                loss = criterion(z_i, z_j)
                total_loss += loss

                loss.backward()
                optimizer.step()

                pbar.update(1)

    return {'loss': float(total_loss / num_batches)}
