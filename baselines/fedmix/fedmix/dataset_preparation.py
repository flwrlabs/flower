"""..."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset, TensorDataset
import os
import random
import json
from torchvision.datasets import CIFAR10, CIFAR100
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split

def _download_cifar10():
    """..."""
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    trainset = CIFAR10("./dataset", train=True, download=True, transform=train_transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=test_transform)
    return trainset, testset


def _download_cifar100():
    """..."""
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=test_transform)
    return trainset, testset


def _partition_cifar(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    """..."""
    partition_size = int(len(trainset) / num_clients)
    np.random.seed(seed)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)

    client_datasets = []

    # indices_used = set()

    for _client_id in range(num_clients):
        selected_classes = np.random.choice(
            num_classes, num_classes_per_client, replace=False
        )
        selected_indices = []

        for class_idx in selected_classes:
            selected_indices.extend(class_indices[class_idx])

        np.random.shuffle(selected_indices)
        client_indices = selected_indices[:partition_size]

        # indices_used.update(client_indices)

        client_dataset = Subset(trainset, client_indices)
        client_datasets.append(client_dataset)

    # print(len(indices_used))

    return client_datasets


def _partition_cifar_new(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    """..."""
    labels = set(range(num_classes))
    np.random.seed(seed)

    nlbl = [
        np.random.choice(len(labels), num_classes_per_client, replace=False)
        for u in range(num_clients)
    ]
    check = set().union(*[set(a) for a in nlbl])

    while len(check) < len(labels):
        missing = labels - check
        for m in missing:
            nlbl[np.random.randint(0, num_clients)][
                np.random.randint(0, num_classes_per_client)
            ] = m
        check = set().union(*[set(a) for a in nlbl])

    class_map = {c: [u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
    assignment = np.zeros(len(trainset))
    targets = np.array(trainset.targets)

    for lbl, users in class_map.items():
        ids = np.where(targets == lbl)[0]
        assignment[ids] = np.random.choice(users, len(ids))

    dataset_indices = [np.where(assignment == i)[0] for i in range(num_clients)]

    print('number of data points of each client:', [len(lst) for lst in dataset_indices])
    print('total number of data points across clients:', sum([len(lst) for lst in dataset_indices]))
    print('number of unique data points:', len({i for lst in dataset_indices for i in lst}))

    return [Subset(trainset, ind) for ind in dataset_indices]

def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset

def _partition_cifar_new_new(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    random.seed(seed)
    trainset.targets = torch.tensor(trainset.targets)
    trainset_sorted = _sort_by_class(trainset)

    # create sub-dataset, one per class
    img_count, _ = np.histogram(trainset.targets, bins=num_classes)
    num_images_class = img_count[0] # number of images per class
    print(f"{num_images_class = }")
    classes_buckets = [Subset(trainset_sorted, np.arange(i*num_images_class, (i+1)*num_images_class)) for i in range(num_classes)]

    # Figure out how many buckets we need to create given
    total_buckets = num_clients * num_classes_per_client
    print(f"{total_buckets = }")
    buckets_per_class = int(total_buckets / num_classes)
    print(f"{buckets_per_class = }")

    # construct dictionary of data buckets
    # each entry corresponds to the buckets of a class
    # buckets in a class have been constructed by evenly splitting the contents of that class
    # taking into account the number of clients and number of classes clients will take
    data_buckets_dict = {}
    for cls_id, cls_bucket in enumerate(classes_buckets):
        imgs_bucket = len(cls_bucket)
        imgs_per_data_bucket = imgs_bucket // buckets_per_class
        data_buckets = [Subset(cls_bucket, np.arange(i*imgs_per_data_bucket, min(imgs_bucket, (i+1)*imgs_per_data_bucket))) for i in range(buckets_per_class)]
        data_buckets_dict[cls_id] = data_buckets

    def get_data_buckets_for_client(buckets_dict):

        def _pop_from_buckets(class_id):

            val = buckets_dict[class_id].pop(-1)
            if len(buckets_dict[class_id]) == 0:
                del buckets_dict[class_id]

            return val

        classes_remaining = list(buckets_dict.keys())
        if len(classes_remaining) == 1:
            # If one class remains, take all the samples
            classes_to_use = classes_remaining * len(buckets_dict[classes_remaining[0]])
        else:
            classes_to_use = random.sample(classes_remaining, min(len(classes_remaining), num_classes_per_client))

        return ConcatDataset([_pop_from_buckets(i) for i in classes_to_use])

    datasets = [get_data_buckets_for_client(data_buckets_dict) for _ in range(num_clients)]

    print('number of data points of each client:', [len(dd) for dd in datasets])
    print('total number of data points across clients:', sum([len(dd) for dd in datasets]))
    unique_lbls_count = []
    for dd in datasets:
        num_labels = len(set([int(lbl) for _, lbl in dd]))
        unique_lbls_count.append(num_labels)
    print('number of unique data points:', unique_lbls_count)
    return datasets


def _download_femnist(num_clients):
    os.system(f'cd fedmix/femnist && ./preprocess.sh -s niid --iu {num_clients / 3550} --sf 0.1 -t sample --smplseed 1697538548 --spltseed 1697539598')


def _partition_femnist(num_clients, seed):
    train_path = 'fedmix/femnist/data/train'
    random.seed(seed)

    # train_path = '/scratch/apoorva_v.iitr/femnist/data/train'
    train_json_files = [f for f in os.listdir(train_path) if f.endswith('.json')]
    client_datasets_dict = {}

    train_user_ids = []

    for train_json in train_json_files:
        with open(os.path.join(train_path, train_json), 'r') as file:
            data = json.load(file)
            user_data = data['user_data']
            for user_id, user_info in user_data.items():
                train_user_ids.append(user_id)
                x = np.array(user_info['x'])
                x = x.reshape(-1, 1, 28, 28)
                y = np.array(user_info['y'])

                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y)

                client_datasets_dict[user_id] = TensorDataset(x, y)

    user_ids = list(client_datasets_dict.keys())
    selected_user_ids = random.sample(user_ids, num_clients)

    print('selected users:', selected_user_ids)

    client_datasets = []
    for user_id in selected_user_ids:
        client_datasets.append(client_datasets_dict[user_id])

    test_path = 'fedmix/femnist/data/test'
    # test_path = '/scratch/apoorva_v.iitr/femnist/data/test'
    test_json_files = [f for f in os.listdir(test_path) if f.endswith('.json')]
    test_x, test_y = [], []

    test_user_ids = []

    for test_json in test_json_files:
        with open(os.path.join(test_path, test_json), 'r') as file:
            data = json.load(file)
            user_data = data['user_data']
            for user_id, user_info in user_data.items():
                test_user_ids.append(user_id)
                if user_id in selected_user_ids:
                    x = user_info['x']
                    y = user_info['y']

                    test_x.extend(x)
                    test_y.extend(y)

    x = torch.tensor(np.array(test_x).reshape(-1, 1, 28, 28), dtype=torch.float32)
    y = torch.tensor(np.array(test_y))
    testset = TensorDataset(x, y)

    print('length of client datasets:', [len(c) for c in client_datasets])
    print('train set size:', sum([len(c) for c in client_datasets]))
    print('test set size:', len(testset))

    # print('train users:', train_user_ids)
    # print('test users:', test_user_ids)
    # print('common user ids:', set(test_user_ids).intersection(train_user_ids))

    return client_datasets, testset


def _mash_data(client_datasets, mash_batch_size, num_classes):
    """..."""
    mashed_data = []
    for client_dataset in client_datasets:
        mashed_image, mashed_label = [], []

        for i, (image, label) in enumerate(client_dataset):
            mashed_image.append(image)
            mashed_label.append(torch.tensor([label]))

            if (not mash_batch_size == "all") and (i + 1) % mash_batch_size == 0:
                mashed_data.append(
                    (
                        torch.squeeze(
                            torch.mean(
                                torch.stack(mashed_image[-mash_batch_size:]), dim=0
                            )
                        ),
                        torch.mean(
                            F.one_hot(
                                torch.squeeze(
                                    torch.stack(mashed_label[-mash_batch_size:])
                                ),
                                num_classes,
                            ).to(dtype=torch.float32),
                            dim=0,
                        ),
                    )
                )

        if mash_batch_size == "all":
            mashed_data.append(
                (
                    torch.squeeze(torch.mean(torch.stack(mashed_image), dim=0)),
                    torch.mean(
                        F.one_hot(
                            torch.squeeze(torch.stack(mashed_label)), num_classes
                        ).to(dtype=torch.float32),
                        dim=0,
                    ),
                )
            )

        mashed_image, mashed_label = [], []

    print('length of mashed data:', len(mashed_data))
    print('shapes of mashed data:', mashed_data[0][0].shape, mashed_data[0][1].shape)
    print('sample mashed label:', mashed_data[0][1])

    # save sample image
    transforms.ToPILImage()(mashed_data[0][0]).save('./_static/sample_mashed_image.png')

    return mashed_data
