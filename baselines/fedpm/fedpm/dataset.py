"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from fedpm.dataset_preparation import *


def get_data_loaders(
    dataset, nclients, batch_size, classes_pc=10, split="iid", data_path=None
):
    transforms_train, transforms_eval = get_default_data_transforms(
        dataset=dataset, verbose=True
    )
    # This is to evaluate models at the devices
    client_val_loader = [[] for _ in range(nclients)]
    data_path = Path(data_path).joinpath(dataset)
    if split == "iid":
        data_train, data_test = dataset_dict[dataset](
            transform=transforms_train, iid=True, data_path=data_path
        )
        data_train_list = split_dataset(dataset=data_train, n_clients=nclients)
        data_loader_client_list = [
            DataLoader(local_data, batch_size=batch_size, shuffle=True)
            for local_data in data_train_list
        ]
        data_loader_test = DataLoader(
            data_test, batch_size=len(data_test), shuffle=True
        )
        return data_loader_client_list, client_val_loader, data_loader_test
    else:
        # Get data
        x_train, y_train, x_test, y_test = dataset_dict[dataset](
            iid=False, data_path=filename
        )
        split = split_image_data(
            x_train, y_train, n_clients=nclients, classes_per_client=classes_pc
        )

        split_tmp = shuffle_list(split)

        client_loaders = [
            torch.utils.data.DataLoader(
                CustomImageDataset(x, y, transforms_train),
                batch_size=batch_size,
                shuffle=True,
            )
            for x, y in split_tmp
        ]

        test_loader = torch.utils.data.DataLoader(
            CustomImageDataset(x_test, y_test, transforms_eval),
            batch_size=len(x_test),
            shuffle=False,
        )

    return client_loaders, client_val_loader, test_loader
