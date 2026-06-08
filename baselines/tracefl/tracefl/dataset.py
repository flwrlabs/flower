"""Handle basic dataset creation.

This module provides dataset loading functionality for TraceFL baseline, supporting
multiple datasets without caching.
"""

import logging

from tracefl.dataset_preparation import ClientsAndServerDatasets


def get_clients_server_data(cfg):
    """Obtain datasets for clients and server based on the provided configuration.

    This function creates the dataset using ClientsAndServerDatasets and returns it.
    No caching is used - everything is kept in memory.

    Parameters
    ----------
    cfg : object
        A configuration object that must contain attributes for data
        distribution parameters.

    Returns
    -------
    dict
        A dictionary containing the datasets for both clients and server.
    """
    # Direct dataset creation - no caching
    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_data()

    faulty_clients = getattr(cfg, "faulty_clients_ids", []) or []
    label2flip = getattr(cfg, "label2flip", {}) or {}

    if faulty_clients and label2flip:
        logging.info(
            "Converting faulty clients %s using label map %s",
            faulty_clients,
            label2flip,
        )
        updated_client2class = dict(ds_dict.get("client2class", {}))

        for faulty_id in faulty_clients:
            client_key = str(faulty_id)
            client_dataset = ds_dict["client2data"].get(client_key)
            if client_dataset is None:
                logging.warning(
                    "Requested faulty client %s not found in dataset "
                    "partition; skipping conversion.",
                    client_key,
                )
                continue

            converted = convert_client2_faulty_client(client_dataset, label2flip)
            ds_dict["client2data"][client_key] = converted["ds"]
            updated_client2class[client_key] = {
                str(label): int(count)
                for label, count in converted["label2count"].items()
            }

        ds_dict["client2class"] = updated_client2class

    logging.info("Dataset created successfully for %s", cfg.data_dist.dname)

    return ds_dict


def load_central_server_test_data(cfg):
    """Load the test data intended for the central server.

    Parameters
    ----------
    cfg : object
        A configuration object required to initialize the dataset creation process.

    Returns
    -------
    object
        The test dataset for the central server.
    """
    d_obj = ClientsAndServerDatasets(cfg).get_data()
    return d_obj["server_testdata"]


def convert_client2_faulty_client(ds, label2flip, target_label_col="label"):
    """Transform a client's dataset to simulate a faulty client by flipping specified
    labels.

    Parameters
    ----------
    ds : Dataset
        The dataset to be modified.
    label2flip : dict
        A dictionary mapping original label values to the flipped label values.
    target_label_col : str, optional
        The key in each example dict corresponding to the target label.
        Default is 'label'.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'ds': The transformed dataset with flipped labels.
            - 'label2count': A dictionary mapping each label to the count of its
              occurrences in the transformed dataset.
    """

    def flip_label(example):
        label = None
        try:
            label = example[target_label_col].item()
        except (AttributeError, KeyError):
            label = example[target_label_col]
        if label in label2flip:
            example[target_label_col] = label2flip[label]
        return example

    ds = ds.map(flip_label).with_format("torch")
    label2count = {}

    for example in ds:
        label = example[target_label_col].item()
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1

    return {"ds": ds, "label2count": label2count}
