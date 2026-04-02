from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, NaturalIdPartitioner
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path



HF_DS = {
    "synthea_small": "danimanjah/synthea_small",
}

fds = None
scaler = StandardScaler() # /!\ valid only on simulation, otherwise a federated scaling should be conducted


def fit_scaler():
    global fds, scaler
    train_fds = fds["train"]
    X_train = np.asarray(train_fds["features"], dtype=np.float32)
    scaler.fit(X_train)


def cache_local_ds(path_prefix:str):
    global fds
    # 1) Load
    X = pd.read_csv(f"{path_prefix}/X.csv")
    Y = pd.read_csv(f"{path_prefix}/Y.csv")

    assert len(X) == len(Y), "X and Y do not have the same number of rows"

    # 2) Make sure y is a 1D array
    y = Y.iloc[:, 0].to_numpy().astype(np.int64)

    # 3) Encode categorical columns simply
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=cat_cols)
    X_array = X.to_numpy(dtype=np.float32)

    # 4) Build HF dataset
    fds = Dataset.from_dict({
        "features": X_array,
        "label": y,
    })

    # 5) Split
    fds = fds.train_test_split(test_size=0.3, seed=42)

    # 6) Torch format
    fds.set_format(type="torch", columns=["features", "label"])


def instantiate_ds_and_get_features(dataset=None, local_path_prefix = None):
    global fds
    if fds is None: 
        if local_path_prefix is not None:
            cache_local_ds(local_path_prefix)
            fit_scaler()
        else:
            fds = load_dataset(HF_DS[dataset],"all")
            fit_scaler()
    return fds["train"][0]["features"]

def load_global_data():

    global fds, scaler
    test_fds = fds["test"]
    
    X_test = np.asarray(test_fds["features"], dtype=np.float32)
    y_test = np.asarray(test_fds["label"], dtype=np.int64)

    
    # ---- local standardization ----

    X_test = scaler.transform(X_test).astype(np.float32)

    test_fds = Dataset.from_dict({
        "features": X_test,
        "label": y_test,
    })

    test_fds.set_format(type="torch", columns=["features", "label"])
    testloader = DataLoader(test_fds, 
                            batch_size=32, #Hardcoded
                            shuffle=False)
    return testloader

def load_local_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioner_strat="iid",
    dirichlet_alpha=None,
    seed=42,
):
    
    if partitioner_strat == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    elif partitioner_strat =="natural":
           partitioner = NaturalIdPartitioner(partition_by="hospital_id")    
    elif partitioner_strat == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=dirichlet_alpha,
            min_partition_size=500,
            self_balancing=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partitioner_strat: {partitioner_strat}")

    partitioner.dataset = fds["train"]
    if  num_partitions > partitioner.num_partitions:
        raise ValueError(
                    f"Requested num_partitions={num_partitions}, "
                    f"but only {partitioner.num_partitions} natural partitions exist "
                    f"in fds['train'] based on hospital_id."
                )

    client_dataset = partitioner.load_partition(partition_id)

    # split local partition
    partition_train_val = client_dataset.train_test_split(test_size=0.2, seed=seed)
    train_ds = partition_train_val["train"]
    val_ds = partition_train_val["test"]

    # ---- local standardization ----
    X_train = np.asarray(train_ds["features"], dtype=np.float32)
    y_train = np.asarray(train_ds["label"], dtype=np.int64)

    X_val = np.asarray(val_ds["features"], dtype=np.float32)
    y_val = np.asarray(val_ds["label"], dtype=np.int64)

    
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = Dataset.from_dict({
        "features": X_train,
        "label": y_train,
    })
    val_ds = Dataset.from_dict({
        "features": X_val,
        "label": y_val,
    })

    train_ds.set_format(type="torch", columns=["features", "label"])
    val_ds.set_format(type="torch", columns=["features", "label"])

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return trainloader, valloader