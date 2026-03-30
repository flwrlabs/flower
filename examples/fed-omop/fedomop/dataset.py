from datasets import Dataset
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path

X_path = Path("./preprocess_MIMIC/data/output/cohort_icu_readmission_24_1_HF/X.csv")
Y_path = Path("./preprocess_MIMIC/data/output/cohort_icu_readmission_24_1_HF/Y.csv")

# 1) Load
X = pd.read_csv(X_path)
Y = pd.read_csv(Y_path)

assert len(X) == len(Y), "X and Y do not have the same number of rows"

# 2) Make sure y is a 1D array
y = Y.iloc[:, 0].to_numpy().astype(np.int64)

# 3) Encode categorical columns simply
cat_cols = X.select_dtypes(include=["object", "category"]).columns
X = pd.get_dummies(X, columns=cat_cols)
X_array = X.to_numpy(dtype=np.float32)

# 4) Build Hugging Face dataset
fds = Dataset.from_dict({
    "features": X_array,
    "label": y,
})

# 5) Split
fds = fds.train_test_split(test_size=0.3, seed=42)

# 6) Torch format
fds.set_format(type="torch", columns=["features", "label"])


def get_mimic_features():
    return fds["train"][0]["features"]

def load_global_data_mimic():
    test_fds = fds["test"]
    test_fds.set_format(type="torch", columns=["features", "label"])
    testloader = DataLoader(test_fds, 
                            batch_size=32, #Hardcoded
                            shuffle=False)
    return testloader

def load_local_data_mimic(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioner_strat="iid",
    dirichlet_alpha=None,
    seed=42,
):
    if partitioner_strat == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
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