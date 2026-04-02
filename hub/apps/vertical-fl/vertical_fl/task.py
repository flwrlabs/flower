import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import VerticalSizePartitioner
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Fare",
    "Siblings/Spouses Aboard",
    "Name",
    "Parents/Children Aboard",
    "Pclass",
]


def load_and_preprocess(
    dataframe: pd.DataFrame,
):
    """Preprocess a subset of the titanic-survival dataset columns into a purely
    numerical numpy array suitable for model training."""

    # Make a copy to avoid modifying the original
    X_df = dataframe.copy()

    # Identify which columns are present
    available_cols = set(X_df.columns)

    # ----------------------------------------------------------------------
    # FEATURE ENGINEERING ON NAME (if present)
    # ----------------------------------------------------------------------
    if "Name" in available_cols:
        X_df["Title"] = X_df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        X_df["NameLength"] = X_df["Name"].str.len()
        X_df = X_df.drop(columns=["Name"])

    # ----------------------------------------------------------------------
    # IDENTIFY NUMERIC + CATEGORICAL COLUMNS
    # ----------------------------------------------------------------------
    categorical_cols = []
    if "Sex" in X_df.columns:
        categorical_cols.append("Sex")
    if "Title" in X_df.columns:
        categorical_cols.append("Title")
    if "Pclass" in X_df.columns:
        categorical_cols.append("Pclass")

    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    # ----------------------------------------------------------------------
    # HANDLE MISSING VALUES
    # ----------------------------------------------------------------------
    if numeric_cols:
        X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].median())

    # ----------------------------------------------------------------------
    # PREPROCESSOR (TRANSFORM TO PURE NUMERIC)
    # ----------------------------------------------------------------------
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # ----------------------------------------------------------------------
    # FIT TRANSFORMER & CONVERT TO NUMPY
    # ----------------------------------------------------------------------
    X_full = preprocessor.fit_transform(X_df)

    # Ensure output is always a dense numpy array
    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()

    return X_full.astype(np.float32)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, feature_splits: list[int]):
    """..."""

    global fds
    if fds is None:
        partitioner = VerticalSizePartitioner(
            partition_sizes=feature_splits,
            active_party_columns="Survived",
            active_party_columns_mode="create_as_last",
        )

        fds = FederatedDataset(
            dataset="julien-c/titanic-survival", partitioners={"train": partitioner}
        )

    # Load partition
    partition = fds.load_partition(partition_id)

    # Process partition
    return load_and_preprocess(dataframe=partition.to_pandas())


class ClientModel(nn.Module):
    def __init__(self, input_size, out_feat_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, out_feat_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return self.fc2(x)


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.hidden = nn.Linear(input_size, 96)
        self.fc = nn.Linear(96, 1)
        self.bn = nn.BatchNorm1d(96)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.bn(x)
        x = self.fc(x)
        return self.sigmoid(x)


def evaluate_head_model(
    head: ServerModel, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute accuracy of head."""
    head.eval()
    with torch.no_grad():
        correct = 0
        # Re-compute embeddings for accuracy (detached from grad)
        embeddings_eval = embeddings.detach()
        output = head(embeddings_eval)
        predicted = (output > 0.5).float()
        correct += (predicted == labels).sum().item()
        accuracy = correct / len(labels) * 100

    return accuracy
