"""Prepare WEAR dataset to run TinyHAR model in Federated Learning."""

import os
import warnings
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


def get_learning_type_name(input_string: str) -> str:
    """Tranform names for plot names."""
    return " ".join(word.capitalize() for word in input_string.split("_"))


# Function to extract the numerical part of the file name:
def extract_number(file_path: str) -> int:
    """Get all CSV file paths in a sorted way."""
    base_name = os.path.basename(file_path)
    number = int(
        base_name.split("_")[1].split(".")[0]
    )  # Extract the number after 'sbj_' and convert it to an integer

    return number


def load_data(user_path: str, keep_NULL=True):
    """Load the data."""
    # Read csv file as a pandas dataframe from path:
    user_df = pd.read_csv(user_path)

    # Drop sbj_id column:
    user_df = user_df.drop("sbj_id", axis=1)

    # Define X and y columns:
    X_columns = [
        "RA_x",
        "RA_y",
        "RA_z",
        "RL_x",
        "RL_y",
        "RL_z",
        "LL_x",
        "LL_y",
        "LL_z",
        "LA_x",
        "LA_y",
        "LA_z",
    ]
    y_column = ["label"]

    # Rename columns:
    user_df.columns = X_columns + y_column

    if keep_NULL:
        # Rename NaN values with 'NULL' in label column:
        user_df[y_column] = user_df[y_column].fillna("NULL")

    # Remove missing values in data:
    user_df = user_df.dropna().reset_index(drop=True)

    # Transform standart scaling to X values:
    user_df[X_columns] = pd.DataFrame(
        StandardScaler().fit_transform(user_df[X_columns]), columns=X_columns
    )

    X_features, y_labels = user_df.iloc[:, :-1].values, user_df.iloc[:, -1].values

    return X_features, y_labels


def manual_data_split(
    X_features: NDArray[np.generic],
    y_labels: NDArray[np.generic],
    SPLIT_SIZE: float = 0.2,
) -> Tuple[
    NDArray[np.generic], NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]
]:
    """Split data manually."""
    # Initialize test arrays as lists:
    # y_test_label = []
    # X_test_features = []

    y_test_label: List[np.generic] = []
    X_test_features: List[NDArray[np.generic]] = []

    # Ensure inputs are NumPy arrays
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    # Iterate through y_labels and split the data:
    start_idx = 0
    while start_idx < len(y_labels):
        current_label = y_labels[start_idx]
        end_idx = start_idx

        # Find the end of the current label segment:
        while end_idx < len(y_labels) and y_labels[end_idx] == current_label:
            end_idx += 1

        # Determine the first 20% of the current segment:
        segment_length = end_idx - start_idx
        split_index = int(segment_length * SPLIT_SIZE)

        # Add the first 20% to the test sets:

        y_test_label.extend(y_labels[start_idx : start_idx + split_index])

        X_test_features.extend(X_features[start_idx : start_idx + split_index])

        # Remove the first 20% from the original arrays:
        y_labels = np.delete(y_labels, np.s_[start_idx : start_idx + split_index])
        X_features = np.delete(
            X_features, np.s_[start_idx : start_idx + split_index], axis=0
        )

        # Adjust the end_idx after deletion:
        start_idx = end_idx - split_index

    # Convert to numpy arrays:
    # y_test_label = np.array(y_test_label)
    # To (with type hint):
    y_test_label_np: NDArray[np.generic] = np.array(y_test_label)
    # X_test_features = np.array(X_test_features)
    X_test_features_np: NDArray[np.generic] = np.array(X_test_features)

    return X_features, y_labels, X_test_features_np, y_test_label_np


def take_most_common_label_in_a_window(
    arr: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    """Take most common label."""
    max_index = (arr.size(0) // sequence_length) * sequence_length

    most_common_values: List[int] = []

    # Loop through 50-element chunks up to max_index:
    for i in range(0, max_index, sequence_length):
        chunk = arr[i : i + sequence_length]  # Get 50-element chunk

        # Find the most common element in the chunk
        most_common = Counter(chunk.tolist()).most_common(1)
        most_common_value = most_common[0][0]  # Extract most common value
        most_common_values.append(most_common_value)

    # Convert the list of most common values back into a PyTorch tensor:
    most_common_tensor = torch.tensor(most_common_values)
    return most_common_tensor


def generate_dataloaders(
    train_features: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_features: NDArray[np.float32],
    test_labels: NDArray[np.int64],
    batch_size=32,
    sequence_length=50,
) -> Tuple[DataLoader, DataLoader]:
    """Generate dataloaders."""
    # ############################# Convert lists to torch tensors #################
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    n_train_samples = (
        train_features_tensor.shape[0] // sequence_length
    ) * sequence_length
    n_test_samples = (
        test_features_tensor.shape[0] // sequence_length
    ) * sequence_length
    # Here, removing step happens:
    train_features_tensor = train_features_tensor[:n_train_samples]
    test_features_tensor = test_features_tensor[:n_test_samples]
    # Reshape data to match the input shape expected by TinyHAR:
    # Now number of training data is 202250, sequence_length is 50, 12 features.
    # With the step below, shape of data will be
    # torch.Size([2045, 1, 50, 12])  2045 = 202250 / 50
    # So, there are 2045 windows with the size of 50.
    # In other words, there are 2045 matrix with 50 rows, 12 columns, depth is 1.
    train_features_tensor = train_features_tensor.reshape(
        -1, 1, sequence_length, train_features_tensor.shape[1]
    )
    test_features_tensor = test_features_tensor.reshape(
        -1, 1, sequence_length, test_features_tensor.shape[1]
    )

    # ############################### Preparing Label ############################
    # Trim labels accordingly. Take each windows and from
    # each windows select the most common label:
    train_labels_tensor = take_most_common_label_in_a_window(
        train_labels_tensor, sequence_length
    )
    test_labels_tensor = take_most_common_label_in_a_window(
        test_labels_tensor, sequence_length
    )

    # ############################### Create TensorDatasets ##########################
    trainset = TensorDataset(train_features_tensor, train_labels_tensor)
    testset = TensorDataset(test_features_tensor, test_labels_tensor)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    # testloader = DataLoader(testset, batch_size=1, shuffle=False)

    return trainloader, testloader


def add_pred_to_all_data(user_labels, user_test_labels, user_pred_labels: list) -> list:
    """Replace testing labels with predictions in pure user_labels."""
    final_array = []

    num = 0
    for i, label in enumerate(user_labels):

        if num < len(user_test_labels) and label == user_test_labels[num]:
            final_array.append(user_pred_labels[num])
            num += 1

        else:
            final_array.append(user_labels[i])

    return final_array


class training_functions:
    """Tain class."""

    def train_model(model, trainloader, rnd, opt, learning_rate, device):
        """Tarin model."""
        model.to(device)  # correction 1
        criterion = nn.CrossEntropyLoss()

        if opt == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif opt == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            print("Optimizer name error!")

        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for _, (features, labels) in enumerate(trainloader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainloader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {rnd} -> Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    def train_model_federated(
        model, trainloader, num_epochs, opt, learning_rate, device
    ):
        """Train federated model."""
        model.to(device)  # correction 1
        criterion = nn.CrossEntropyLoss()

        if opt == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif opt == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            print("Optimizer name error!")

        for _ in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for _, (features, labels) in enumerate(trainloader):
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()  # Zero the parameter gradients

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Compute statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(trainloader)
            accuracy = correct_predictions / total_predictions

            return epoch_loss, accuracy


class evaluation_functions:
    """Evaluationn function for client."""

    def evaluate_model(model, testloader, result_print=True, device=None, cfg=None):
        """Evaluate the model performance."""
        model.to(device)  # correction 1

        model.eval()  # Set the model to evaluation mode
        criterion = nn.CrossEntropyLoss()

        total = 0
        correct = 0
        test_loss = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():  # Disable gradient computation for evaluation
            for _, (features, labels) in enumerate(testloader):
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate loss and accuracy
        loss = test_loss / len(testloader)
        accuracy = 100 * correct / total

        # Calculate precision, recall, and F1-score
        precision, recall, fscore, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro"
        )

        if result_print:
            print(
                f"{'*'*10} Client ID {cfg.sub_id} {'*'*10}\n"
                f"Distributed Loss: {loss:.4f}\n"
                f"Distributed Accuracy: {accuracy:.2f}%\n"
                f"Distributed Precision: {precision:.4f}\n"
                f"Distributed Recall: {recall:.4f}\n"
                f"Distributed F1-Score: {fscore:.4f}\n"
                f"{'*'*34}\n"
            )

        return loss, accuracy, precision, recall, fscore

    def get_ground_truth_and_predictions(model, testloader, DEVICE):
        """Get the ground truth data for predictions and comparing the results."""
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode
        criterion = nn.CrossEntropyLoss()
        total = 0
        correct = 0
        test_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():  # Disable gradient computation for evaluation
            for _, (features, labels) in enumerate(testloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        return all_labels, all_predictions

    def get_label_based_results(all_labels, all_predictions, reversed_labels_set):
        """Get the label based results for comparising the results."""
        # Convert labels and predictions to tensors
        all_labels_tensor = torch.tensor(all_labels)
        all_predictions_tensor = torch.tensor(all_predictions)

        # Calculate per-label metrics using classification_report
        res = classification_report(
            all_labels, all_predictions, zero_division=0, output_dict=True
        )

        res_with_label_names = {}

        # Calculate Cross Entropy Loss and accuracy for each label
        for numeric_label, label_name in reversed_labels_set.items():
            if (
                str(numeric_label) in res
            ):  # Check if the label exists in classification report
                # Filter predictions and labels for the current label
                mask = all_labels_tensor == numeric_label
                label_predictions = all_predictions_tensor[mask]
                label_labels = all_labels_tensor[mask]

                # Calculate accuracy for the current label
                label_accuracy = (
                    (label_predictions == label_labels).float().mean().item()
                    if label_labels.size(0) > 0
                    else 0.0
                )

                # Calculate loss for the current label if there are predictions
                if label_labels.size(0) > 0:
                    label_logits = F.one_hot(
                        label_predictions, num_classes=len(reversed_labels_set)
                    ).float()
                    label_loss = F.cross_entropy(label_logits, label_labels).item()
                else:
                    label_loss = float("nan")  # No samples for this label

                # Add metrics to the report for this label
                res_with_label_names[label_name] = res[str(numeric_label)]
                res_with_label_names[label_name]["accuracy"] = label_accuracy
                res_with_label_names[label_name]["loss"] = label_loss
            else:
                # Initialize metrics for labels not present in predictions or GT
                res_with_label_names[label_name] = {
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1-score": float("nan"),
                    "support": 0.0,
                    "accuracy": 0.0,
                    "loss": float("nan"),
                }

        # Add any other overall metrics, like 'accuracy' or 'macro avg'
        for key, value in res.items():
            if not key.isdigit():  # Skip numeric keys
                res_with_label_names[key] = value

        return res_with_label_names
