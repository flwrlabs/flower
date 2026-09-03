"""Prepare the CICIDS2018 dataset for federated learning."""

import os
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


FDS=False
clientsets = None
trainloaders=[]
valloaders=[]
num_classes=[]
num_features=1

def get_client_logger():
    logger = logging.getLogger(f"Client_{partition_id}")
    if not logger.hasHandlers():
        os.makedirs("logs", exist_ok=True)
        handler = logging.FileHandler(f"logs/Clients_data_info.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def load_dataset():
    """
    Loads and cleans the dataset from the given directory.

    - Replaces infinities with NaN
    - Drops rows with NaN values
    - Returns a cleaned pandas DataFrame
    """
    global FDS
    FDS=True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "dataset.csv")
    dataset = pd.read_csv(dataset_path)
    return dataset
# ----------------------------
# Select specific features + encode labels
# ----------------------------
def pre_process_dataset(dataset: pd.DataFrame):
    """
    Select relevant features and convert string labels to numeric class indices.
    """
    selected_columns = [
        'Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Bwd Pkt Len Mean',
        'Bwd Pkt Len Std', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Max', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
        'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Var', 'Bwd Seg Size Avg',
        'Subflow Fwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
        'Fwd Seg Size Min', 'Label'
    ]
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    data = dataset.loc[:, selected_columns]
    data.dropna()

    class2idx = {
        "Bot": 0, "Benign": 1, "Infilteration": 2, "DoS attacks-GoldenEye": 3,
        "DoS attacks-Hulk": 4, "DoS attacks-SlowHTTPTest": 5, "DDOS attack-HOIC": 6,
        "DoS attacks-Slowloris": 7, "SSH-Bruteforce": 8, "DDOS attack-LOIC-UDP": 9,
        "Brute Force -Web": 10, "FTP-BruteForce": 11, "Brute Force -XSS": 12,
        "SQL Injection": 13, "DDoS attacks-LOIC-HTTP": 14
    }

    # Replace labels with integers
    data['Label'].replace(class2idx, inplace=True)
    data['Label'] = data['Label'].astype('float')

    return data

# ----------------------------
# Filter out labels manually or randomly
# ----------------------------
def filter_training_labels(partition_id:int, train_df: pd.DataFrame, drop_labels: list = None,
                           random_drop: bool = True, min_random_drop: int = 1, max_drop: int=1, logger=None):
    """
    Filters out specific or randomly selected labels from the training data.
    """
   
    if 'Label' not in train_df.columns:
        raise ValueError("DataFrame must contain a 'Label' column.")
    
    if random_drop:
        all_labels = train_df['Label'].unique().tolist()
        num = random.randint(min_random_drop, max_drop)
        drop_labels = random.sample(all_labels, min(num, len(all_labels) - 1))
    elif drop_labels is None:
         drop_labels = []
   
    logger.info(f"Client {partition_id} dropped classes: {drop_labels}")
    filtered_df = train_df[~train_df['Label'].isin(drop_labels)].reset_index(drop=True)
    num_classes = filtered_df['Label'].nunique()
    logger.info(f"Client {partition_id} existing classes after drop: {num_classes}")
    return filtered_df, drop_labels, num_classes

# ----------------------------
# Normalize features & create PyTorch Datasets
# ----------------------------
def scale_datasets(train, test, y_col='Label'):
    """
    Scales feature values using MinMaxScaler and wraps them into PyTorch Datasets.
    """
    X_train, y_train = train.drop(columns=[y_col]), train[y_col]
    X_val, y_val = test.drop(columns=[y_col]), test[y_col]
    num_features = len(X_train.columns)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    class ClassifierDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    return train_dataset, val_dataset, num_features

# ----------------------------
# Main data preparation function
# ----------------------------


    
    
def prepare_dataset(partition_id: int, num_partitions: int,  max_drop:int, total_classes:int):
    """
    Loads, processes, splits, and partitions CICIDS2018 dataset into num_partitions federated clients  and one partition for the server (the last partition).

    Returns:
        trainloader (List[DataLoader]): Training data per client.
        valloader (List[DataLoader]): Validation data per client.
    """
      # Only initialize `FederatedDataset` once
     # pylint: disable=global-statement
    global num_features 
    global FDS
    global clientsets
    global trainloaders
    global valloaders
    global num_classes
    #Loading and preprocessing dataset only once in this process.
    
    if FDS is False: 
        logger=get_client_logger()
        dataset= load_dataset()        
        dataset = pre_process_dataset(dataset)
        rows_per_client = len(dataset) // num_partitions
        clientsets =[]
        # Partition the dataset equally among clients
        for _ in range(num_partitions - 1):
           client_df, dataset = train_test_split( dataset, train_size=rows_per_client, random_state=20, stratify=dataset ['Label'] )
           clientsets.append(client_df.reset_index(drop=True))
        # Last client gets the remaining rows (including leftovers)
        clientsets.append(dataset.reset_index(drop=True))
        for idx, trainset_ in enumerate(clientsets):
            client_train, client_val = train_test_split(
            trainset_,
            test_size=0.2,
            random_state=2023,
            stratify=trainset_['Label']
            )
            if trainset_ is clientsets[-1]:
         # Special behavior for the server
              client_train, dropped, num_classes_cl = filter_training_labels(9,client_train, random_drop=False, max_drop=max_drop, logger=logger )
            else:     
              client_train, dropped, num_classes_cl = filter_training_labels(idx,client_train, random_drop=True, max_drop=max_drop, logger=logger )
       
            train_dataset, val_dataset, num_features = scale_datasets(client_train, client_val)
            trainloaders.append(DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0))
            valloaders.append(DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0))
            num_classes.append(num_classes_cl) 
    
    return trainloaders[partition_id], valloaders[partition_id], num_classes[partition_id], num_features
