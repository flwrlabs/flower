"""Client utils funnctions."""

import glob
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
from omegaconf import DictConfig, OmegaConf

from fedfittech.flwr_utils.utils_for_tinyhar import (
    extract_number,
    generate_dataloaders,
    load_data,
    manual_data_split,
)

from .TinyHAR import TinyHAR

def download_data_for_client():
    """Load data for clients from WEAR repository."""
    DOWNLOAD_URL = (f"https://ubi29.informatik.uni-siegen.de/"
                    f"wear_dataset/raw/inertial/50hz/")
    # Download the dataset if not already downloaded
    if not os.path.exists("./inertial_data/"):
        print("Downloading inertial data from WEAR repository...")
        os.makedirs("./inertial_data/", exist_ok=True)
        os.system(f"wget -P ./inertial_data/ {DOWNLOAD_URL}*.csv") 
    # PATH = os.path.join(os.getcwd(), "./inertial_data/")


def load_data_for_client(cfg, user_num):
    """Load data for cliets."""
    PATH = os.path.join(os.getcwd(), "./inertial_data/")

    # Get the list of CSV file paths in order:s
    csv_file_paths = sorted(glob.glob(os.path.join(PATH, "*.csv")), key=extract_number)

    for path in csv_file_paths:
        sub = int(path.split("/")[-1].split("_")[1].split(".")[0])
        if sub == user_num:
            current_user_path = path
            cfg.sub_id = sub
        if sub >= len(
            csv_file_paths
        ):  # as sub_0 is included so sub_0 to sub_23, len 24
            print(f"Subject {sub} is not in data")
    # Number to labels:
    reversed_labels_set = {v: k for k, v in cfg.labels_set.items()}
    cfg.reversed_labels_set = reversed_labels_set

    # Data cleaning and Processing
    user_real_train_labels_all, user_real_test_labels_all = [], []
    (
        user_train_features_all,
        user_train_labels_all,
        user_test_features_all,
        user_test_labels_all,
    ) = ([], [], [], [])
    training_dataloader_all, testing_dataloader_all = [], []

    X_user_features, y_user_labels = load_data(current_user_path)

    user_total_class = len(np.unique(y_user_labels))
    if user_total_class < cfg.NUM_CLASS:
        print(f"Sbj_{user_num} has {user_total_class} classes.")

    (
        X_user_train_features,
        y_user_train_labels,
        X_user_test_features,
        y_user_test_labels,
    ) = manual_data_split(X_user_features, y_user_labels)

    # Save real label values for analysis:
    user_real_train_labels_all.append(y_user_train_labels)
    user_real_test_labels_all.append(y_user_test_labels)

    # Transform the labels to integer labels
    y_user_train_labels = [cfg.labels_set[label] for label in y_user_train_labels]
    y_user_test_labels = [cfg.labels_set[label] for label in y_user_test_labels]

    user_train_features_all.append(X_user_train_features)
    user_train_labels_all.append(y_user_train_labels)
    user_test_features_all.append(X_user_test_features)
    user_test_labels_all.append(y_user_test_labels)

    # Load data with sequence length of 100 as a dataloader:
    trainloader, testloader = generate_dataloaders(
        user_train_features_all[0],
        user_train_labels_all[0],
        user_test_features_all[0],
        user_test_labels_all[0],
        batch_size=cfg.BATCH_SIZE,
        sequence_length=cfg.WINDOW_SIZE,
    )
    training_dataloader_all.append(trainloader)
    testing_dataloader_all.append(testloader)
    cfg.num_train_examples = len(X_user_train_features)
    cfg.num_test_examples = len(X_user_test_features)
    return training_dataloader_all, testing_dataloader_all, cfg


def get_net_and_config():
    """Get the model and config hyperparameters."""
    config_path = "./config"
    config_file_name = "base.yaml"
    cfg = load_config(config_path=config_path, config_file=config_file_name)

    preference_for_NULL = cfg.preference_for_NULL

    if preference_for_NULL in ["False", "no", "0"]:
        del cfg.labels_set["NULL"]

    elif preference_for_NULL not in ["True", "yes", "1", "False", "No", "0"]:
        raise ValueError("Invalid input. Please enter True, False, Yes, No, 1, or 0.")

    cfg.NUM_CLASS = len(cfg.labels_set)  # 19
    cfg.NUM_FEATURES = cfg.NUM_FEATURES  # 3 * 4  # ((x, y, z) * (RA, RL, LA, LL))

    net = TinyHAR(
        input_shape=(cfg.BATCH_SIZE, 1, cfg.WINDOW_SIZE, cfg.NUM_FEATURES),
        number_class=cfg.NUM_CLASS,
        filter_num=cfg.NUM_FILTERS,
    )

    return net, cfg


def load_config(config_path: str, config_file: str) -> DictConfig:
    """Load YAML configuration file using OmegaConf.

    Args:
        config_path (str): Path to the configuration directory.
        config_file (str): Name of the configuration file.

    Returns
    -------
        DictConfig: Loaded configuration.
    """
    # Construct the full path to the config file
    config_file = f"{config_path}/{config_file}"

    # Load the configuration
    config = OmegaConf.load(config_file)
    # Runtime type check to ensure it's DictConfig
    if not isinstance(config, DictConfig):
        raise TypeError("Expected a DictConfig but got a ListConfig or other type.")

    # Disable struct mode if you want to modify the config dynamically
    OmegaConf.set_struct(config, False)

    return config


def get_model_plot_directory(
    plt_dir=True, model_dir=True, csv_dir=True, config=None, server_round="_"
):
    """Get the directories for saving the model, results."""
    server_round = str(server_round)
    # Generate directory name with current date and time
    file_date = datetime.now().strftime("%d-%m-%Y_%H-%M")
    plt_directory_name = None
    model_directory_name = None
    csv_directory_name = None

    root_log_path = os.path.join("./Flower_log", f"Experiment_One_Logs_{file_date}")
    os.makedirs(root_log_path, exist_ok=True)
    file_path = os.path.join(root_log_path, "hyperparameters.json")

    # Save the configuration file
    OmegaConf.save(config=config, f=file_path)

    if plt_dir:
        plt_directory_name = os.path.join(root_log_path, "Experiment_one_Plots")
        os.makedirs(
            plt_directory_name, exist_ok=True
        )  # Creates directory if it doesn't exist
    if model_dir:
        model_directory_name = os.path.join(root_log_path, "Experiment_one_Models")
        os.makedirs(model_directory_name, exist_ok=True)
    if csv_dir:
        csv_directory_name = os.path.join(root_log_path, "Experiment_Result_CSVs")
        os.makedirs(csv_directory_name, exist_ok=True)

    return plt_directory_name, model_directory_name, csv_directory_name, root_log_path



def download_all_inertial_data(cfg):
    """Download inertial dataset."""
    base_url = cfg.Data_download_path
    data_dir = "./inertial_data/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        # Get directory listing
        print("Fetching file list...")
        response = requests.get(base_url)
        if response.status_code != 200:
            print(f"Failed to access inertial data URL: HTTP {response.status_code}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')

        # Extract all .csv filenames
        csv_files = [link.get('href') for link in links if link.get('href', '').endswith('.csv')]

        if not csv_files:
            print("No CSV files found.")
            return

        print(f"Found {len(csv_files)} CSV files. Starting download...")

        os.makedirs(data_dir, exist_ok=True)

        for filename in csv_files:
            file_url = base_url + filename
            file_path = os.path.join(data_dir, filename)

            print(f"Downloading {filename}...")
            file_response = requests.get(file_url, timeout=10)
            if file_response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
            else:
                print(f"Failed to download {filename} (HTTP {file_response.status_code})")

        print("Download complete.")
    elif os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' already exists. Skipping download.")    



