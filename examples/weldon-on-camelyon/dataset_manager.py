import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/api/records/7053167/files-archive"


def fetch_camelyon(data_path: Path):
    files_archive = "files-archive"
    if not (data_path / files_archive).exists():
        print("Downloading the dataset (~400Mb), this may take a few minutes.")
        subprocess.run(
            ["wget", f"{URL}"],
            check=True,
            cwd=data_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        with zipfile.ZipFile(data_path / files_archive, "r") as zip_ref:
            zip_ref.extractall(data_path / "tiles_0.5mpp")


def reset_data_folder(data_path: Path):
    """Reset data folder to it's original state i.e. all the data in the img_dir_path
    folder."""
    # Deleting old experiment folders
    if data_path.is_dir():
        shutil.rmtree(data_path)


def creates_data_folder(img_dir_path, dest_folder):
    """Creates the `dest_folder` and hard link the data passed in index to it.

    Args:
        img_dir_path (Path): Folder with the data in npy format, and the associated
            index.csv file.
        dest_folder (Path): Folder to put the needed data.
        index (np.ndarray): A 2D array (file_name, target) of the data wanted in the dest
        folder.
    """
    index = np.loadtxt(img_dir_path / "index.csv", delimiter=",", dtype="<U32")[1:]

    dest_folder.mkdir(exist_ok=True, parents=True)

    for idx, sample_arr in enumerate(index):
        file_name = sample_arr[0] + ".tif.npy"
        out_name = f"{idx}_{file_name}"
        os.link(img_dir_path / file_name, dest_folder / out_name)
        index[idx, 0] = out_name

    np.savetxt(dest_folder / "index.csv", index, fmt="%s", delimiter=",")

    return dest_folder
