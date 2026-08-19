"""Module to preprocess the NIST Special Database 19 into FeMNIST/FEMNIST."""

import pathlib
from logging import DEBUG, WARN
from typing import Dict, Union

import pandas as pd
from flwr.common.logger import log
from PIL import Image
from tqdm import tqdm

from flwr_baselines.publications.leaf.femnist.dataset.utils import (
    calculate_series_hashes,
    hex_decimal_to_char,
)


# pylint: disable=too-many-instance-attributes
class NISTPreprocessor:
    """Preprocess NIST images from two directories divided by class and by
    writer.

    The preprocessing procedure include different step for FEMNIST and
    FeMNIST.

    Using two datasets (by_class and by_write, actually one dataset but
    divided differently) is required to merge information about the
    writer_id and label.
    """

    def __init__(
        self, data_dir: Union[str, pathlib.Path], to_dataset: str = "FeMNIST"
    ) -> None:
        self._data_dir = (
            data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        )
        self._raw_data_dir = self._data_dir / "raw"
        self._processed_dir = self._data_dir / ("processed" + "_" + to_dataset)
        self._processed_images_dir = self._processed_dir / "images"
        self._processed_images_information_path = (
            self._processed_dir / "processed_images_to_labels.csv"
        )
        self._raw_images_information_path = self._data_dir / "raw_images_to_labels.csv"
        self._by_class_nist = self._raw_data_dir / "by_class"
        self._by_writer_nist = self._raw_data_dir / "by_write"
        self._writer_df: pd.DataFrame = pd.DataFrame()
        self._class_df: pd.DataFrame = pd.DataFrame()
        self._df: pd.DataFrame = pd.DataFrame()
        self._preprocessed_df: pd.DataFrame = pd.DataFrame()

    def preprocess(self, overwrite: bool = False) -> None:
        """Extracts necessary information to create data that has both writer
        and class information and preprocesses the dataset as by the authors of
        the FEMNIST paper (which is not the same as by the EMNIST paper).

        1. Extract writer_id from the directory structure.
        2. Extracts class_id from the directory structure.
        3. Calculate hashes from the images to enable merging data.
        4. Merge information based on hash values.
        5. Preprocess images (reduce the size of them, use LANCZOS resampling).
        6. Create csv file with the path.
        """
        # Do not preprocess if the preprocessed information already exist
        if self._processed_images_information_path.exists() and not overwrite:
            log(
                WARN,
                "The preprocessed information already exists in %s. "
                "It's assumed that the preprocessed images exist too."
                "Specify 'overwrite' as True to preprocess the images and recreate the reference "
                "information.",
                self._processed_images_information_path,
            )
            return

        # Extraction th raw images information if it does not already exist
        if not self._raw_images_information_path.exists():
            self._writer_df = self._extract_writer_information()
            self._class_df = self._extract_class_information()
            self._calculate_hashes()
            self._df = self._merge_class_and_writer_information()
            log(
                DEBUG,
                "Saving information about raw images to %s started",
                self._raw_images_information_path,
            )
            self._df.to_csv(self._raw_images_information_path)
            log(
                DEBUG,
                "Saving information about raw images to %s done",
                self._raw_images_information_path,
            )
        else:
            self._df = pd.read_csv(self._raw_images_information_path, index_col=0)
        self.create_dir_structure()
        self._preprocessed_df = self._preprocess_images()
        log(
            DEBUG,
            "Saving information about raw images to %s started",
            self._processed_images_information_path,
        )
        self._preprocessed_df.to_csv(self._processed_images_information_path)
        log(
            DEBUG,
            "Saving information about raw images to %s done",
            self._processed_images_information_path,
        )

    def create_dir_structure(self) -> None:
        """Create directory structure required for the dataset storage."""
        log(DEBUG, "Directory structure creation started")
        self._processed_dir.mkdir(exist_ok=True)
        log(DEBUG, "Directory %s got created/already existed", self._processed_dir)
        self._processed_images_dir.mkdir(exist_ok=True)
        log(
            DEBUG,
            "Directory %s got created/already existed",
            self._processed_images_dir,
        )
        log(DEBUG, "Directory structure creation done")

    def _extract_writer_information(self) -> pd.DataFrame:
        """Extract writer id based on the path (directories) it was placed
        in."""
        log(DEBUG, "Writer information preprocessing started")
        images_paths = list(self._by_writer_nist.glob("*/*/*/*"))
        writer_df = pd.DataFrame(images_paths, columns=["path_by_writer"])
        writer_df["writer_id"] = writer_df["path_by_writer"].map(
            lambda x: x.parent.parent.name
        )
        log(DEBUG, "The created writer_df has shape %s", str(writer_df.shape))
        log(DEBUG, "Writer information preprocessing done")
        return writer_df

    def _extract_class_information(self) -> pd.DataFrame:
        """Extract class (label) based on the path (directories) it was placed
        in.

        It also transforms hexadecimal ascii information into readable
        character.
        """
        log(DEBUG, "Class information preprocessing started")
        hsf_and_train_dir = self._by_class_nist.glob("*/*")
        hsf_dirs = [path for path in hsf_and_train_dir if "train" not in str(path)]
        images_paths = []
        for hsf_path in hsf_dirs:
            images_paths.extend(list(hsf_path.glob("*")))
        class_df = pd.DataFrame(images_paths, columns=["path_by_class"])
        chars = class_df["path_by_class"].map(lambda x: x.parent.parent.name)
        class_df["character"] = chars.map(hex_decimal_to_char)
        log(DEBUG, "The created class_df has shape %s", str(class_df.shape))
        log(DEBUG, "Class information preprocessing done")
        return class_df

    def _merge_class_and_writer_information(self) -> pd.DataFrame:
        log(DEBUG, "Merging of the class and writer information by hash values started")
        merged = pd.merge(self._writer_df, self._class_df, on="hash")
        log(
            DEBUG,
            "The calculated new df merged based on hashes has shape %s",
            str(merged.shape),
        )
        log(DEBUG, "Merging of the class and writer information by hash values done")
        return merged

    def _calculate_hashes(self) -> None:
        log(DEBUG, "Hashes calculation started")
        # Assumes that the class_df and writer_df are created
        self._writer_df["hash"] = calculate_series_hashes(
            self._writer_df["path_by_writer"]
        )
        self._class_df["hash"] = calculate_series_hashes(
            self._class_df["path_by_class"]
        )
        log(DEBUG, "Hashes calculation done")

    def _preprocess_images(self) -> pd.DataFrame:
        """Preprocess images - resize to 28x28 and save them in the processed directory.

        Returns
            preprocessed_df_info: pd.DataFrame
            dataframe with information about the path, writer_id, character(label)
        """
        log(DEBUG, "Image preprocessing started")
        writer_to_character_to_count: Dict[str, Dict[str, int]] = dict()
        resized_size = (28, 28)
        new_df = self._df.copy()
        new_df["path"] = pathlib.Path("")
        for index, row in tqdm(self._df.iterrows(), total=self._df.shape[0]):
            file_path = row["path_by_writer"]
            img = Image.open(file_path)
            gray = img.convert("L")
            gray.thumbnail(resized_size, Image.Resampling.LANCZOS)
            writer_id = row["writer_id"]
            character = row["character"]
            if writer_id not in writer_to_character_to_count:
                writer_to_character_to_count[writer_id] = {}
            if character not in writer_to_character_to_count[writer_id]:
                writer_to_character_to_count[writer_id][character] = 0
            image_name = (
                "character_"
                + character
                + "_"
                + "nth_"
                + str(writer_to_character_to_count[writer_id][character])
                + "_"
                + "author_"
                + writer_id
                + ".png"
            )
            writer_to_character_to_count[writer_id][character] += 1
            new_path = self._processed_images_dir / image_name
            gray.save(new_path)
            new_df.iloc[index, -1] = new_path
        log(
            DEBUG,
            "The dataframe with the references to preprocessed data has shape %s",
            str(new_df.shape),
        )
        log(DEBUG, "Image preprocessing done")
        new_df = new_df.loc[:, ["path", "writer_id", "character"]]
        return new_df
