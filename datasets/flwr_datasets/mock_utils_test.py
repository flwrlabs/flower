# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for mocking datasets."""


import io
import random
import string
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from PIL import Image

import datasets
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value


def _generate_artificial_strings(
    num_rows: int, num_unique: int, string_length: int, seed: int = 42
) -> list[str]:
    """Create list of strings for categories or labels mocking.

    Note to keep the seed the same if you reuse this function for in creation of the
    dataset for multiple splits.

    Parameters
    ----------
    num_rows: int
        Number of rows = number of elements in the list.
    num_unique: int
        Number of unique strings that will be initially created.
    string_length: int
        Length of each string.
    seed: int
        Seed to the random package.

    Returns
    -------
    string_column : List[str]
        List of generated strings.
    """
    random.seed(seed)
    unique_strings: set[str] = set()
    while len(unique_strings) < num_unique:
        random_str = "".join(
            random.choices(string.ascii_letters + string.digits, k=string_length)
        )
        unique_strings.add(random_str)

    unique_strings_list = list(unique_strings)
    artificial_column = unique_strings_list.copy()
    remaining_to_allocate = num_rows - num_unique
    for _ in range(remaining_to_allocate):
        artificial_column.append(random.choice(unique_strings_list))
    return artificial_column


def _generate_artificial_categories(num_rows: int, choices: list[Any]) -> list[str]:
    """Create list of strings from given `choices` list."""
    artificial_column = choices.copy()
    remaining_to_allocate = num_rows - len(choices)
    for _ in range(remaining_to_allocate):
        artificial_column.append(random.choice(choices))
    return artificial_column


def _generate_random_word(length: int) -> str:
    """Generate a random word of the given length."""
    return "".join(random.choices(string.ascii_letters, k=length))


def _generate_random_text_column(num_rows: int, length: int) -> list[str]:
    """Generate a list of random text of specified length."""
    text_col = []
    for _ in range(num_rows):
        text_col.append(_generate_random_word(length))
    return text_col


def _generate_random_sentence(
    min_word_length: int,
    max_word_length: int,
    min_sentence_length: int,
    max_sentence_length: int,
) -> str:
    """Generate a random sentence with words of random lengths."""
    sentence_length = random.randint(min_sentence_length, max_sentence_length)
    sentence: list[str] = []
    while len(" ".join(sentence)) < sentence_length:
        word_length = random.randint(min_word_length, max_word_length)
        word = _generate_random_word(word_length)
        sentence.append(word)
    return " ".join(sentence)


def _generate_random_sentences(
    num_rows: int,
    min_word_length: int,
    max_word_length: int,
    min_sentence_length: int,
    max_sentence_length: int,
) -> list[str]:
    """Generate a list of random sentences."""
    text_col = [
        _generate_random_sentence(
            min_word_length, max_word_length, min_sentence_length, max_sentence_length
        )
        for _ in range(num_rows)
    ]
    return text_col


def _make_num_rows_none(column: list[Any], num_none: int) -> list[Any]:
    """Assign none num_none times to the given list."""
    column_copy = column.copy()
    none_positions = random.sample(range(len(column_copy)), num_none)
    for pos in none_positions:
        column_copy[pos] = None
    return column_copy


def _generate_random_date(
    start_date: datetime,
    end_date: datetime,
    date_format: str = "%a %b %d %H:%M:%S %Y",
    as_string: bool = True,
) -> str | datetime:
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    random_seconds = random.randint(0, int(time_between_dates.total_seconds()))
    random_date = start_date + timedelta(seconds=random_seconds)

    if as_string:
        return random_date.strftime(date_format)
    return random_date


def _generate_random_date_column(
    num_rows: int,
    start_date: datetime,
    end_date: datetime,
    date_format: str = "%a %b %d %H:%M:%S %Y",
    as_string: bool = True,
) -> list[str | datetime]:
    """Generate a list of random dates."""
    return [
        _generate_random_date(start_date, end_date, date_format, as_string)
        for _ in range(num_rows)
    ]


def _generate_random_int_column(num_rows: int, min_int: int, max_int: int) -> list[int]:
    """Generate a list of ints."""
    return [random.randint(min_int, max_int) for _ in range(num_rows)]


def _generate_random_bool_column(num_rows: int) -> list[bool]:
    """Generate a list of bools."""
    return [random.choice([True, False]) for _ in range(num_rows)]


def _generate_random_image_column(
    num_rows: int,
    image_size: tuple[int, int] | tuple[int, int, int],
    simulate_type: str,
) -> list[Any]:
    """Simulate the images with the format that is found in HF Hub.

    Directly using `Image.fromarray` does not work because it creates `PIL.Image.Image`.
    """
    # Generate numpy images
    np_images = []
    for _ in range(num_rows):
        np_images.append(np.random.randint(0, 255, size=image_size, dtype=np.uint8))
    # Change the format to the PIL.PngImagePlugin.PngImageFile
    # or the PIL.JpegImagePlugin.JpegImageFile format
    pil_imgs = []
    for np_image in np_images:
        # Convert the NumPy array to a PIL image
        pil_img_beg = Image.fromarray(np_image)

        # Save the image to an in-memory bytes buffer
        in_memory_file = io.BytesIO()
        pil_img_beg.save(in_memory_file, format=simulate_type)
        in_memory_file.seek(0)

        # Reload the image as a PngImageFile
        pil_image_end = Image.open(in_memory_file)
        pil_imgs.append(pil_image_end)
    return pil_imgs


def generate_random_audio_column(
    num_rows: int,
    sampling_rate: int,
    length_in_samples: int,
) -> list[dict[str, Any]]:
    """Simulate the audio column.

    Audio column in the dataset is comprised from an array or floats, sample_rate and a
    path.
    """
    # Generate numpy images
    audios = []
    for _ in range(num_rows):
        audio_array = np.random.uniform(low=-1.0, high=1.0, size=length_in_samples)
        audios.append(
            {"path": None, "array": audio_array, "sampling_rate": sampling_rate}
        )
    return audios


def _mock_sentiment140(num_rows: int) -> Dataset:
    users = _generate_artificial_strings(
        num_rows=num_rows, num_unique=30, string_length=5
    )
    sentiment = _generate_artificial_categories(num_rows=num_rows, choices=[0, 4])
    query = ["NO_QUERY"] * num_rows

    # Sentences
    min_word_length = 3
    max_word_length = 8
    min_sentence_length = 20
    max_sentence_length = 60

    text = _generate_random_sentences(
        num_rows,
        min_word_length,
        max_word_length,
        min_sentence_length,
        max_sentence_length,
    )

    start_date = datetime(2009, 1, 1)
    end_date = datetime(2010, 12, 31)
    date_format = "%a %b %d %H:%M:%S %Y"

    # Generate a list of random dates as strings
    date = _generate_random_date_column(
        num_rows, start_date, end_date, date_format, as_string=True
    )

    features = Features(
        {
            "text": Value(dtype="string"),
            "date": Value(dtype="string"),
            "user": Value(dtype="string"),
            "sentiment": Value(dtype="int32"),
            "query": Value(dtype="string"),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "user": users,
            "sentiment": sentiment,
            "query": query,
            "text": text,
            "date": date,
        },
        features=features,
    )
    return dataset


def _mock_cifar100(num_rows: int) -> Dataset:
    imgs = _generate_random_image_column(num_rows, (32, 32, 3), "PNG")
    unique_fine_labels = _generate_artificial_strings(
        num_rows=100, num_unique=100, string_length=10, seed=42
    )
    fine_label = _generate_artificial_categories(num_rows, unique_fine_labels)
    unique_coarse_labels = _generate_artificial_strings(
        num_rows=20, num_unique=20, string_length=10, seed=42
    )

    coarse_label = _generate_artificial_categories(num_rows, unique_coarse_labels)
    features = Features(
        {
            "img": datasets.Image(decode=True),
            "fine_label": ClassLabel(names=unique_fine_labels),
            "coarse_label": ClassLabel(names=unique_coarse_labels),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {"img": imgs, "coarse_label": coarse_label, "fine_label": fine_label},
        features=features,
    )
    return dataset


def _mock_svhn_cropped_digits(num_rows: int) -> Dataset:
    imgs = _generate_random_image_column(num_rows, (32, 32, 3), "PNG")
    unique_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    label = _generate_artificial_categories(num_rows, unique_labels)
    features = Features(
        {
            "image": datasets.Image(decode=True),
            "label": ClassLabel(names=unique_labels),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {"image": imgs, "label": label}, features=features
    )
    return dataset


def _mock_speach_commands(num_rows: int) -> Dataset:
    sampling_rate = 16_000
    length_in_samples = 16_000
    imgs = generate_random_audio_column(
        num_rows=num_rows,
        sampling_rate=sampling_rate,
        length_in_samples=length_in_samples,
    )
    unique_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    label = _generate_artificial_categories(num_rows, unique_labels)
    is_unknown = _generate_random_bool_column(num_rows)
    utterance_id = _generate_random_int_column(num_rows, 0, 10)
    unique_ids = _generate_random_text_column(num_rows // 10, 5)
    speaker_id = _generate_artificial_categories(num_rows, unique_ids)
    speaker_id = _make_num_rows_none(speaker_id, 10)
    features = Features(
        {
            "audio": datasets.Audio(
                sampling_rate=sampling_rate, num_channels=1, decode=True
            ),
            "is_unknown": Value(dtype="bool"),
            "speaker_id": Value(dtype="string"),
            "utterance_id": Value(dtype="int8"),
            "label": ClassLabel(names=unique_labels),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "audio": imgs,
            "is_unknown": is_unknown,
            "speaker_id": speaker_id,
            "utterance_id": utterance_id,
            "label": label,
        },
        features=features,
    )
    return dataset


dataset_name_to_mock_function = {
    "cifar100": _mock_cifar100,
    "sentiment140": _mock_sentiment140,
    "svhn_cropped_digits": _mock_svhn_cropped_digits,
    "speech_commands_v0.01": _mock_speach_commands,
}


def _load_mocked_dataset(
    dataset_name: str,
    num_rows: list[int],
    split_names: list[str],
    subset: str = "",
) -> DatasetDict:
    dataset_dict = {}
    name = dataset_name if subset == "" else dataset_name + "_" + subset
    dataset_creation_fnc = dataset_name_to_mock_function[name]
    for params in zip(num_rows, split_names, strict=False):
        dataset_dict[params[1]] = dataset_creation_fnc(params[0])
    return datasets.DatasetDict(dataset_dict)


def _load_mocked_dataset_by_partial_download(
    dataset_name: str,
    split_name: str,
    skip_take_list: list[tuple[int, int]],
    subset_name: str | None = None,
) -> Dataset:
    """Download a partial dataset.

    This functionality is not supported in the datasets library. This is an informal
    way of achieving partial dataset download by using the `streaming=True` and creating
    a dataset.Dataset from in-memory objects.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset (passed to load_dataset).
    split_name: str
        Name of the split (passed to load_dataset) e.g. "train".
    skip_take_list: List[Tuple[int, int]]
        The streaming mode has a specific type of accessing the data, the first tuple
        value is how many samples to skip, the second is how many samples to take. Due
        to this mechanism, diverse samples can be taken (especially if the dataset is
        sorted by the natural_id for NaturalIdPartitioner).
    subset_name: Optional[str]
        Name of the subset (passed to load_dataset) e.g. "v0.01" for speech_commands.

    Returns
    -------
    dataset: Dataset
        The dataset with the requested samples.
    """
    dataset = datasets.load_dataset(
        dataset_name,
        name=subset_name,
        split=split_name,
        streaming=True,
        trust_remote_code=True,
    )
    dataset_list = []
    # It's a list of dict such that each dict represent a single sample of the dataset
    # The sample is exactly the same as if the full dataset was downloaded and indexed
    for skip, take in skip_take_list:
        # dataset.skip(n).take(m) in streaming mode is equivalent (in terms of return)
        # to the fully downloaded dataset index: dataset[n+1: (n+1 + m)]
        dataset_list.extend(list(dataset.skip(skip).take(take)))
    return Dataset.from_list(dataset_list)


def _load_mocked_dataset_dict_by_partial_download(
    dataset_name: str,
    split_names: list[str],
    skip_take_lists: list[list[tuple[int, int]]],
    subset_name: str | None = None,
) -> DatasetDict:
    """Like _load_mocked_dataset_by_partial_download but for many splits."""
    assert len(split_names) == len(
        skip_take_lists
    ), "The split_names should be thesame length as the skip_take_lists."
    dataset_dict = {}
    for split_name, skip_take_list in zip(split_names, skip_take_lists, strict=False):
        dataset_dict[split_name] = _load_mocked_dataset_by_partial_download(
            dataset_name, split_name, skip_take_list, subset_name
        )
    return DatasetDict(dataset_dict)
