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
"""Build the Flower Example docs."""

import os
import re
import shutil
import subprocess
from pathlib import Path

CategoryContent = dict[str, str]

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "examples" / "docs" / "source" / "index.rst"

INITIAL_TEXT = """
Flower Examples Documentation
-----------------------------

Welcome to Flower Examples' documentation. `Flower <https://flower.ai>`_ is
a friendly federated AI framework.

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers,
engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack

Quickstart Examples
-------------------

Flower Quickstart Examples are a collection of demo projects that show how you
can use Flower in combination with other existing frameworks or technologies.

"""

TABLE_HEADERS = (
    "\n.. list-table::\n   :widths: 50 15 15 15\n   "
    ":header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
)

CATEGORIES: dict[str, CategoryContent] = {
    "quickstart": {"table": TABLE_HEADERS, "list": ""},
    "advanced": {"table": TABLE_HEADERS, "list": ""},
    "other": {"table": TABLE_HEADERS, "list": ""},
}

URLS: dict[str, str] = {
    # Frameworks
    "Android": "https://www.android.com/",
    "catboost": "https://catboost.ai/docs/en/",
    "C++": "https://isocpp.org/",
    "Docker": "https://www.docker.com/",
    "JAX": "https://jax.readthedocs.io/en/latest/",
    "Java": "https://www.java.com/",
    "Keras": "https://keras.io/",
    "Kotlin": "https://kotlinlang.org/",
    "MLX": "https://ml-explore.github.io/mlx/build/html/index.html",
    "MONAI": "https://monai.io/",
    "PEFT": "https://huggingface.co/docs/peft/index",
    "Swift": "https://www.swift.org/",
    "TensorFlowLite": "https://www.tensorflow.org/lite",
    "fastai": "https://fast.ai/",
    "lifelines": "https://lifelines.readthedocs.io/en/latest/index.html",
    "lightning": "https://lightning.ai/docs/pytorch/stable/",
    "numpy": "https://numpy.org/",
    "opacus": "https://opacus.ai/",
    "pandas": "https://pandas.pydata.org/",
    "scikit-learn": "https://scikit-learn.org/",
    "tensorboard": "https://www.tensorflow.org/tensorboard",
    "tensorflow": "https://www.tensorflow.org/",
    "torch": "https://pytorch.org/",
    "torchvision": "https://pytorch.org/vision/stable/index.html",
    "transformers": "https://huggingface.co/docs/transformers/index",
    "wandb": "https://wandb.ai/home",
    "whisper": "https://huggingface.co/openai/whisper-tiny",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    # Datasets
    "Adult Census Income": "https://www.kaggle.com/datasets/uciml/adult-census-income/data",
    "Alpaca-GPT4": "https://huggingface.co/datasets/vicgalle/alpaca-gpt4",
    "CIFAR-10": "https://huggingface.co/datasets/uoft-cs/cifar10",
    "HIGGS": "https://archive.ics.uci.edu/dataset/280/higgs",
    "IMDB": "https://huggingface.co/datasets/stanfordnlp/imdb",
    "Iris": "https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html",
    "MNIST": "https://huggingface.co/datasets/ylecun/mnist",
    "MedNIST": "https://medmnist.com/",
    "Oxford Flower-102": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    "SpeechCommands": "https://huggingface.co/datasets/google/speech_commands",
    "Titanic": "https://www.kaggle.com/competitions/titanic",
    "Waltons": (
        "https://lifelines.readthedocs.io/en/latest/"
        "lifelines.datasets.html#lifelines.datasets.load_waltons"
    ),
}


def _convert_to_link(search_result: str) -> str:
    if "," in search_result:
        result = ""
        for part in search_result.split(","):
            result += f"{_convert_to_link(part)}, "
        return result[:-2]

    search_result = search_result.strip()
    name, url = search_result, URLS.get(search_result, None)
    if url:
        return f"`{name.strip()} <{url.strip()}>`_"
    return search_result


def _read_metadata(example: str) -> tuple[str, str, str, str]:
    with open(os.path.join(example, "README.md"), encoding="utf-8") as f:
        content = f.read()

    metadata_match = re.search(r"^---(.*?)^---", content, re.DOTALL | re.MULTILINE)
    if not metadata_match:
        raise ValueError("Metadata block not found")
    metadata = metadata_match.group(1)

    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if not title_match:
        raise ValueError("Title not found in metadata")
    title = title_match.group(1).strip()

    tags_match = re.search(r"^tags:\s*\[(.+?)\]$", metadata, re.MULTILINE)
    if not tags_match:
        raise ValueError("Tags not found in metadata")
    tags = tags_match.group(1).strip()

    dataset_match = re.search(
        r"^dataset:\s*\[(.*?)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if not dataset_match:
        raise ValueError("Dataset not found in metadata")
    dataset = dataset_match.group(1).strip()

    framework_match = re.search(
        r"^framework:\s*\[(.*?|)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if not framework_match:
        raise ValueError("Framework not found in metadata")
    framework = framework_match.group(1).strip()

    dataset = _convert_to_link(re.sub(r"\s+", " ", dataset).strip())
    framework = _convert_to_link(re.sub(r"\s+", " ", framework).strip())
    return title, tags, dataset, framework


def _add_table_entry(example: str, tag: str, table_var: str) -> bool:
    title, tags, dataset, framework = _read_metadata(example)
    example_name = Path(example).stem
    table_entry = (
        f"   * - `{title} <{example_name}.html>`_ \n     "
        f"- {framework} \n     - {dataset} \n     - {tags}\n\n"
    )
    if tag in tags:
        CATEGORIES[table_var]["table"] += table_entry
        CATEGORIES[table_var]["list"] += f"  {example_name}\n"
        return True
    return False


def _copy_markdown_files(example: str) -> None:
    for file in os.listdir(example):
        if file.endswith(".md"):
            src = os.path.join(example, file)
            dest = os.path.join(
                str(ROOT),
                "examples",
                "docs",
                "source",
                os.path.basename(example) + ".md",
            )
            shutil.copyfile(src, dest)


def _add_gh_button(example: str) -> None:
    gh_text = (
        '[<img src="_static/view-gh.png" alt="View on GitHub" width="200"/>]'
        f"(https://github.com/flwrlabs/flower/blob/main/hub/apps/{example})"
    )
    readme_file = os.path.join(str(ROOT), "examples", "docs", "source", example + ".md")
    with open(readme_file, "r+", encoding="utf-8") as f:
        content = f.read()
        if gh_text not in content:
            content = re.sub(
                r"(^# .+$)", rf"\1\n\n{gh_text}", content, count=1, flags=re.MULTILINE
            )
            f.seek(0)
            f.write(content)
            f.truncate()


def _copy_images(example: str) -> None:
    static_dir = os.path.join(example, "_static")
    dest_dir = os.path.join(str(ROOT), "examples", "docs", "source", "_static")
    if os.path.isdir(static_dir):
        for file in os.listdir(static_dir):
            if file.endswith((".jpg", ".png", ".jpeg")):
                shutil.copyfile(
                    os.path.join(static_dir, file), os.path.join(dest_dir, file)
                )


def _add_all_entries() -> None:
    examples_dir = os.path.join(ROOT, "hub", "apps")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "docs":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example)


def _main() -> None:
    if INDEX.exists():
        INDEX.unlink()

    with INDEX.open("w", encoding="utf-8") as index_file:
        index_file.write(INITIAL_TEXT)

    examples_dir = os.path.join(ROOT, "hub", "apps")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "docs":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example_path)
            if not _add_table_entry(example_path, "quickstart", "quickstart"):
                if not _add_table_entry(example_path, "comprehensive", "comprehensive"):
                    if not _add_table_entry(example_path, "advanced", "advanced"):
                        _add_table_entry(example_path, "", "other")

    with INDEX.open("a", encoding="utf-8") as index_file:
        index_file.write(CATEGORIES["quickstart"]["table"])

        index_file.write("\nAdvanced Examples\n-----------------\n")
        index_file.write(
            "Advanced Examples are mostly for users that are both familiar with "
            "Federated Learning but also somewhat familiar with Flower's main "
            "features.\n"
        )
        index_file.write(CATEGORIES["advanced"]["table"])

        index_file.write("\nOther Examples\n--------------\n")
        index_file.write(
            "Flower Examples are a collection of example projects written with "
            "Flower that explore different domains and features. You can check "
            "which examples already exist and/or contribute your own example.\n"
        )
        index_file.write(CATEGORIES["other"]["table"])

        _add_all_entries()

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Quickstart\n  :hidden:\n\n"
        )
        index_file.write(CATEGORIES["quickstart"]["list"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Advanced\n  :hidden:\n\n"
        )
        index_file.write(CATEGORIES["advanced"]["list"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Others\n  :hidden:\n\n"
        )
        index_file.write(CATEGORIES["other"]["list"])

        index_file.write("\n")


if __name__ == "__main__":
    _main()
    subprocess.call(f"cd {ROOT}/examples/docs && make html", shell=True)
