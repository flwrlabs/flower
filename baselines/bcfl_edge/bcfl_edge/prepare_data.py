"""bcfl_edge: Download Speech Commands v0.01 and convert it to Parquet.

`google/speech_commands` on the Hugging Face Hub is a loading script, which
recent versions of `datasets` cannot run. This script builds the same data from
Google's official archive instead, keeping the paper's eight keywords and the
official test split (`testing_list.txt`). The remaining clips, including the
official validation list, form the training split.

    python -m bcfl_edge.prepare_data data/speech_commands

Then pass the printed absolute path as the `dataset` run-config value.
"""

import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, Features

from bcfl_edge.dataset import KEYWORDS

# The same official file, from Google Cloud Storage first.
URLS = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.01.tar.gz",
    "https://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
)

README = """---
configs:
- config_name: v0.01
  data_files:
  - split: train
    path: v0.01/train.parquet
  - split: test
    path: v0.01/test.parquet
---
Speech Commands v0.01 (CC BY 4.0, https://arxiv.org/abs/1804.03209),
restricted to the keywords: {keywords}.
"""


def _wanted(name: str) -> bool:
    name = name.removeprefix("./")
    return name.split("/")[0] in KEYWORDS or name == "testing_list.txt"


def _clips(root: Path, split: str):
    test_files = set((root / "testing_list.txt").read_text().split())
    for label, word in enumerate(KEYWORDS):
        for wav in sorted((root / word).glob("*.wav")):
            if (f"{word}/{wav.name}" in test_files) == (split == "test"):
                yield {
                    "audio": {"bytes": wav.read_bytes(), "path": None},
                    "label": label,
                }


def convert(root: Path, out_dir: Path) -> None:
    """Write the train and test splits of an extracted archive as Parquet."""
    features = Features({"audio": Audio(), "label": ClassLabel(names=list(KEYWORDS))})
    (out_dir / "v0.01").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dataset = Dataset.from_generator(
            _clips, gen_kwargs={"root": root, "split": split}, features=features
        )
        dataset.to_parquet(out_dir / "v0.01" / f"{split}.parquet")
        print(f"{split}: {len(dataset)} clips")
    (out_dir / "README.md").write_text(README.format(keywords=", ".join(KEYWORDS)))


def download(path: Path) -> None:
    """Download the archive, trying each official URL in turn."""
    for url in URLS:
        print(f"Downloading {url} (1.4 GB)...")
        try:
            urllib.request.urlretrieve(url, path)
            return
        except OSError as error:
            print(f"  failed: {error}")
    raise SystemExit(
        "Could not download the archive. Download it in a browser from\n"
        f"  {URLS[0]}\nand pass it with --archive <path>."
    )


def main() -> None:
    """Download, extract and convert the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded archive"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / "speech_commands_v0.01.tar.gz"
            download(archive)
        root = Path(tmp) / "extracted"
        with tarfile.open(archive) as tar:
            # Extract only the eight keyword folders and the test list.
            members = [m for m in tar.getmembers() if _wanted(m.name)]
            tar.extractall(root, members=members, filter="data")
        convert(root, args.out_dir)
    print(f"Done. Run with: --run-config \"dataset='{args.out_dir.resolve()}'\"")


if __name__ == "__main__":
    main()
