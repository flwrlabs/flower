"""fedrag: A Flower Federated RAG app."""

import os
from pathlib import Path

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CORPUS_DIR = os.path.join(DIR_PATH, "./corpus")


class DownloadCorpora:

    @classmethod
    def download(cls, corpus: str, download_dir: str = None) -> str:

        if not download_dir:
            download_dir = CORPUS_DIR
            Path(download_dir).mkdir(parents=True, exist_ok=True)

        fullpath = os.path.join(download_dir, corpus)
        # if the path already exists then skip corpus downloading
        if os.path.exists(fullpath):
            return fullpath
        if corpus != "statpearls":
            os.system(
                "GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(
                    corpus, fullpath
                )
            )
            # Go to the new directory and pull all large files using the Git LFS extension, and back again
            os.system("cd {:s} && git lfs pull && cd ..".format(fullpath))
        else:
            # Download directly from the NIH repo
            os.system(
                "wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(
                    fullpath
                )
            )
            os.system(
                "tar -xzvf {:s} -C {:s}".format(
                    os.path.join(fullpath, "statpearls_NBK430685.tar.gz"), fullpath
                )
            )
            print("Chunking the statpearls corpus...")
            # Use the provided statpearls.py script to split the files into chunks
            os.system("python {}".format(os.path.join(DIR_PATH, "statpearls.py")))
        print(f"Downloaded {corpus} corpus at {fullpath}.")
        return fullpath
