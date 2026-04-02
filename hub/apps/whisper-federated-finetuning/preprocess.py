import argparse
from multiprocessing import Pool
from time import time

import tomli
from datasets import load_dataset

from whisper_example.dataset import load_data

parser = argparse.ArgumentParser(description="Whisper preprocessing")

parser.add_argument(
    "--partition-id", type=int, help="The partition to create and save."
)

args = parser.parse_args()


# Open and read the pyproject.toml
with open("pyproject.toml", "rb") as file:
    flwr_config = tomli.load(file)["tool"]["flwr"]

# Display
print(flwr_config)
remove_cols = flwr_config["app"]["config"]["remove-cols"]
num_supernodes = flwr_config["federations"]["local-sim"]["options"]["num-supernodes"]

# If specified one partition, only that one will be processed and saved to the current directory
if args.partition_id:
    print(f"Pre-processing partition {args.partition_id} only.")
else:
    print(f"Pre-processing dataset into {num_supernodes} partitions.")


def process_one_partition(partition_id: int, save: bool = False):
    pp = load_data(partition_id, remove_cols)
    if save:
        file_name = f"partition_{partition_id}"
        pp.save_to_disk(file_name)
        print(f"Saved partition to disk: {file_name}")


if __name__ == "__main__":

    # Download train set
    _ = load_dataset("speech_commands", "v0.02", split="train", token=False)

    # Parallelize the processing of each partition in the dataset
    t_start = time()
    num_proc = None  # set it if you want to limit the number of processes

    if args.partition_id:
        process_one_partition(args.partition_id, True)

    else:
        with Pool(num_proc) as pool:
            pool.map(process_one_partition, range(num_supernodes))
        print(
            f"Pre-processing {num_supernodes} partitions took: {time() - t_start:.2f} s"
        )
