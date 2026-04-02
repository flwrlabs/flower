"""fedrag: A Flower Federated RAG app."""

""" 
This file is a modified copy of the https://github.com/Teddy-XiongGZ/MIRAGE/blob/main/src/utils.py 
script located in the MIRAGE toolkit.
"""

import json

import requests


class MirageQA:

    # Where the curated MIRAGE QA benchmark dataset is located.
    RAW_JSON_FILE = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/refs/heads/main/benchmark.json"

    def __init__(self, data, benchmark_filepath):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(benchmark_filepath))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset[self.index[key]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")

    @classmethod
    def download(cls, filepath):
        response = requests.get(cls.RAW_JSON_FILE, stream=True)
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
