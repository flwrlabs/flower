"""fedrag: A Flower Federated RAG app."""

import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
from collections import OrderedDict

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FAISS_DEFAULT_CONFIG = os.path.join(DIR_PATH, "retriever.yaml")
CORPUS_DIR = os.path.join(DIR_PATH, "../data/corpus")


class Retriever:

    def __init__(self, config_file=None):
        if not config_file:
            self.config = yaml.safe_load(open(FAISS_DEFAULT_CONFIG, "r"))
        else:
            self.config = yaml.safe_load(open(config_file, "r"))
        device = st_util.get_device_name()
        # load the embedding model and define the embeddings dimensions
        # for the device placement of the SentenceTransformers model we resort
        # to use the device name returned by `sentence_transformers.util.get_device_name()`
        # which will be called by the SentenceTransformer constructor when creating the model
        self.emb_model = SentenceTransformer(self.config["embedding_model"])
        self.emb_dim = self.config["embedding_dimension"]

    def build_faiss_index(self, dataset_name, batch_size=32, num_chunks=None):
        dataset_dir = os.path.join(CORPUS_DIR, f"{dataset_name}")
        index_path = os.path.join(dataset_dir, "faiss.index")
        doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")

        try:
            # erase previous files whenever
            # index builder is called
            os.remove(index_path)
            os.remove(doc_ids_path)
        except OSError:
            pass

        all_embeddings, all_doc_ids = [], []
        chunk_dir = os.path.join(dataset_dir, "chunk")
        all_files = [f.path for f in os.scandir(chunk_dir)]  # get full paths
        # if chunks is given just load the specified
        # number of chunks; useful for dev and debug purposes
        if num_chunks:
            all_files = all_files[:num_chunks]

        # Loop through all the .jsonl files, load the id and the content of
        # each document and for each document generate its embeddings
        for filename in tqdm(all_files):
            batch_content, batch_ids = [], []
            with open(filename, "r", encoding="utf-8") as infile:
                for line in infile:
                    doc = json.loads(line)
                    doc_id = doc.get("id", "")
                    content = doc.get("content", "")
                    batch_ids.append(doc_id)
                    batch_content.append(content)

                    if len(batch_ids) > batch_size:
                        # Generate embeddings for the batch
                        batch_embeddings = self.emb_model.encode(
                            batch_content, convert_to_numpy=True
                        )
                        all_embeddings.extend(batch_embeddings)
                        all_doc_ids.extend(batch_ids)
                        batch_content, batch_ids = [], []

                # Process last batch
                if batch_content:
                    batch_embeddings = self.emb_model.encode(
                        batch_content, convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    all_doc_ids.extend(batch_ids)

        # Filter out embeddings if they do not have the expected dimensions
        filtered_embeddings = [
            embedding
            for embedding in all_embeddings
            if embedding is not None and embedding.shape == (self.emb_dim,)
        ]
        # FAISS needs float32, hence the casting
        embeddings = np.array(filtered_embeddings).astype("float32")
        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Quantizer for IVF
        quantizer = faiss.IndexFlatL2(d)

        # Number of clusters
        nlist = int(np.sqrt(len(embeddings)))

        # METRIC_L2 measures dissimilarity, hence the lower the score the better!
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Train the index
        index.train(embeddings)

        # Add the embeddings to the index
        index.add(embeddings)

        # Save the index
        faiss.write_index(index, index_path)

        # Save document IDs
        np.save(doc_ids_path, np.array(all_doc_ids))

        return

    def query_faiss_index(self, dataset_name, query, knn=8):
        dataset_dir = os.path.join(CORPUS_DIR, f"{dataset_name}")
        index_path = os.path.join(dataset_dir, "faiss.index")
        doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")

        if not os.path.exists(doc_ids_path) or not os.path.exists(index_path):
            raise RuntimeError("FAISS index is not built yet.")

        # 1. Load the FAISS index and document IDs
        index = faiss.read_index(index_path)
        doc_ids = np.load(doc_ids_path)

        # 2. Generate query embedding
        query_embedding = self.emb_model.encode(query)

        # 3. Search the index
        # CAUTION: since our FAISS index is built with
        # IndexIVFFlat and metric faiss.METRIC_L2, the
        # lower the score the better, since L2 Distance
        # measures dissimilarity.
        doc_scores, doc_idx = index.search(np.array([query_embedding]), knn)

        # 4. Retrieve the relevant document IDs
        doc_scores = doc_scores[0]  # flatten scores
        retrieved_doc_ids = doc_ids[doc_idx][0]  # flatten ids

        # 5. Prepare and return the results
        chunk_dir = os.path.join(dataset_dir, "chunk")
        final_res = OrderedDict()
        for i, (doc_id, doc_score) in enumerate(zip(retrieved_doc_ids, doc_scores)):
            doc_pref_suf = doc_id.split("_")
            doc_name, snippet_idx = "_".join(doc_pref_suf[:-1]), int(doc_pref_suf[-1])
            full_file = os.path.join(chunk_dir, doc_name + ".jsonl")
            loaded_snippet = json.loads(
                open(full_file).read().strip().split("\n")[snippet_idx]
            )
            rank = i + 1
            final_res[doc_id] = {
                "rank": int(rank),
                "score": float(doc_score),
                "title": str(loaded_snippet["title"]),
                "content": str(loaded_snippet["content"]),
            }

        return final_res

    @classmethod
    def index_exists(cls, dataset_name):
        dataset_dir = os.path.join(CORPUS_DIR, f"{dataset_name}")
        index_path = os.path.join(dataset_dir, "faiss.index")
        doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")
        return os.path.exists(index_path) and os.path.exists(doc_ids_path)
