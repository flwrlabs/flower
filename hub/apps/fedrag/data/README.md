# Introduction

This folder holds the necessary scripts to download the necessary corpora and
create indices for more efficient document retrieval for the FedRAG workflow. All four corpora listed below were derived from the [MedRAG Toolkit](https://github.com/Teddy-XiongGZ/MedRAG) [[1]](#ref1).

The currently supported corpora are:

| **Corpus**              | **Size** | **#Doc.** | **#Snippets** | **Domain** |
| ----------------------- | -------- | --------- | ------------- | ---------- |
| PubMed [[2]](#ref2)     | ~70GBs   | 23.9M     | 23.9M         | Biomedical |
| StatPearls [[3]](#ref3) | ~2GBs    | 9.3k      | 301.2k        | Clinics    |
| Textbooks [[4]](#ref4)  | ~209MBs  | 18        | 125.8k        | Medicine   |
| Wikipedia [[5]](#ref5)  | ~44GBs   | 6.5M      | 29.9M         | General    |

For index generation we use the [FAISS](https://github.com/facebookresearch/faiss) library.

> [!NOTE]
> Please note that for each corpus, its corresponding index might need exactly the same disk space as the documents being indexed.

# How-To-Run

To download the data and prepare the retrieval indices, please run:

```bash
./prepare.sh
```

By default, the bash script will download the `StatePearls` and `Textbooks` corpus and create an index using up to the
first `100` downloaded chunks. The bash script calls the `prepare.py` script for download and indexing. To
**download all four corpora** and **prepare the index using all files** of each corpus, please run the bash script
as follows:

```bash
./prepare.sh --datasets "pubmed" "statpearls" "textbooks" "wikipedia" --index_num_chunks 0
```

## Corpora

To download the data we use the [MedRAG Toolkit](https://github.com/Teddy-XiongGZ/MedRAG) and the associated MedRAG
dataset repository in [Hugging Face](https://huggingface.co/MedRAG), which holds all the curated corpus.

> [!NOTE]
> According to the [privacy policy](https://www.statpearls.com/home/privacypolicy/) of StatPearls, the StatPearls content is not allowed to be distributed.\
> The StatPearls data are downloading directly from the [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK430685/)
> and the `statpearls.py` script provided by [MedRAG toolkit](https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/data/statpearls.py) is used to chunk the data.

The `download.py` script is responsible for downloading the corpus and executing the required data chunking operations.

## FAISS Indexer

We use the [FAISS](https://github.com/facebookresearch/faiss) index library for index creation and querying due to its
use simplicity and efficient similarity search.

To map the snippets of each corpus to an embedding space and train and create the corpus index, by default we use the
[all-MiniLM-L6-v2 sentence-transformer model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
which maps sentences & paragraphs to a 384 dimensional dense vector space.

The index is created by the `../fedrag/retriever.py` script. To build the FAISS indexer, the function `build_faiss_index()`
takes as an input the number of chunks (files) upon which the indexer should be built. If the value is set to `0`, then all
files are considered during index creation, else only the first `[:num_chunks]` files are considered. The script
also provides a helper function for document retrieval, _assuming the index has already been generated_.

Specifically, the function `query_faiss_index()` reads the created index for the requested dataset, converts the input
query to its embedded representation and performs a similarity search between the query's embedded vector and the
corpus' snippet vectorized formats.

The current implementation of document retrieval for the FAISS index is built with `IndexIVFFlat` and uses the
`faiss.METRIC_L2` metric, which means that the lower the score of a retrieved document the better, since L2 Distance
measures dissimilarity.

If you want to use another embedding model for generating the embeddings, please change the following properties
in the `retriever.yaml` file.

```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: 384
```

## QA Benchmark Datasets

All the QA benchmark datasets are downloaded from the [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) benchmark [[1]](#ref1).
The `prepare.py` script is responsible for downloading the QA datasets. The script downloads the datasets after all the
corpora are downloaded and the related FAISS indices are built.

# References

1. <a id="ref1"></a> Xiong, Guangzhi, Qiao Jin, Zhiyong Lu, and Aidong Zhang. "Benchmarking retrieval-augmented generation for medicine." In Findings of the Association for Computational Linguistics ACL 2024, pp. 6233-6251. 2024.

2. <a id="ref2"></a> PubMed corpus was created from articles located at: https://pubmed.ncbi.nlm.nih.gov/

3. <a id="ref3"></a> StatPearls corpus was created by using 9,330 publicly available StatPearls articles through the NCBI Bookshelf: https://www.ncbi.nlm.nih.gov/books/NBK430685/

4. <a id="ref4"></a> Textbooks corpus was used in the work of: Jin, Di, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. "What disease does this patient have? a large-scale open domain question answering dataset from medical exams." Applied Sciences 11, no. 14 (2021): 6421.
   The corpus is also available at: https://github.com/jind11/MedQA

5. <a id="ref5"></a> Wikipedia corpus was used in the work of: Thakur, Nandan, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. "BEIR: A heterogenous benchmark for zero-shot evaluation of information retrieval models." In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2021). The corpus is also available at: https://huggingface.co/datasets/wikipedia
