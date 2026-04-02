"""fedrag: A Flower Federated RAG app."""

from typing import List

from flwr.common.typing import Parameters

from fedrag.retriever import Retriever


def str_to_parameters(text: List[str]) -> Parameters:
    tensors = [str.encode(t) for t in text]
    return Parameters(tensors=tensors, tensor_type="string")


def parameters_to_str(parameters: Parameters) -> List[str]:
    text = [param.decode() for param in parameters.tensors]
    return text


def index_exists(corpus_names):
    for corpus_name in corpus_names:
        # no need to initialize a Retriever object,
        # just call the class method.
        if not Retriever.index_exists(corpus_name):
            raise RuntimeError(
                f"Please first download the corpus and "
                f"create the corresponding index for {corpus_name}."
            )
