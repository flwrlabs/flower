"""fedrag: A Flower Federated RAG app."""

import hashlib
import os
import time
from collections import defaultdict
from itertools import cycle
from time import sleep

import numpy as np
from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from sklearn.metrics import accuracy_score

from fedrag.llm_querier import LLMQuerier
from fedrag.mirage_qa import MirageQA
from fedrag.task import index_exists


def node_online_loop(grid: Grid) -> list[int]:
    node_ids = []
    while not node_ids:
        # Get IDs of nodes available
        node_ids = grid.get_node_ids()
        # Wait if no node is available
        sleep(1)
    return node_ids


def get_hash(doc):
    # Create and return an SHA-256 hash for the given document
    return hashlib.sha256(doc.encode())


def merge_documents(documents, scores, knn, k_rrf=0, reverse_sort=False) -> list[str]:
    RRF_dict = defaultdict(dict)
    sorted_scores = np.array(scores).argsort()
    if reverse_sort:  # from larger to smaller scores
        sorted_scores = sorted_scores[::-1]
    sorted_documents = [documents[i] for i in sorted_scores]

    if k_rrf == 0:
        # If k_rff is not set then simply return the
        # sorted documents based on their retrieval score
        return sorted_documents[:knn]
    else:
        for doc_idx, doc in enumerate(sorted_documents):
            # Given that some returned results/documents could be extremely
            # large we cannot use the original document as a dictionary key.
            # Therefore, we first hash the returned string/document to a
            # representative hash code, and we use that code as a key for
            # the final RRF dictionary. We follow this approach, because a
            # document could  have been retrieved twice by multiple clients
            # but with different scores and depending on these scores we need
            # to maintain its ranking
            doc_hash = get_hash(doc)
            RRF_dict[doc_hash]["rank"] = 1 / (k_rrf + doc_idx + 1)
            RRF_dict[doc_hash]["doc"] = doc

        RRF_docs = sorted(RRF_dict.values(), key=lambda x: x["rank"], reverse=True)
        docs = [rrf_res["doc"] for rrf_res in RRF_docs][
            :knn
        ]  # select the final top-k / k-nn
        return docs


def submit_question(
    grid: Grid,
    question: str,
    question_id: str,
    knn: int,
    node_ids: list,
    corpus_names_iter: iter,
):

    messages = []
    # Send the same Message to each connected node (which run `ClientApp` instances)
    for node_idx, node_id in enumerate(node_ids):
        # The payload of a Message is of type RecordDict
        # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
        # which can carry different types of records. We'll use a ConfigRecord object
        # We need to create a new ConfigRecord() object for every node, otherwise
        # if we just override a single key, e.g., corpus_name, the grid will send
        # the same object to all nodes.
        config_record = ConfigRecord()
        config_record["question"] = question
        config_record["question_id"] = question_id
        config_record["knn"] = knn
        # Round-Robin assignment of corpus to individual clients
        # by infinitely looping over the corpus names.
        config_record["corpus_name"] = next(corpus_names_iter)

        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=node_id,
            group_id=str(question_id),
        )
        messages.append(message)

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    print("Received {}/{} results".format(len(replies), len(messages)))

    documents, scores = [], []
    for reply in replies:
        if reply.has_content():
            documents.extend(reply.content["docs_n_scores"]["documents"])
            scores.extend(reply.content["docs_n_scores"]["scores"])

    return documents, scores


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    node_ids = node_online_loop(grid)

    # k-reciprocal-rank-fusion is used by the server to merge
    # the results returned by the clients
    k_rrf = int(context.run_config["k-rrf"])
    # k-nearest-neighbors for document retrieval at each client
    knn = int(context.run_config["k-nn"])
    corpus_names = context.run_config["clients-corpus-names"].split("|")
    corpus_names = [c.lower() for c in corpus_names]  # make them lower case
    # Before we start the execution of the FedRAG pipeline,
    # we need to make sure we have downloaded the corpus and
    # created the respective indices
    index_exists(corpus_names)
    # Create corpus iterator
    corpus_names_iter = cycle(corpus_names)
    qa_datasets = context.run_config["server-qa-datasets"].split("|")
    qa_datasets = [qa_d.lower() for qa_d in qa_datasets]  # make them lower case
    qa_num = context.run_config.get("server-qa-num", None)
    model_name = context.run_config["server-llm-hfpath"]
    use_gpu = context.run_config.get("server-llm-use-gpu", False)
    use_gpu = True if use_gpu.lower() == "true" else False

    mirage_file = os.path.join(os.path.dirname(__file__), "../data/mirage.json")
    datasets = {key: MirageQA(key, mirage_file) for key in qa_datasets}

    llm_querier = LLMQuerier(model_name, use_gpu)
    expected_answers, predicted_answers, question_times, unanswered_questions = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(int),
    )
    for dataset_name in qa_datasets:
        q_idx = 0
        print("Evaluating Dataset: [{:s}] ".format(dataset_name))
        for q in datasets[dataset_name]:
            q_idx += 1
            q_id = f"{dataset_name}_{q_idx}"
            # exit question loop if number of questions has been exceeded
            if qa_num and q_idx > qa_num:
                break
            question = q["question"]
            q_st = time.time()
            docs, scores = submit_question(
                grid, question, q_id, knn, node_ids, corpus_names_iter
            )
            merged_docs = merge_documents(docs, scores, knn, k_rrf)
            options = q["options"]
            answer = q["answer"]

            prompt, predicted_answer = llm_querier.answer(
                question, merged_docs, options, dataset_name
            )

            # If the model did not predict any value,
            # then discard the question
            if predicted_answer is not None:
                expected_answers[dataset_name].append(answer)
                predicted_answers[dataset_name].append(predicted_answer)
                q_et = time.time()
                q_time = q_et - q_st  # elapsed time in seconds
                question_times[dataset_name].append(q_time)
            else:
                unanswered_questions[dataset_name] += 1

    print(
        "Below, for each benchmark dataset (QA Dataset), we show: \n"
        "(1) the evaluation results in terms of the total number of Federated RAG queries executed (Total Questions). \n"
        "(2) the total number of queries answered by the LLM when prompted with the retrieved documents from the federation clients (Answered Questions). \n"
        "(3) the overall performance of the Federated RAG pipeline (Accuracy), i.e., expected answer vs. predicted answer by the LLM. \n"
        "(4) the mean wall-clock time (Mean Querying Time) for executing all Federated RAG queries; from the time the server submits the query to "
        "the clients to the time the server receives the final prediction result from the LLM model when prompted with the retrieved documents.\n"
    )
    for dataset_name in qa_datasets:
        exp_ans = expected_answers[dataset_name]
        pred_ans = predicted_answers[dataset_name]
        not_answered = unanswered_questions[dataset_name]
        total_questions = len(exp_ans) + not_answered
        accuracy = 0.0
        if exp_ans and pred_ans:  # make sure that both collections have values inside
            accuracy = accuracy_score(exp_ans, pred_ans)
        elapsed_time = np.mean(question_times[dataset_name])
        print(
            f"QA Dataset: {dataset_name} \n"
            f"Total Questions: {total_questions} \n"
            f"Answered Questions: {len(pred_ans)} \n"
            f"Accuracy: {accuracy} \n"
            f"Mean Querying Time: {elapsed_time} \n"
        )
