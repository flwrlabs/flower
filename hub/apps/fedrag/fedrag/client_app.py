"""fedrag: A Flower Federated RAG app."""

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from fedrag.retriever import Retriever

# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):

    node_id = context.node_id

    # Extract question
    question = str(msg.content["config"]["question"])
    question_id = str(msg.content["config"]["question_id"])

    # Extract corpus name
    corpus_name = str(msg.content["config"]["corpus_name"])

    # Initialize retrieval system
    retriever = Retriever()
    # Use the knn value for retrieving the closest-k documents to the query
    knn = int(msg.content["config"]["knn"])
    retrieved_docs = retriever.query_faiss_index(corpus_name, question, knn)

    # Create lists with the computed scores and documents
    scores = [doc["score"] for doc_id, doc in retrieved_docs.items()]
    documents = [doc["content"] for doc_id, doc in retrieved_docs.items()]
    print(
        "ClientApp: {} - Question ID: {} - Retrieved: {} documents.".format(
            node_id, question_id, len(documents)
        )
    )

    # Create reply record with retrieved documents.
    docs_n_scores = ConfigRecord(
        {
            "documents": documents,
            "scores": scores,
        }
    )
    reply_record = RecordDict({"docs_n_scores": docs_n_scores})

    # Return message
    return Message(reply_record, reply_to=msg)
