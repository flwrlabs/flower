"""quickstart-catboost: A Flower / CatBoost app."""

import json

from catboost import CatBoostClassifier, Pool
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecord, Context, Message, RecordDict

from quickstart_catboost.task import (
    convert_to_catboost,
    convert_to_model_dict,
    load_data,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:

    # Load partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    (X_train, y_train), (X_test, y_test), cat_features = load_data(
        partition_id, num_partitions
    )

    # Instantiate local model
    iterations = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    depth = context.run_config["depth"]
    cbc = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=42,
        verbose=0,
        cat_features=cat_features,
    )

    # Load global model
    # Note: In the first round, the global model is empty since no trees boosted yet.
    global_model_dict = msg.content["gmodel"]["model"]
    cbc_init = convert_to_catboost(global_model_dict) if global_model_dict else None

    # Local training
    cbc.fit(X_train, y_train, init_model=cbc_init)

    # Evaluation
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
    metrics = cbc.eval_metrics(eval_pool, metrics=["AUC"])
    auc = metrics["AUC"][-1]

    # Extract boosted trees
    model_dict = convert_to_model_dict(cbc)
    num_trees = len(model_dict["oblivious_trees"])
    # Extract the last N=iterations trees for sever aggregation
    model_dict["oblivious_trees"] = model_dict["oblivious_trees"][
        num_trees - iterations : num_trees
    ]
    model_dict_b = json.dumps(model_dict).encode("utf-8")

    # Construct reply message
    metric_and_model_record = ConfigRecord({"AUC": auc, "model_dict": model_dict_b})
    content = RecordDict({"metric_and_model": metric_and_model_record})
    return Message(content=content, reply_to=msg)
