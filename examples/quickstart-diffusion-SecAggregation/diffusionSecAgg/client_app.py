import warnings
import torch
import gc
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict, ConfigRecord
from peft import get_peft_model_state_dict
from diffusionSecAgg.task import get_lora_model, train_lora_step, load_data, evaluate_lora_step


from flwr.common import Context
from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
warnings.filterwarnings("ignore", category=FutureWarning)

# Flower ClientApp
app = ClientApp(mods=[
    secaggplus_mod,
],)

@app.train()
def train(msg: Message, context: Context) -> Message:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    base_model = context.run_config["base-model"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = context.run_config["local-epochs"]

    train_loader, _ = load_data(
        partition_id,
        num_partitions,
        image_size=context.run_config["image_size"],
        batch_size=context.run_config["batch_size"]
    )

    pipe, model_dtype = get_lora_model(base_model, device)
    loss_train = train_lora_step(pipe, epochs, train_loader, device, model_dtype)

    # Extract only LoRA parameters for sending back
    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    lora_state_dict = {
        k: v.detach().cpu()
        for k, v in lora_state_dict.items()
    }

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    content = RecordDict({
        "fitres.parameters": ArrayRecord(lora_state_dict),
        "fitres.num_examples": MetricRecord({
            "num_examples": len(train_loader),
        }),
        "fitres.metrics": ConfigRecord({
            "loss": float(loss_train),
        }),
        "fitres.status": ConfigRecord({
            "code": 0,           # 0 = OK
            "message": "OK",
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    base_model = context.run_config["base-model"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    _, test_loader = load_data(partition_id, num_partitions, image_size=context.run_config["image_size"], batch_size=context.run_config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe, model_dtype = get_lora_model(base_model, device)
    loss_val, psnr = evaluate_lora_step(pipe, test_loader, device, model_dtype)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    content = RecordDict({
        "evaluateres.loss": MetricRecord({
            "loss": float(loss_val),
        }),
        "evaluateres.num_examples": MetricRecord({
            "num_examples": len(test_loader),
        }),
        "evaluateres.metrics": ConfigRecord({
            "psnr": float(psnr),
        }),

        "evaluateres.status": ConfigRecord({
            "code": 0,
            "message": "OK",
        })
    })
    return Message(content=content, reply_to=msg)
