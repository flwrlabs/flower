"""Memory-optimized Flower client for LoRA fine-tuning of Stable Diffusion."""

import warnings
import torch
import gc
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from peft import get_peft_model_state_dict
from diffusion_example.task import get_lora_model, train_lora_step, load_data, evaluate_lora_step

warnings.filterwarnings("ignore", category=FutureWarning)

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    """Perform one local training step with memory optimizations."""

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    base_model = context.run_config["base-model"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = int(context.run_config.get("image-size", 64))
    batch_size = int(context.run_config.get("batch-size", 8))

    train_loader, _ = load_data(partition_id, num_partitions, image_size=image_size, batch_size=batch_size)

    pipe, model_dtype = get_lora_model(base_model, device)

    # Restore only LoRA parameters
    arrays = msg.content["arrays"]
    current_state = pipe.unet.state_dict()

    # Update only the LoRA parameters, keep base model frozen
    for key in arrays.keys():
        if key in current_state:
            current_state[key] = torch.tensor(arrays[key])

    pipe.unet.load_state_dict(current_state, strict=False)

    loss_train = train_lora_step(pipe, train_loader, device, model_dtype)

    # Extract only LoRA parameters for sending back
    lora_state_dict = get_peft_model_state_dict(pipe.unet)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = MetricRecord({
        "loss": float(loss_train),
        "num-examples": len(train_loader)
    })

    content = RecordDict({
        "arrays": ArrayRecord(lora_state_dict),
        "metrics": metrics
    })

    return Message(content=content, reply_to=msg)



@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate with memory optimizations."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    base_model = context.run_config["base-model"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    image_size = int(context.run_config.get("image-size", 64))
    batch_size = int(context.run_config.get("batch-size", 8))

    _, test_loader = load_data(partition_id, num_partitions, image_size=image_size, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe, model_dtype = get_lora_model(base_model, device)

    # Restore parameters
    arrays = msg.content["arrays"]
    current_state = pipe.unet.state_dict()
    for key in arrays.keys():
        if key in current_state:
            current_state[key] = torch.tensor(arrays[key])

    pipe.unet.load_state_dict(current_state, strict=False)

    loss_val = evaluate_lora_step(pipe, test_loader, device, model_dtype)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = MetricRecord({
        "loss": float(loss_val),
        "num-examples": len(test_loader)
    })

    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
