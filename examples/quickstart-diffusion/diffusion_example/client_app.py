import warnings
import torch
import gc
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from peft import get_peft_model_state_dict
from diffusion_example.task import get_lora_model, train_lora_step, load_data, evaluate_lora_step, train_with_opacus, \
    clip_lora_update, add_gaussian_noise, add_laplace_noise

warnings.filterwarnings("ignore", category=FutureWarning)

# Flower ClientApp
app = ClientApp()

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

    train_loader, _ = load_data(partition_id, num_partitions, image_size=context.run_config["image_size"], batch_size=context.run_config["batch_size"])
    pipe, model_dtype = get_lora_model(base_model, device)

    arrays = msg.content["arrays"]
    current_state = pipe.unet.state_dict()

    # Update only the LoRA parameters, keep base model frozen
    for key in arrays.keys():
        if key in current_state:
            current_state[key] = torch.tensor(arrays[key])
    pipe.unet.load_state_dict(current_state, strict=False)

    if context.run_config["use-sample-dp"]:
        dp_noise_multiplier = float(context.run_config["dp-noise-multiplier"])
        dp_max_grad_norm = float(context.run_config["dp-max-grad-norm"])
        dp_target_delta = float(context.run_config["dp-target-delta"])

        loss_train, epsilon = train_with_opacus(
            pipe, train_loader, device, model_dtype,
            epochs = epochs, noise_multiplier=dp_noise_multiplier,
            max_grad_norm=dp_max_grad_norm,
            delta=dp_target_delta,
        )
    else:
        loss_train = train_lora_step(pipe, epochs, train_loader, device, model_dtype)
        epsilon = None

    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    lora_state_dict = {
        k: v.detach().cpu()
        for k, v in lora_state_dict.items()
    }

    if context.run_config["use_output_dp"]:
        output_dp_mechanism = context.run_config["output_dp_mechanism"]
        output_dp_max_norm = context.run_config["output_dp_max_norm"]

        lora_state_dict = clip_lora_update(
            lora_state_dict,
            max_norm=output_dp_max_norm
        )

        if output_dp_mechanism == "gaussian":
            output_dp_sigma = context.run_config["output_dp_sigma"]
            lora_state_dict = add_gaussian_noise(
                lora_state_dict,
                sigma=output_dp_sigma, # output_dp_sigma = 0.2
                max_norm=output_dp_max_norm  # output_dp_max_norm = 0.6
            )
        elif output_dp_mechanism == "laplace":
            output_dp_epsilon = context.run_config["output_dp_epsilon"]
            lora_state_dict = add_laplace_noise(
                lora_state_dict,
                epsilon=output_dp_epsilon,
                max_norm=output_dp_max_norm
            )

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = MetricRecord({
        "loss": float(loss_train),
        "num-examples": len(train_loader)
    })

    if epsilon is not None:
        metrics["epsilon"] = float(epsilon)

    content = RecordDict({
        "arrays": ArrayRecord(lora_state_dict),
        "metrics": metrics
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
