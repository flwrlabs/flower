import os
from logging import INFO

import torch
from peft import get_peft_model_state_dict, LoraConfig
from diffusionSecAgg.task import get_lora_model, generate_image

from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters, log
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.strategy import FedAvg
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = 0
    weighted_psnr = 0.0
    weighted_loss = 0.0

    for num_examples, m in metrics:
        total_examples += num_examples
        if "psnr" in m:
            weighted_psnr += num_examples * m["psnr"]
        if "loss" in m:
            weighted_loss += num_examples * m["loss"]
    aggregated = {}

    if total_examples > 0:
        if weighted_psnr > 0:
            aggregated["psnr"] = weighted_psnr / total_examples
        if weighted_loss > 0:
            aggregated["loss"] = weighted_loss / total_examples
    return aggregated

def fit_weighted_average(metrics):
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(losses) / sum(examples)}


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    base_model = context.run_config["base-model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe, model_dtype = get_lora_model(base_model, device)
    global_lora = get_peft_model_state_dict(pipe.unet)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])

    num_rounds = int(context.run_config["num-server-rounds"])
    parameters = ndarrays_to_parameters(global_lora)

    strategy = FedAvg(
        fraction_fit=fraction_train,
        min_fit_clients=2,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=parameters,
    )

    context = LegacyContext(
        context=context,
        config=ServerConfig(
            num_rounds=num_rounds,
            round_timeout=3600
        ),
        strategy=strategy,
    )

    log(INFO, f" Starting federated diffusion training for {num_rounds} rounds...")
    log(INFO, f" Using base model: {base_model}")
    log(INFO, f" Training LoRA parameters only ({len(global_lora)} layers)")

    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        max_weight=context.run_config["max-weight"],
    )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, context)

    save_dir = "final_lora_model"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(global_lora, os.path.join(save_dir, "adapter_model.bin"))
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.05,
        bias="none",
    )

    config.save_pretrained(save_dir)

    prompt = context.run_config["prompt"]
    negative_prop = context.run_config["negative_prompt"]

    generate_image(device, model_dtype, base_model, prompt, negative_prop, False)
    generate_image(device, model_dtype, base_model, prompt, negative_prop, True)



