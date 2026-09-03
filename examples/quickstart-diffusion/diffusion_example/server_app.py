import os
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from peft import get_peft_model_state_dict, LoraConfig
from diffusion_example.task import get_lora_model, generate_image
from flwr.common import log
from logging import INFO


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    base_model = context.run_config["base-model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe, model_dtype = get_lora_model(base_model, device)
    peft_state_dict = get_peft_model_state_dict(pipe.unet)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])

    initial_arrays = ArrayRecord(peft_state_dict)
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    log(INFO, f" Starting federated diffusion training for {num_rounds} rounds...")
    log(INFO, f" Using base model: {base_model}")
    log(INFO, f" Training LoRA parameters only ({len(peft_state_dict)} layers)")

    if context.run_config["use-sample-dp"]:
        log(INFO, f" Sample-level Differential Privacy enabled using Opacus (DP-SGD with gradient clipping and noise injection).")

    if context.run_config["use_output_dp"]:
        if context.run_config["output_dp_mechanism"]== "gaussian":
            log(INFO, f" Output-level Differential Privacy enabled using Gaussian noise for result perturbation.")
        else:
            log(INFO, f" Output-level Differential Privacy enabled using Laplace noise for result perturbation.")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=3600,
    )

    final_state = result.arrays.to_torch_state_dict()
    save_dir = "final_lora_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(final_state, os.path.join(save_dir, "adapter_model.bin"))
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v"],  # Only attention layers
        lora_dropout=0.05,
        bias="none",
    )
    config.save_pretrained(save_dir)
    print(f"Saved final LoRA model at: {save_dir}")
    generate_image(device, model_dtype, base_model, context.run_config["prompt"], context.run_config["negative_prompt"], False)
    generate_image(device, model_dtype, base_model, context.run_config["prompt"], context.run_config["negative_prompt"], True)
