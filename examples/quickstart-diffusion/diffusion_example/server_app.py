"""Memory-optimized Flower server for federated LoRA fine-tuning."""
import os
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from peft import get_peft_model_state_dict, LoraConfig
from diffusion_example.task import get_lora_model, generate_image

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load base model and extract initial LoRA parameters only
    base_model = context.run_config["base-model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model just to get LoRA structure
    pipe, torch_dtype = get_lora_model(base_model, device)

    peft_state_dict = get_peft_model_state_dict(pipe.unet)
    initial_arrays = ArrayRecord(peft_state_dict)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configure training strategy
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start the federated training
    print(f"\n Starting federated diffusion training for {num_rounds} rounds...")
    print(f" Using base model: {base_model}")
    print(f" Training LoRA parameters only ({len(peft_state_dict)} layers)")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=7200,
    )

    final_state = result.arrays.to_torch_state_dict()
    save_dir = "final_lora_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(final_state, os.path.join(save_dir, "adapter_model.bin"))
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Only attention layers
        lora_dropout=0.0,
        bias="none",
    )
    config.save_pretrained(save_dir)
    print(f"Saved final LoRA model at: {save_dir}")
    generate_image(context, device, torch_dtype)
