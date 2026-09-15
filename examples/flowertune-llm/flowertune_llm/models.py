import math

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def is_adapter_parameter(name: str) -> bool:
    """Return whether a state entry belongs to a PEFT LoRA-family adapter."""
    return "lora_" in name


def is_communicated_adapter_parameter(name: str, *, freeze_a: bool) -> bool:
    """Return whether an adapter state entry needs federated synchronization."""
    return is_adapter_parameter(name) and (not freeze_a or ".lora_B." in name)


def get_model(model_cfg: DictConfig, *, for_server: bool = False):
    """Load model with appropriate dtype and device map settings."""

    # Keep adapter initialization identical across clients and repeated runs.
    torch.manual_seed(int(getattr(model_cfg, "seed", 2026)))

    dtype = getattr(model_cfg, "dtype", "bfloat16")
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    load_in_4bit = bool(getattr(model_cfg, "load_in_4bit", False)) and not for_server
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(getattr(model_cfg, "bnb_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                getattr(model_cfg, "bnb_double_quant", True)
            ),
            bnb_4bit_compute_dtype=torch_dtype,
        )
        load_kwargs["device_map"] = {"": 0}
    elif hasattr(model_cfg, "device_map") and model_cfg.device_map:
        load_kwargs["device_map"] = model_cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        **load_kwargs,
    )

    if not load_in_4bit and getattr(model_cfg, "device_map", "") == "cpu":
        model = model.to("cpu")

    lora_cfg = getattr(model_cfg, "lora", {})
    if bool(lora_cfg.get("enabled", False)):
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        targets = lora_cfg.get(
            "target_modules",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        if isinstance(targets, str):
            targets = [target.strip() for target in targets.split(",") if target.strip()]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora_cfg.get("rank", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=list(targets),
                bias="none",
                use_rslora=bool(lora_cfg.get("use_rslora", False)),
                use_dora=bool(lora_cfg.get("use_dora", False)),
            ),
        )
        if bool(lora_cfg.get("freeze_a", False)):
            for name, parameter in model.named_parameters():
                if ".lora_A." in name:
                    parameter.requires_grad_(False)

    return model
