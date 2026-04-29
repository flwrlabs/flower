import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionPipeline
from flwr_datasets import FederatedDataset
from peft import LoraConfig, get_peft_model, PeftModel
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fds = None


def collate_fn(batch):
    pixel_values = torch.stack([torch.as_tensor(item["pixel_values"]) for item in batch])
    return {"pixel_values": pixel_values}

def load_data(
        partition_id: int,
        num_partitions: int,
        image_size: int = 64,
        batch_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    """Load Oxford Flowers data for diffusion model training."""

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="nkirschi/oxford-flowers",
            partitioners={"train": partitioner}
        )

    partition = fds.load_partition(partition_id)

    # --- Image preprocessing for RGB flowers ---
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def transform_function(examples):
        key = "image" if "image" in examples else "img"
        images = []
        for img in examples[key]:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")  # ensure 3 channels
            images.append(transform(img))
        return {"pixel_values": images}

    partition = partition.map(
        transform_function,
        batched=True,
        remove_columns=list(partition.column_names)
    )

    # Split into train/test (20% test)
    partition_train_test = partition.train_test_split(test_size=0.20, seed=42)

    # Limit dataset size for quick federated demo
    def limit_dataset(dataset, n_samples, seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % 2**32
        rng = np.random.default_rng(seed)
        n = min(n_samples, len(dataset))
        indices = rng.choice(len(dataset), size=n, replace=False)
        return dataset.select(indices)

    partition_train_test["train"] = limit_dataset(partition_train_test["train"], 400)
    partition_train_test["test"] = limit_dataset(partition_train_test["test"], 80)

    trainload = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    testload = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"Partition {partition_id}: {len(partition_train_test['train'])} training samples, "f"{len(partition_train_test['test'])} test samples")

    return trainload, testload


def get_lora_model(base_model: str, device: torch.device):
    """Load Stable Diffusion model with memory-optimized LoRA adapters."""
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    pipe = enable_memory_efficient_attention(pipe)
    pipe.to(device)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Only attention layers
        lora_dropout=0.0,
        bias="none",
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    print(f"Trainable parameters: {sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)}")
    return pipe, torch_dtype


def enable_memory_efficient_attention(pipe):
    """Enable memory-efficient attention mechanisms."""
    try:
        pipe.unet.set_use_memory_efficient_attention_xformers(True)
    except:
        try:
            pipe.unet.enable_attention_slicing()
        except:
            pass

    # Enable CPU offloading for VAE and text encoder
    try:
        pipe.vae.enable_slicing()
        pipe.enable_attention_slicing()
    except:
        pass

    return pipe

def train_lora_step(pipe, dataloader, device, model_dtype):
    """Perform a single memory-efficient LoRA update."""
    pipe.unet.train()
    pipe.unet.requires_grad_(True)

    # Only optimize LoRA parameters
    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=1e-5)

    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        images = batch["pixel_values"]
        if isinstance(images, list):
            images = torch.stack(images)

        images = images.to(device, dtype=model_dtype)
        optimizer.zero_grad()

        # Encode images to latents (diffusion model training)
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

        # Sample random timestep and noise
        timestep = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
        noise = torch.randn_like(latents, dtype=model_dtype)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

        # UNet forward pass (use empty text embeddings for unconditional training)
        text_embeddings = torch.zeros(images.size(0), 77, 768, device=device, dtype=model_dtype)
        noise_pred = pipe.unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample

        loss = nn.functional.mse_loss(noise_pred, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_lora_step(pipe, dataloader, device, model_dtype):
    """Evaluate LoRA adapters with minimal memory usage."""
    pipe.unet.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]

            # Handle both Tensor and list cases
            if isinstance(pixel_values, torch.Tensor):
                images = pixel_values.to(device, dtype=model_dtype)
            elif isinstance(pixel_values, list):
                images = torch.stack(pixel_values).to(device, dtype=model_dtype)
            else:
                raise TypeError(f"Unexpected batch type: {type(pixel_values)}")

            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            timestep = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(latents, dtype=model_dtype)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

            text_emb = torch.zeros(images.size(0), 77, 768, device=device, dtype=model_dtype)
            noise_pred = pipe.unet(noisy_latents, timestep, encoder_hidden_states=text_emb).sample

            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

def generate_image(context, device, torch_dtype):
    base_model = context.run_config["base-model"]
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    base_path = "final_lora_model"
    current_dir = os.getcwd()
    lora_path = os.path.join(current_dir, base_path)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to(device)

    prompt = context.run_config["prompt"]
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image.save("federated_diffusion_sample.png")
    print("Image generated and saved as 'federated_diffusion_sample.png'")


