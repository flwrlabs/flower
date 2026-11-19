import gc
import os
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline
from flwr_datasets import FederatedDataset
from peft import LoraConfig, get_peft_model
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
        batch_size: int = 2
) -> tuple[DataLoader, DataLoader]:
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="nkirschi/oxford-flowers",
            partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id)
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
            img = img.convert("RGB")
            images.append(transform(img))
        return {"pixel_values": images}

    partition = partition.map(
        transform_function,
        batched=True,
        remove_columns=list(partition.column_names)
    )
    partition_train_test = partition.train_test_split(test_size=0.20, seed=42)

    train_load = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_load = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"Partition {partition_id}: {len(partition_train_test['train'])} training samples, "f"{len(partition_train_test['test'])} test samples")
    return train_load, test_load

def get_lora_model(base_model: str, device: torch.device):
    torch_dtype = torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=False,
    )

    pipe = enable_memory_efficient_attention(pipe)
    pipe.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v"],  # Only attention layers
        lora_dropout=0.05,
        bias="none"
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.config.addition_embed_type = None
    pipe.unet.add_embedding = None

    print(f"Trainable parameters: {sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)}")
    return pipe, torch_dtype


def enable_memory_efficient_attention(pipe):
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

def clear_memory(device):
    if device == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect()


def train_lora_step(pipe, epochs, dataloader, device, model_dtype):
    pipe.unet.train()
    pipe.unet.requires_grad_(True)

    pipe.unet.enable_gradient_checkpointing()
    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]

    if device == "cuda":
        lr = 5e-6
        grad_clip = 0.3
        timestep_min, timestep_max = 100, 900
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        torch.cuda.empty_cache()
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cudnn.benchmark = False
    else:
        lr = 1e-4
        grad_clip = 1.0
        timestep_min, timestep_max = 0, pipe.scheduler.config.num_train_timesteps
        use_amp = False
        scaler = None

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=lr,
        weight_decay=1e-6,
        eps=1e-7 if device == "cuda" else 1e-8,
        foreach=True,  # Multi-tensor ops (saves memory)
        fused=(device == "cuda")  # Fused kernels on GPU
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for  batch_idx, batch in enumerate(dataloader):
            if isinstance(batch["pixel_values"], list):
                images = torch.stack([batch["pixel_values"][0]])  # Single image
            else:
                images = batch["pixel_values"][:1]  # First image only

            texts = batch.get("text", ["a photo"] * images.size(0))
            images = images.to(device, dtype=model_dtype, non_blocking=(device == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with (torch.no_grad()):
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

                if torch.isnan(latents).any():
                    print(f"NaN in latents at batch {batch_idx}, skipping")
                    continue

                if torch.isnan(latents).any():
                    torch.cuda.empty_cache()
                    continue

            if device == "cuda":
                timestep = torch.randint(
                    timestep_min, timestep_max,
                    (images.size(0),), device=device
                ).long()
            else:
                timestep = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (images.size(0),), device=device
                ).long()

            noise = torch.randn_like(latents, dtype=model_dtype, device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

            # Encode text PROPERLY
            with torch.no_grad():
                text_inputs = pipe.tokenizer(
                    texts,
                    max_length=pipe.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                text_embeddings = pipe.text_encoder(
                    text_inputs.input_ids
                )[0]

                if torch.isnan(text_embeddings).any():
                    print(f"NaN in text embeddings at batch {batch_idx}")
                    text_embeddings = torch.zeros_like(text_embeddings)

            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    noise_pred = pipe.unet(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states=text_embeddings,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample

            loss = F.mse_loss(
                noise_pred.float(),
                noise.float(),
                reduction="mean"
            )
            loss = torch.clamp(loss, max=5.0)

            if torch.isnan(loss) or torch.isinf(loss):
                clear_memory(device)
                continue

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # CRITICAL: Clip before step to prevent OOM
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            clear_memory(device)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_lora_step(pipe, dataloader, device, model_dtype):
    pipe.unet.to(device, dtype=model_dtype)
    pipe.vae.to(device, dtype=model_dtype)

    pipe.unet.eval()
    pipe.vae.eval()

    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                images = pixel_values.to(device, dtype=model_dtype)
            elif isinstance(pixel_values, list):
                images = torch.stack(pixel_values).to(device, dtype=model_dtype)
            else:
                raise TypeError(f"Unexpected batch type: {type(pixel_values)}")

            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            timestep = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (images.size(0),),
                device=device
            ).long()

            noise = torch.randn_like(latents, dtype=model_dtype)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
            cross_attention_dim = pipe.unet.config.cross_attention_dim

            text_embeddings = torch.zeros(
                images.size(0),
                pipe.tokenizer.model_max_length,
                cross_attention_dim,
                device=device,
                dtype=model_dtype
            )
            noise_pred = pipe.unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            psnr = 10.0 * torch.log10(1.0 / loss)
            total_psnr += psnr.item()

            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0, total_psnr / num_batches


def clip_lora_update(lora_state, max_norm):
    total_norm = torch.sqrt(
        sum(v.norm() ** 2 for v in lora_state.values())
    )
    scale = min(1.0, max_norm / (total_norm + 1e-6))
    return {k: v * scale for k, v in lora_state.items()}



def generate_image(
        device,
        model_dtype,
        base_model,
        prompt,
        negative_prompt,
        use_lora: bool,
        seed: int = 42,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=model_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=False,
    )

    pipe = enable_memory_efficient_attention(pipe)
    pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    # ---- Apply LoRA ONLY if use_lora=True ----
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v"],
            lora_dropout=0.05,
            bias="none"
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)

        base_path = "final_lora_model"
        current_dir = os.getcwd()

        lora_path = os.path.join(current_dir, base_path)
        pipe.load_lora_weights(lora_path)

        pipe.unet.config.addition_embed_type = None
        pipe.unet.add_embedding = None

        tag = "after_finetune"
    else:
        tag = "before_finetune"

    if use_lora:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
    out_name = f"federated_diffusion_{tag}.png"
    image.save(out_name)



