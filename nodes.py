import logging
import os

import torch

import comfy.model_base
import comfy.model_management
import folder_paths
from comfy.weight_adapter import adapter_maps
from comfy_api.latest import io

from .dataset import build_dataset, cached_cond, parse_dataset_paths
from .pairs import ReferencePairs
from .trainer import TARGET_PRESETS, train


class EZMOSSTrainLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EZMOSSTrainLoRA",
            display_name="EZ MOSS-Image LoRA Trainer",
            category="model/training",
            description="Trains a LoRA/LoHa/LoKr/OFT on the MOSS-Image transformer from folders of image + .txt caption pairs. "
                        "Latents and text encodings are cached to disk. Works with bf16 and int8/convrot MOSS files; "
                        "the base weights stay quantized and frozen.",
            is_output_node=True,
            inputs=[
                io.Model.Input("model", tooltip="MOSS-Image diffusion model. An int8 convrot file keeps the transformer around 5.4 GB."),
                io.Clip.Input("clip", tooltip="MOSS-Image text encoder (Qwen3.5-9B), only used for captions that are not cached yet."),
                io.Vae.Input("vae", tooltip="Flux2 VAE, only used for images that are not cached yet."),
                io.String.Input("dataset_paths", multiline=True, default="",
                                tooltip="One folder per line. Each image needs a same-named .txt caption next to it. Add `| N` after a folder to repeat it N times per epoch, e.g. C:/data/style | 2"),
                io.String.Input("output_name", default="moss_lora", tooltip="File name in the first loras folder. Intermediate saves get a _<step> suffix."),
                io.Combo.Input("algorithm", options=list(adapter_maps), default="LoRA"),
                io.Int.Input("rank", default=16, min=1, max=256, tooltip="LoRA/LoHa rank. For LoKr and OFT it is the factorization / block size hint."),
                io.Float.Input("alpha", default=1.0, min=0.0, max=256.0, step=0.5, tooltip="Scale is alpha / rank. OneTrainer's default 1.0 pairs with learning rate 3e-4. For OFT this is the constraint."),
                io.Combo.Input("target_layers", options=list(TARGET_PRESETS), default="attn-mlp", tooltip="Transformer block linears to train. attn-mlp matches OneTrainer's MOSS LoRA preset."),
                io.Int.Input("steps", default=1500, min=1, max=1000000, tooltip="Optimizer steps."),
                io.Float.Input("learning_rate", default=3e-4, min=1e-7, max=1.0, step=1e-6, round=False),
                io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="Images per step, drawn from one resolution bucket."),
                io.Int.Input("grad_accumulation", default=1, min=1, max=256, tooltip="Batches accumulated per optimizer step. Raises the effective batch without more VRAM."),
                io.Int.Input("resolution", default=512, min=256, max=2048, step=64, tooltip="Training area is resolution x resolution pixels. 512 is comfortable on 8 GB, 768+ needs more VRAM or a larger reserve offload."),
                io.Boolean.Input("bucketing", default=True, tooltip="Aspect ratio buckets with the same pixel area. Off center-crops everything to a square."),
                io.Combo.Input("optimizer", options=["AdamW", "Prodigy", "Adafactor", "SGD"], default="AdamW",
                               tooltip="Prodigy is Prodigy + schedule-free (prodigy-plus-schedule-free): it finds the step size itself, use learning_rate 1.0 and the constant scheduler."),
                io.Combo.Input("lr_scheduler", options=["constant", "cosine", "linear"], default="constant"),
                io.Int.Input("warmup_steps", default=50, min=0, max=100000),
                io.Float.Input("weight_decay", default=0.01, min=0.0, max=1.0, step=0.001),
                io.Float.Input("max_grad_norm", default=1.0, min=0.0, max=100.0, step=0.1, tooltip="Gradient clipping, 0 disables."),
                io.Float.Input("caption_dropout", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Chance to train a sample with the empty prompt."),
                io.Combo.Input("timestep_distribution", options=["logit_normal", "uniform"], default="logit_normal"),
                io.Float.Input("timestep_shift", default=1.0, min=0.1, max=20.0, step=0.1, tooltip="Shifts training noise levels towards high noise (>1) like the sampling shift. 1.0 matches OneTrainer."),
                io.Boolean.Input("gradient_checkpointing", default=True, tooltip="Recompute block activations in the backward pass. Needed for 6-8 GB."),
                io.Float.Input("vram_reserve_gb", default=2.5, min=0.0, max=64.0, step=0.25,
                               tooltip="Legacy model loading only (ComfyUI started without dynamic VRAM): VRAM kept free for activations and gradients, weights that do not fit in the rest are streamed from RAM. Dynamic VRAM pages weights by itself and ignores this."),
                io.Int.Input("save_every", default=250, min=0, max=1000000, tooltip="Save an intermediate file every N steps, 0 only saves at the end."),
                io.Combo.Input("save_dtype", options=["bf16", "fp32"], default="bf16"),
                io.Int.Input("log_every", default=10, min=1, max=10000, tooltip="Console log interval in steps."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input("cache_dir", default="", tooltip="Where latents and text encodings are cached. Empty uses a .moss_cache folder inside each dataset folder."),
                io.Int.Input("validation_every", default=100, min=0, max=1000000,
                             tooltip="Log a validation loss on up to 4 dataset images at fixed noise levels and noise every N steps, a much steadier progress signal than the training loss. 0 disables."),
                io.Boolean.Input("compile", default=True,
                                 tooltip="torch.compile the transformer blocks: about 25% faster and about 1 GB less VRAM, the first step takes about a minute longer. Needs triton (triton-windows on Windows), skipped with a warning when it is missing."),
                io.Boolean.Input("reference_training", default=False,
                                 tooltip="Train a reference adapter for EZ MOSS Apply References instead of a plain LoRA. Needs a .json sidecar next to each image (see the README) and captions "
                                         "from the 'reference fields (JSON)' preset: every step pairs an image with character or style references and leaves what they supply out of the caption."),
                io.Int.Input("text_batch", default=8, min=1, max=64,
                             tooltip="Captions encoded together while caching text. 8 is about 1.6x faster than 1 on an 8 GB card and gives the same result, "
                                     "larger batches are not faster. Lower it if caching runs out of memory, 1 encodes one caption at a time."),
                io.Boolean.Input("resume", default=False,
                                 tooltip="Continue the run saved under output_name from its last save, interruption or end, using <output_name>.training_state. "
                                         "Needs the same settings, steps can be raised to train longer and the seed of the saved run is used."),
            ],
            outputs=[
                io.Custom("LORA_MODEL").Output(display_name="lora", tooltip="Trained weights, connect to Load LoRA Model to try them."),
                io.Custom("LOSS_MAP").Output(display_name="loss_map", tooltip="Loss history for Plot Loss Graph."),
                io.Int.Output(display_name="steps"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, vae, dataset_paths, output_name, algorithm, rank, alpha, target_layers, steps, learning_rate, batch_size, grad_accumulation,
                resolution, bucketing, optimizer, lr_scheduler, warmup_steps, weight_decay, max_grad_norm, caption_dropout, timestep_distribution, timestep_shift,
                gradient_checkpointing, vram_reserve_gb, save_every, save_dtype, log_every, seed, cache_dir, validation_every, compile, reference_training, text_batch, resume) -> io.NodeOutput:
        if not isinstance(model.model, comfy.model_base.MOSSImage):
            raise ValueError(f"EZ MOSS Trainer needs a MOSS-Image model, got {type(model.model).__name__}.")
        if algorithm not in adapter_maps or target_layers not in TARGET_PRESETS or save_dtype not in ("bf16", "fp32"):
            raise ValueError("Invalid algorithm, target_layers or save_dtype.")

        lora_dir = os.path.abspath(folder_paths.get_folder_paths("loras")[0])
        save_path = os.path.abspath(os.path.join(lora_dir, output_name.strip()))
        if os.path.commonpath((lora_dir, save_path)) != lora_dir or save_path == lora_dir:
            raise ValueError(f"output_name must stay inside {lora_dir}")
        if save_path.endswith(".safetensors"):
            save_path = save_path[:-len(".safetensors")]

        folders = parse_dataset_paths(dataset_paths)
        cache_dir = os.path.abspath(cache_dir.strip()) if cache_dir.strip() else ""
        with torch.inference_mode(False):
            items = build_dataset(clip, vae, model.model.latent_format, folders, resolution, bucketing, cache_dir, text_batch, encode_captions=not reference_training)
            pairs = ReferencePairs(items) if reference_training else None
            if pairs is not None:
                pairs.cache_text(clip, text_batch)
            empty_cond = cached_cond(clip, "", cache_dir, folders[0][0]) if caption_dropout > 0 else None
            # free the text encoder and VAE, the transformer needs the VRAM and pinned RAM while training
            comfy.model_management.unload_model_and_clones(clip.patcher)
            comfy.model_management.unload_model_and_clones(vae.patcher)
            cfg = {"algorithm": algorithm, "rank": rank, "alpha": alpha, "target": target_layers, "steps": steps, "learning_rate": learning_rate,
                   "batch_size": batch_size, "grad_accumulation": grad_accumulation, "resolution": resolution, "bucketing": bucketing,
                   "optimizer": optimizer, "lr_scheduler": lr_scheduler, "warmup_steps": warmup_steps, "weight_decay": weight_decay,
                   "max_grad_norm": max_grad_norm, "caption_dropout": caption_dropout, "timestep_distribution": timestep_distribution,
                   "timestep_shift": timestep_shift, "gradient_checkpointing": gradient_checkpointing, "vram_reserve_gb": vram_reserve_gb,
                   "save_every": save_every, "save_dtype": save_dtype, "log_every": log_every, "seed": seed, "validation_every": validation_every, "compile": compile,
                   "reference_training": reference_training}
            logging.info(f"[MOSS Trainer] Saving to {save_path}*.safetensors")
            with torch.enable_grad():
                lora, loss_map = train(model.clone(), items, empty_cond, cfg, save_path, pairs, resume)
        return io.NodeOutput(lora, loss_map, steps)

