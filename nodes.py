import logging
import os
import time
from io import BytesIO

import numpy as np
import torch
from aiohttp import web
from PIL import Image

import comfy.model_base
import comfy.model_management
import comfy.samplers
import folder_paths
from comfy.weight_adapter import adapter_maps
from comfy_api.latest import io
from server import PromptServer

from .dataset import build_dataset, cached_cond, parse_dataset_paths, parse_resolutions
from .trainer import TARGET_PRESETS, sample_requests, train

# selected by default when installed, the default 8 steps at cfg 1 need a distilled LoRA
DISTILLED_LORA = "8-steps-dmd2-lora.safetensors"

# training progress of the current run: loss history, validation and sample previews, sent to every open tab and
# loaded by tabs opened later. Written by the training loop, one trainer node runs at a time
progress = {"run": "", "step": 0, "ema": None, "losses": [], "validation": [], "validation_steps": [], "samples": []}


def send_progress(**extra):
    """Broadcasts a delta for the node's loss graph and preview flipper. Loss values ride along incrementally as
    extra {loss_start, losses}; the rest is resent whole, it is always small."""
    state = {"run": progress["run"], "step": progress["step"], "ema": progress["ema"], "loss_start": len(progress["losses"]), "losses": [],
             "validation": progress["validation"], "validation_steps": progress["validation_steps"], "samples": progress["samples"]}
    state.update(extra)
    PromptServer.instance.send_sync("ez_moss.progress", state)


def reset_progress(samples_dir):
    """Called when a run queues: clears the graph. Previews of the same output_name stay on disk in the .samples
    folder and fill the flipper right away."""
    entries = []
    if os.path.isdir(samples_dir):
        for name in os.listdir(samples_dir):
            if name.startswith("preview_") and name.endswith(".png") and name[len("preview_"):-4].isdigit():
                entries.append({"step": int(name[len("preview_"):-4]), "file": name, "dir": samples_dir})
    entries.sort(key=lambda entry: entry["step"])
    progress.clear()
    progress.update(run=str(time.time()), step=0, ema=None, losses=[], validation=[], validation_steps=[], samples=entries)
    send_progress()


def report(step, new_losses, ema, validation, validation_steps):
    """Called by the training loop at every log interval: appends the new loss values and carries validation along."""
    progress["losses"].extend(round(v, 4) for v in new_losses)
    progress["step"] = step
    progress["ema"] = None if ema is None else round(float(ema), 4)
    progress["validation"] = [round(v, 4) for v in validation]
    progress["validation_steps"] = list(validation_steps)
    send_progress(loss_start=len(progress["losses"]) - len(new_losses), losses=[round(v, 4) for v in new_losses])


def show_preview(image, step, samples_dir):
    """Saves a training preview into the run's .samples folder and shows it on every open trainer node."""
    buffer = BytesIO()
    Image.fromarray((image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(buffer, "PNG")
    name = f"preview_{step:06d}.png"
    try:
        os.makedirs(samples_dir, exist_ok=True)
        with open(os.path.join(samples_dir, name), "wb") as f:
            f.write(buffer.getvalue())
    except OSError:
        logging.warning(f"[MOSS Trainer] Preview at step {step} could not be saved to {samples_dir}, skipping it")
        return
    if not any(entry["file"] == name for entry in progress["samples"]):
        progress["samples"].append({"step": step, "file": name, "dir": samples_dir})
    progress["step"] = step
    send_progress()


@PromptServer.instance.routes.post("/ez_moss/sample")
async def sample_route(request):
    settings = await request.json()
    if settings["lora"] != "None" and folder_paths.get_full_path("loras", settings["lora"]) is None:
        return web.json_response({"error": f"LoRA not found: {settings['lora']}"}, status=400)
    sample_requests.append({k: settings[k] for k in ("width", "height", "steps", "cfg", "sampler", "scheduler", "lora", "lora_strength")})
    return web.json_response({})


@PromptServer.instance.routes.get("/ez_moss/progress")
async def progress_route(request):
    # full state for tabs opened mid-run, the websocket only carries deltas
    return web.json_response(progress)


@PromptServer.instance.routes.get("/ez_moss/sample/image")
async def sample_file_route(request):
    # serves recorded previews only, the index is the only lookup
    try:
        index = int(request.query.get("index", -1))
    except ValueError:
        raise web.HTTPNotFound
    samples = progress["samples"]
    if 0 <= index < len(samples):
        path = os.path.join(samples[index]["dir"], samples[index]["file"])
        if os.path.isfile(path):
            return web.FileResponse(path)
    raise web.HTTPNotFound


class EZMOSSTrainLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        schedulers = comfy.samplers.KSampler.SCHEDULERS
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
                io.Combo.Input("algorithm", options=list(adapter_maps), default="LoRA",
                               tooltip="Adapter to train: LoRA (low-rank matrices, the usual default), LoHa (half-rank hadamard decomposition), LoKr (Kronecker product, fewer trainable parameters at the same rank), "
                                       "OFT (orthogonal rotations, keeps the network's function shape; rank sizes its blocks instead of scaling the update). Muon and Prodigy train LoRA, LoHa and LoKr, not OFT."),
                io.Int.Input("rank", default=8, min=1, max=256, tooltip="LoRA/LoHa rank. For LoKr and OFT it is the factorization / block size hint."),
                io.Float.Input("alpha", default=8.0, min=0.0, max=256.0, step=0.5, tooltip="Scale is alpha / rank, so raising it strengthens every step much like the learning rate does. For OFT this is the constraint."),
                io.Combo.Input("target_layers", options=list(TARGET_PRESETS), default="attn-mlp", tooltip="Transformer block linears to train. attn-mlp matches OneTrainer's MOSS LoRA preset."),
                io.Int.Input("steps", default=1500, min=1, max=1000000, tooltip="Optimizer steps."),
                io.Float.Input("learning_rate", default=2e-4, min=1e-7, max=1.0, step=1e-6, round=False),
                io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="Images per step, drawn from one resolution bucket."),
                io.Int.Input("grad_accumulation", default=1, min=1, max=256, tooltip="Batches accumulated per optimizer step. Raises the effective batch without more VRAM."),
                io.String.Input("resolution", default="512", tooltip="Training area is resolution x resolution pixels. A comma separated list like 512,768,1024 trains all of them: each image is assigned to one of the resolutions. "
                                "512 is comfortable on 8 GB, 768+ needs more VRAM or a larger reserve offload, and buckets far above 512 make steps several times more expensive."),
                io.Boolean.Input("bucketing", default=True, tooltip="Aspect ratio buckets with the same pixel area. Off center-crops everything to a square."),
                io.Combo.Input("optimizer", options=["AdamW", "Muon", "Prodigy", "Adafactor", "SGD"], default="Muon",
                               tooltip="Muon orthogonalises each update of the LoRA matrices, its step is scaled to AdamW's so the same learning rates apply. It trains LoRA, LoHa and LoKr, not OFT. "
                                       "Prodigy is Prodigy + schedule-free (prodigy-plus-schedule-free): it finds the step size itself, use learning_rate 1.0, the constant scheduler and max_grad_norm 0. "
                                       "Its estimate is fed by how well the gradient lines up with the weight displacement, so with a fresh sigma per step it can sit on its d0 floor for a while before it starts moving. Watch d in the console log."),
                io.Float.Input("d0", default=1e-4, min=1e-8, max=1.0, step=1e-8, round=False,
                               tooltip="Prodigy only: its first step-size estimate, and its floor, since d only grows from here. The library default 1e-6 stalls and barely learns here, 1e-4 picks up like AdamW at 3e-4 with batch_size 2."),
                io.Float.Input("d_coef", default=1.0, min=0.01, max=10.0, step=0.05, round=False,
                               tooltip="Prodigy only: multiplier on the estimated step size, the library's preferred tuning knob. Above 1 pushes the step size up sooner, below 1 for stability."),
                io.Int.Input("prodigy_steps", default=0, min=0, max=100000,
                             tooltip="Prodigy only: freeze the step-size estimate after N steps and free its memory. Stops the found step size from drifting late in a run, 0 keeps adapting."),
                io.Float.Input("schedulefree_c", default=0.0, min=0.0, max=200.0, step=1.0,
                               tooltip="Prodigy only: averaging strength of the schedule-free point that gets saved. 0 follows the library rule; short runs with small batches, this trainer's usual case, benefit from 6-12."),
                io.Combo.Input("lr_scheduler", options=["constant", "cosine", "linear"], default="cosine",
                               tooltip="Cosine eases the learning rate down to 0 over the run, linear decays it evenly, constant never changes it. Prodigy wants constant."),
                io.Int.Input("warmup_steps", default=100, min=0, max=100000,
                             tooltip="Steps that ramp the learning rate up from 0 before the schedule takes over. Steadies the first steps at higher learning rates."),
                io.Float.Input("weight_decay", default=0.01, min=0.0, max=1.0, step=0.001,
                               tooltip="L2 penalty on the trained weights, keeps them from growing without bound on long runs. 0 keeps the update purely driven by the loss."),
                io.Float.Input("max_grad_norm", default=1.0, min=0.0, max=100.0, step=0.1, tooltip="Gradient clipping, 0 disables."),
                io.Float.Input("caption_dropout", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Chance to train a sample with the empty prompt."),
                io.Combo.Input("timestep_distribution", options=["logit_normal", "uniform"], default="logit_normal"),
                io.Float.Input("timestep_shift", default=1.0, min=0.1, max=20.0, step=0.1, tooltip="Shifts training noise levels towards high noise (>1) like the sampling shift. 1.0 matches OneTrainer."),
                io.Int.Input("timestep_every", default=1, min=1, max=100000,
                             tooltip="Steps that share the same noise level. 1 draws a new level every step. Holding it for several steps keeps the gradient direction consistent, "
                                     "which Prodigy's step-size estimate needs to find a step size, at the cost of a coarser noise-level curriculum."),
                io.Boolean.Input("gradient_checkpointing", default=True, tooltip="Recompute block activations in the backward pass. Needed for 6-8 GB."),
                io.Float.Input("vram_reserve_gb", default=2.5, min=0.0, max=64.0, step=0.25,
                               tooltip="Legacy model loading only (ComfyUI started without dynamic VRAM): VRAM kept free for activations and gradients, weights that do not fit in the rest are streamed from RAM. Dynamic VRAM pages weights by itself and ignores this."),
                io.Int.Input("save_every", default=250, min=0, max=1000000, tooltip="Save an intermediate file every N steps, 0 only saves at the end."),
                io.Combo.Input("save_dtype", options=["bf16", "fp32"], default="bf16",
                               tooltip="Precision of the saved files. bf16 is half the size and how inference reads them; fp32 keeps full precision for later edits or fp32 training runs."),
                io.Int.Input("log_every", default=10, min=1, max=10000,
                             tooltip="Console log interval in steps. Every report also updates the loss graph and numbers on the node."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Seed of the noise, sampling and bucket order. The previews reuse it, so a preview at a given step is reproducible."),
                io.String.Input("cache_dir", default="", tooltip="Where latents and text encodings are cached. Empty uses a .moss_cache folder inside each dataset folder."),
                io.Int.Input("validation_every", default=250, min=0, max=1000000,
                             tooltip="Log a validation loss on up to 4 dataset images at fixed noise levels and noise every N steps, a much steadier progress signal than the training loss. 0 disables."),
                io.Boolean.Input("compile", default=True,
                                 tooltip="torch.compile the transformer blocks: about 25% faster and about 1 GB less VRAM, the first step takes about a minute longer. Needs triton (triton-windows on Windows), skipped with a warning when it is missing."),
                io.Int.Input("text_batch", default=8, min=1, max=64,
                             tooltip="Captions encoded together while caching text. 8 is about 1.6x faster than 1 on an 8 GB card and gives the same result, "
                                     "larger batches are not faster. Lower it if caching runs out of memory, 1 encodes one caption at a time."),
                io.Boolean.Input("resume", default=False,
                                 tooltip="Continue the run saved under output_name from its last save, interruption or end, using <output_name>.training_state. "
                                         "Needs the same settings, steps can be raised to train longer and the seed of the saved run is used."),
                io.Int.Input("preview_every", default=250, min=0, max=1000000,
                             tooltip="Sample an image before training, every N steps and at the end, shown on the node (page back through earlier ones with the arrows below it) and saved as preview_<step>.png "
                                     "in .samples/<output_name> inside the first dataset folder, next to its .moss_cache. 0 only samples on the 'sample now' button."),
                io.String.Input("preview_prompt", multiline=True, default="", tooltip="Prompt of the previews, empty uses the first training caption. Encoded when the run starts."),
                io.String.Input("preview_negative", multiline=True, default="", tooltip="Negative prompt of the previews, only used with a cfg other than 1. Encoded when the run starts."),
                io.Int.Input("preview_width", default=1024, min=64, max=4096, step=16,
                             tooltip="Preview size in pixels. Sampling and decoding sizes above 1024 need spare VRAM; running out of it skips the preview, training continues."),
                io.Int.Input("preview_height", default=1024, min=64, max=4096, step=16,
                             tooltip="Preview size in pixels. Sampling and decoding sizes above 1024 need spare VRAM; running out of it skips the preview, training continues."),
                io.Int.Input("preview_steps", default=8, min=1, max=1000, tooltip="8 with cfg 1 suits an 8-step distilled LoRA, without one use about 20 and cfg 4."),
                io.Float.Input("preview_cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, tooltip="1 skips the negative prompt and takes half the time per step."),
                io.Combo.Input("preview_sampler", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("preview_scheduler", options=schedulers, default="radiance_shift" if "radiance_shift" in schedulers else "simple"),
                io.Combo.Input("preview_lora", options=loras, default=DISTILLED_LORA if DISTILLED_LORA in loras else "None",
                               tooltip="LoRA applied on top of the one in training, for the previews only, e.g. a distilled LoRA for fast sampling. It does not affect training or the saved LoRA."),
                io.Float.Input("preview_lora_strength", default=1.0, min=-100.0, max=100.0, step=0.01,
                               tooltip="Strength of preview_lora, 1.0 applies it as trained. 0 disables it without changing the combo."),
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
                resolution, bucketing, optimizer, d0, d_coef, prodigy_steps, schedulefree_c, lr_scheduler, warmup_steps, weight_decay, max_grad_norm, caption_dropout, timestep_distribution, timestep_shift,
                timestep_every, gradient_checkpointing, vram_reserve_gb, save_every, save_dtype, log_every, seed, cache_dir, validation_every, compile,
                text_batch, resume, preview_every, preview_prompt, preview_negative, preview_width, preview_height, preview_steps, preview_cfg, preview_sampler,
                preview_scheduler, preview_lora, preview_lora_strength) -> io.NodeOutput:
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
        # one canonical form, so a resume compares equal whatever the widget sent: "512" matches a saved int, and
        # multi-resolution lists are always ordered the same way
        resolution = parse_resolutions(resolution)
        resolution = resolution[0] if len(resolution) == 1 else ",".join(map(str, resolution))
        cache_dir = os.path.abspath(cache_dir.strip()) if cache_dir.strip() else ""
        # previews live next to the dataset's .moss_cache, one subfolder per output name so reruns and resumes keep theirs
        samples_dir = os.path.abspath(os.path.join(folders[0][0], ".samples", os.path.relpath(save_path, lora_dir)))
        reset_progress(samples_dir)
        with torch.inference_mode(False):
            items = build_dataset(clip, vae, model.model.latent_format, folders, resolution, bucketing, cache_dir, text_batch)
            empty_cond = cached_cond(clip, "", cache_dir, folders[0][0]) if caption_dropout > 0 else None
            positive = cached_cond(clip, preview_prompt.strip(), cache_dir, folders[0][0]) if preview_prompt.strip() else items[0]["cond"]
            negative = cached_cond(clip, preview_negative.strip(), cache_dir, folders[0][0])
            # free the text encoder and VAE, the transformer needs the VRAM and pinned RAM while training
            comfy.model_management.unload_model_and_clones(clip.patcher)
            comfy.model_management.unload_model_and_clones(vae.patcher)
            cfg = {"algorithm": algorithm, "rank": rank, "alpha": alpha, "target": target_layers, "steps": steps, "learning_rate": learning_rate,
                   "batch_size": batch_size, "grad_accumulation": grad_accumulation, "resolution": resolution, "bucketing": bucketing,
                   "optimizer": optimizer, "d0": d0, "d_coef": d_coef, "prodigy_steps": prodigy_steps, "schedulefree_c": schedulefree_c,
                   "lr_scheduler": lr_scheduler, "warmup_steps": warmup_steps, "weight_decay": weight_decay,
                   "max_grad_norm": max_grad_norm, "caption_dropout": caption_dropout, "timestep_distribution": timestep_distribution,
                   "timestep_shift": timestep_shift, "timestep_every": timestep_every, "gradient_checkpointing": gradient_checkpointing, "vram_reserve_gb": vram_reserve_gb,
                   "save_every": save_every, "save_dtype": save_dtype, "log_every": log_every, "seed": seed, "validation_every": validation_every, "compile": compile}
            preview = {"every": preview_every, "width": preview_width, "height": preview_height, "steps": preview_steps, "cfg": preview_cfg, "sampler": preview_sampler, "scheduler": preview_scheduler,
                       "lora": preview_lora, "lora_strength": preview_lora_strength, "positive": positive, "negative": negative, "vae": vae,
                       "dir": samples_dir, "report": report, "show": show_preview}
            logging.info(f"[MOSS Trainer] Saving to {save_path}*.safetensors")
            with torch.enable_grad():
                lora, loss_map = train(model.clone(), items, empty_cond, cfg, save_path, preview, resume)
        return io.NodeOutput(lora, loss_map, steps)

