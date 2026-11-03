import importlib.util
import json
import logging
import math
import os
import random
import time

import safetensors.torch
import torch
from prodigyplus import ProdigyPlusScheduleFree

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sample
from comfy.cli_args import args
import comfy.utils
import folder_paths
import node_helpers
from comfy.ldm.modules.attention import attention_pytorch
from comfy.weight_adapter import WeightAdapterBase, adapter_maps
from comfy.weight_adapter.bypass import BypassInjectionManager
from comfy.weight_adapter.loha import HadaWeight, LohaDiff
from comfy.weight_adapter.oft import OFTDiff
from comfy_extras.nodes_train import patch, unpatch

from .dataset import BucketSampler

TARGET_PRESETS = {
    "attn-mlp": {"q", "k", "v", "qkv", "o", "ffn_up_gate", "ffn_down"},
    "attn-only": {"q", "k", "v", "qkv", "o"},
    "mlp-only": {"ffn_up_gate", "ffn_down"},
}

# settings a resumed run may change, the rest must match the saved run
RESUME_FREE = {"steps", "seed", "save_every", "log_every", "validation_every", "vram_reserve_gb", "compile", "gradient_checkpointing"}

# preview settings from the node's sample button, picked up between training steps
sample_requests = []


class LohaBypass(LohaDiff):
    # core LohaDiff only patches weights, bypass mode needs the additive h(x) = x @ dW^T
    def h(self, x, base_out):
        diff = HadaWeight.apply(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, self.alpha / self.rank)
        return torch.nn.functional.linear(x, diff.to(x.dtype)) * self.multiplier


class RecomputeInBackward(torch.autograd.Function):
    """Runs a streamed op without saving anything but its input, and reruns it right at its backward.

    Streamed weights live in reused cast buffers or paged VRAM, so a weight saved in the forward is gone by the backward.
    Unlike a nested torch.utils.checkpoint, the input goes through save_for_backward, so the enclosing block checkpoint
    can still drop it. params are the op's adapter parameters, they make the output require grad when the input does not."""

    @staticmethod
    def forward(ctx, fn, x, *params):
        ctx.fn = fn
        ctx.autocast = torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type)
        ctx.save_for_backward(x)
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(x.device.type, dtype=ctx.autocast[1], enabled=ctx.autocast[0]):
            x = x.detach().requires_grad_(ctx.needs_input_grad[1])
            out = ctx.fn(x)
        torch.autograd.backward(out, grad)
        return (None, x.grad if ctx.needs_input_grad[1] else None) + (None,) * (len(ctx.needs_input_grad) - 2)


def recompute_in_backward(module, params):
    forward = module.forward

    # dynamo can't trace the custom Function with dynamic shapes, compiled blocks graph-break around it
    @torch.compiler.disable
    def recompute_forward(x):
        if not torch.is_grad_enabled() or not (x.requires_grad or params):
            return forward(x)
        return RecomputeInBackward.apply(forward, x, *params)

    # same convention as nodes_train.patch so unpatch restores it
    module.org_forward = forward
    module.forward = recompute_forward


def setup_adapters(mp, algorithm, rank, alpha, target):
    """Bypass adapters on the transformer block linears. Bypass keeps quantized (int8/convrot) base weights on their quantized path."""
    names = TARGET_PRESETS[target]
    manager = BypassInjectionManager()
    adapters = {}
    for name, module in mp.model.diffusion_model.layers.named_modules():
        if name.rsplit(".", 1)[-1] not in names or getattr(module, "weight", None) is None:
            continue
        key = f"diffusion_model.layers.{name}"
        adapter = adapter_maps[algorithm].create_train(module.weight, rank=rank, alpha=alpha).to(torch.float32)
        if isinstance(adapter, LohaDiff):
            adapter.__class__ = LohaBypass
        elif isinstance(adapter, OFTDiff):
            # the OFT norm constraint has a NaN gradient at exactly zero blocks
            adapter.oft_blocks.data.normal_(std=1e-6)
        adapter.train().requires_grad_(True)
        adapter.alpha.requires_grad_(False)
        manager.add_adapter(key, adapter)
        adapters[key] = adapter
    if not adapters:
        raise ValueError(f"No MOSS transformer layers matched the '{target}' target.")
    return adapters, manager.create_injections(mp.model)


def adapter_state_dict(adapters, dtype):
    dtype = node_helpers.string_to_torch_dtype(dtype)
    sd = {}
    for key, adapter in adapters.items():
        params = {name: p.detach().cpu() for name, p in adapter.named_parameters()}
        if key.endswith(".qkv") and "lora_up.weight" in params:
            # split fused qkv so the LoRA loads on both fused and separate q/k/v MOSS files
            base = key[:-len(".qkv")]
            for proj, up in zip(("q", "k", "v"), params["lora_up.weight"].chunk(3, dim=0)):
                sd[f"{base}.{proj}.lora_up.weight"] = up.to(dtype).contiguous()
                sd[f"{base}.{proj}.lora_down.weight"] = params["lora_down.weight"].to(dtype).clone()
                sd[f"{base}.{proj}.alpha"] = params["alpha"].float().clone()
            continue
        for name, p in params.items():
            sd[f"{key}.{name}"] = p.float() if name == "alpha" else p.to(dtype).contiguous()
    return sd


def save_lora(sd, path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safetensors.torch.save_file(sd, path, metadata={k: str(v) for k, v in metadata.items()})
    logging.info(f"[MOSS Trainer] Saved {path}")


class FusedAdapter:
    """Adapters on slices of one fused linear, a bypass hook takes a single adapter per module."""

    def __init__(self, slices):
        self.slices = slices

    def h(self, x, base_out):
        out = torch.zeros_like(base_out)
        for start, size, adapter in self.slices:
            out[..., start:start + size] = adapter.h(x, None)
        return out

    def g(self, y):
        return y


def lora_injections(mp, name, strength):
    """Bypass injections of a LoRA file. Core's bypass loader drops adapters on a slice of a weight, which is what the q, k and v
    keys of a LoRA become on fused qkv files."""
    lora = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True))
    # moved here because the hooks can't reach the adapters inside a FusedAdapter
    lora = {k: v.to(mp.load_device, mp.model.get_dtype_inference()) if v.ndim > 1 else v for k, v in lora.items()}
    manager = BypassInjectionManager()
    fused = {}
    for key, adapter in comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(mp.model, {})).items():
        if not isinstance(adapter, WeightAdapterBase):
            logging.warning(f"[MOSS Trainer] {name}: {key} is not an adapter, previews skip it")
        elif isinstance(key, tuple):
            adapter.multiplier = strength
            fused.setdefault(key[0], []).append((key[1][1], key[1][2], adapter))
        else:
            manager.add_adapter(key, adapter, strength)
    for key, slices in fused.items():
        manager.add_adapter(key, FusedAdapter(slices), strength)
    return manager.create_injections(mp.model)


def sample_image(mp, preview, seed):
    """One image from the adapters in training through the native samplers, as a [H, W, 3] tensor."""
    device, dtype = mp.load_device, mp.model.get_dtype_inference()
    latent_format = mp.model.latent_format
    latent = torch.zeros([1, latent_format.latent_channels, preview["height"] // latent_format.spacial_downscale_ratio, preview["width"] // latent_format.spacial_downscale_ratio],
                         device=comfy.model_management.intermediate_device())
    injections = lora_injections(mp, preview["lora"], preview["lora_strength"]) if preview["lora"] != "None" else []
    for injection in injections:
        injection.inject(mp)
    try:
        # eager, the blocks would recompile for the grad mode and the hooks of a preview
        with torch.no_grad(), torch.compiler.set_stance("force_eager"), torch.autocast(device.type, dtype=dtype, enabled=dtype in (torch.bfloat16, torch.float16)):
            samples = comfy.sample.sample(mp, comfy.sample.prepare_noise(latent, seed), preview["steps"], preview["cfg"], preview["sampler"], preview["scheduler"],
                                          [[preview["positive"][None], {}]], [[preview["negative"][None], {}]], latent, disable_pbar=True, seed=seed)
    finally:
        for injection in injections:
            injection.eject(mp)
    with torch.no_grad():
        return preview["vae"].decode(samples)[0]


def make_optimizer(name, params, lr, weight_decay, device):
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=device.type == "cuda")
    if name == "Muon":
        # orthogonalises each matrix's update; match_rms_adamw scales the step to AdamW's update size, so learning rates carry over
        return torch.optim.Muon(params, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
    if name == "Prodigy":
        # the library default d0=1e-6 never takes off here: with one random sigma and noise per batch the
        # per-step gradient aligns with the accumulated weights too little for d to grow, so the run trains
        # d0 times too slowly. 1e-4 picks up like AdamW at 3e-4 with batch_size 2.
        return ProdigyPlusScheduleFree(params, lr=lr, weight_decay=weight_decay, d0=1e-4)
    if name == "Adafactor":
        return torch.optim.Adafactor(params, lr=lr, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")


def make_lr_lambda(scheduler, warmup, steps):
    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        if scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if scheduler == "linear":
            return 1.0 - progress
        return 1.0
    return fn


def sample_sigmas(n, distribution, shift, generator, device):
    if distribution == "logit_normal":
        sigma = torch.randn(n, generator=generator, device=device).sigmoid()
    else:
        sigma = torch.rand(n, generator=generator, device=device)
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    # OneTrainer's discrete grid starts at 1/1000
    return sigma.clamp(1e-3, 1.0)


VALIDATION_SIGMAS = (0.25, 0.5, 0.75, 0.9)


def validation_images(items):
    """Up to 4 training images spread over the dataset."""
    targets = [i for i, item in enumerate(items) if item["repeats"]]
    n = min(4, len(targets))
    return [targets[round(i * (len(targets) - 1) / max(1, n - 1))] for i in range(n)]


def make_validation_set(items, indices, seed, device, dtype):
    """The validation images, each at fixed noise levels with fixed noise."""
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    val_set = []
    for index in indices:
        x0 = items[index]["latent"][None].to(device, torch.float32)
        context = items[index]["cond"][None].to(device, dtype)
        for sigma in VALIDATION_SIGMAS:
            val_set.append((x0, context, torch.randn(x0.shape, generator=generator, device=device), torch.tensor([sigma], device=device)))
    return val_set


def entry_loss(predict, entry):
    """Velocity MSE of one validation entry."""
    x0, context, noise, sigma = entry
    s = sigma.view(-1, 1, 1, 1)
    pred = predict((1 - s) * x0 + s * noise, sigma, context)
    return torch.nn.functional.mse_loss(pred.float(), noise - x0).item()


def _fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def train(mp, items, empty_cond, cfg, save_path, preview, resume=False):
    """Trains bypass adapters on the MOSS transformer. Returns (lora state dict, loss map).
    preview holds the sampling settings, conditioning, VAE and show callback of the previews.
    resume continues from the training state saved next to the LoRA files at their last save."""
    device = mp.load_device
    dtype = mp.model.get_dtype_inference()
    autocast = dtype in (torch.bfloat16, torch.float16)

    state_path = f"{save_path}.training_state"
    state = None
    if resume:
        if not os.path.isfile(state_path):
            raise ValueError(f"No training state to resume at {state_path}, it is written with every save of a run.")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        changed = [k for k in cfg if k not in RESUME_FREE and cfg[k] != state["cfg"].get(k)]
        if changed:
            raise ValueError("Resume needs the settings of the saved run, changed: " + ", ".join(f"{k} ({state['cfg'].get(k)} -> {cfg[k]})" for k in changed))
        cfg["seed"] = state["cfg"]["seed"]
        logging.info(f"[MOSS Trainer] Resuming {state_path} from step {state['step']}, seed {cfg['seed']}")
    start_step = state["step"] if state is not None else 0

    torch.manual_seed(cfg["seed"])
    adapters, injections = setup_adapters(mp, cfg["algorithm"], cfg["rank"], cfg["alpha"], cfg["target"])
    params = [p for a in adapters.values() for p in a.parameters() if p.requires_grad]
    if state is not None:
        for key, adapter in adapters.items():
            adapter.load_state_dict(state["adapters"][key])
    logging.info(f"[MOSS Trainer] {cfg['algorithm']} on {len(adapters)} layers ({cfg['target']}), {sum(p.numel() for p in params) / 1e6:.2f}M trainable params")

    comfy.model_management.load_models_gpu([mp], memory_required=int(cfg["vram_reserve_gb"] * 1024 ** 3))
    if mp.is_dynamic():
        logging.info(f"[MOSS Trainer] Dynamic VRAM: {mp.model_size() / 1024 ** 3:.2f} GB of transformer weights are paged in per block")
    else:
        loaded = mp.loaded_size() / 1024 ** 3
        total = mp.model_size() / 1024 ** 3
        logging.info(f"[MOSS Trainer] Transformer weights on GPU: {loaded:.2f} / {total:.2f} GB" + (" (rest streamed from RAM)" if loaded < total - 0.01 else ""))

    blocks = list(mp.model.diffusion_model.layers) if cfg["gradient_checkpointing"] else []
    sampler = BucketSampler(items, cfg["batch_size"], cfg["seed"])
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    val_images = validation_images(items)
    if state is not None:
        # saved indices point into the saved run's dataset, map them by image path so images can be added or removed
        index = {item["image"]: i for i, item in enumerate(items)}
        saved = state.get("images", [item["image"] for item in items])
        moved = {j: index[path] for j, path in enumerate(saved) if path in index}
        added = len(index.keys() - set(saved))
        if added or len(moved) < len(saved):
            logging.info(f"[MOSS Trainer] Dataset changed since the saved run: {added} images added, {len(saved) - len(moved)} removed. Added images join from the next epoch")
        sampler.rng.setstate(state["sampler"][0])
        sampler.epoch = state["sampler"][1]
        sampler.batches = [b for b in ([moved[j] for j in batch if j in moved] for batch in state["sampler"][2]) if b]
        saved_val = state.get("validation_images", val_images)
        kept = [moved[j] for j in saved_val if j in moved]
        if len(kept) == len(saved_val):
            val_images = kept
        else:
            logging.warning("[MOSS Trainer] A validation image was removed, validation uses new images and is not comparable with the earlier steps")
        rng.setstate(state["rng"])
        generator.set_state(state["generator"])
    model_sampling = mp.model.model_sampling
    # quantized checkpoints can pin int8 kitchen attention per block, which has no backward and takes no autocast dtypes in previews
    mp.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = lambda func, *args, **kwargs: attention_pytorch(*args, **kwargs)
    transformer_options = mp.model_options["transformer_options"].copy()
    transformer_options["prefetch_dynamic_vbars"] = mp.is_dynamic()

    def predict(xt, sigma, context):
        timestep = model_sampling.timestep(sigma).float()
        with torch.autocast(device.type, dtype=dtype, enabled=autocast):
            return mp.model.diffusion_model(xt.to(dtype), timestep, context=context, transformer_options=transformer_options)

    def state_dict():
        return adapter_state_dict(adapters, cfg["save_dtype"])

    def save_state(step):
        # fp32 weights, optimizer and sampling state, overwritten at every save so a run can be resumed from its last one
        with open(state_path + ".tmp", "wb") as f:
            torch.save({"cfg": cfg, "step": step, "adapters": {key: a.state_dict() for key, a in adapters.items()},
                        "optimizer": optimizer.state_dict(), "sampler": (sampler.rng.getstate(), sampler.epoch, sampler.batches), "rng": rng.getstate(),
                        "images": [item["image"] for item in items], "validation_images": val_images,
                        "generator": generator.get_state(), "loss": loss_history, "validation": validation, "validation_steps": validation_steps, "ema": ema}, f)
            # on disk before the rename, so a power loss right after a save keeps a complete state
            f.flush()
            os.fsync(f.fileno())
        os.replace(state_path + ".tmp", state_path)

    def validate(step):
        with torch.no_grad():
            validation.append(sum(entry_loss(predict, entry) for entry in val_set) / len(val_set))
        validation_steps.append(step)
        # hand the cached activation memory back to dynamic VRAM's weight cache
        torch.cuda.empty_cache()

    def show_preview(step):
        begin = time.perf_counter()
        try:
            preview["show"](sample_image(mp, preview, cfg["seed"]), step, f"{save_path}_{step:06d}_preview.png")
            logging.info(f"[MOSS Trainer] preview at step {step} in {time.perf_counter() - begin:.1f} s, {preview['width']}x{preview['height']}, {preview['steps']} steps, cfg {preview['cfg']}, "
                         f"{preview['sampler']}/{preview['scheduler']}" + (f", {preview['lora']}" if preview["lora"] != "None" else ""))
        except torch.OutOfMemoryError:
            logging.warning(f"[MOSS Trainer] Out of VRAM on the preview at step {step}, training continues. Lower the preview resolution")
        # the decode loaded the VAE next to the transformer, training needs that VRAM back
        comfy.model_management.unload_model_and_clones(preview["vae"].patcher)
        comfy.model_management.load_models_gpu([mp], memory_required=int(cfg["vram_reserve_gb"] * 1024 ** 3))
        torch.cuda.empty_cache()

    steps = cfg["steps"]
    schedule_free = cfg["optimizer"] == "Prodigy"
    metadata = {"ss_base_model_version": "moss_image", "ss_network_module": cfg["algorithm"], "ss_network_dim": cfg["rank"],
                "ss_network_alpha": cfg["alpha"], "ez_moss_config": json.dumps(cfg)}
    loss_history = state["loss"] if state is not None else []
    validation = state["validation"] if state is not None else []
    validation_steps = state["validation_steps"] if state is not None else []
    ema = state["ema"] if state is not None else None
    pbar = comfy.utils.ProgressBar(steps)
    pbar.update_absolute(start_step)
    logging.info(f"[MOSS Trainer] {len(items)} images, {len(sampler)} batches per epoch, {steps} steps x {cfg['grad_accumulation']} accumulation, batch {cfg['batch_size']}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    step = start_step
    # the memory compiler replays per-block allocation patterns, activations kept for backward break them
    disable_compiler = args.disable_comfy_compiler
    args.disable_comfy_compiler = True
    try:
        for injection in injections:
            injection.inject(mp)
        streamed = {n: m for n, m in mp.model.named_modules() if hasattr(m, "comfy_cast_weights") and (mp.is_dynamic() or hasattr(m, "prev_comfy_cast_weights"))}
        if streamed:
            logging.info(f"[MOSS Trainer] {len(streamed)} layers are streamed from RAM and recomputed in the backward pass")
        if cfg["compile"] and importlib.util.find_spec("triton") is None:
            logging.warning("[MOSS Trainer] compile needs triton (triton-windows on Windows), training without it")
        elif cfg["compile"]:
            # fuses the norm/modulation/rope/activation work around the matmuls, the first steps include compile time
            logging.info("[MOSS Trainer] Compiling transformer blocks, the first step takes about a minute longer")
            for block in mp.model.diffusion_model.layers:
                block.forward = torch.compile(block.forward)
        for block in blocks:
            patch(block)
        for name, m in streamed.items():
            recompute_in_backward(m, [p for p in adapters[name].parameters() if p.requires_grad] if name in adapters else [])
        blocks += streamed.values()
        optimizer = make_optimizer(cfg["optimizer"], params, cfg["learning_rate"], cfg["weight_decay"], device)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            if cfg["optimizer"] == "Prodigy":
                # states written before the d0 bump carry the estimate Prodigy stalled on; Prodigy only
                # ever raises d, so without this a resumed run trains at the saved one forever
                for group in optimizer.param_groups:
                    group["d0"] = 1e-4
                    if group["d"] < group["d0"]:
                        group["d"] = group["d0"]
        # a loaded optimizer keeps initial_lr, so the schedule picks up at start_step
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(cfg["lr_scheduler"], cfg["warmup_steps"], steps), last_epoch=start_step - 1)
        comfy.model_management.in_training = True
        if cfg["validation_every"] > 0:
            val_set = make_validation_set(items, val_images, cfg["seed"], device, dtype)
            if not validation:
                validate(start_step)
                logging.info(f"[MOSS Trainer] validation loss {validation[0]:.4f} before training ({len(val_images)} images x {len(VALIDATION_SIGMAS)} noise levels)")
            val_start = validation[0]
        sample_requests.clear()
        if preview["every"] > 0 and start_step == 0:
            show_preview(0)
        # saved states hold the schedule-free evaluation weights, training steps from the training weights
        if schedule_free:
            optimizer.train()

        def accumulate():
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(cfg["grad_accumulation"]):
                batch = sampler.next()
                # MOSS has no text mask, rope follows the true length: run each caption length as its own forward
                groups = {}
                for i in batch:
                    cond = empty_cond if rng.random() < cfg["caption_dropout"] else items[i]["cond"]
                    groups.setdefault(cond.shape[0], []).append((i, cond))
                for group in groups.values():
                    x0 = torch.stack([items[i]["latent"] for i, _ in group]).to(device, torch.float32)
                    context = torch.stack([c for _, c in group]).to(device, dtype)
                    noise = torch.randn(x0.shape, generator=generator, device=device)
                    sigma = sample_sigmas(len(group), cfg["timestep_distribution"], cfg["timestep_shift"], generator, device)
                    s = sigma.view(-1, 1, 1, 1)
                    pred = predict((1 - s) * x0 + s * noise, sigma, context)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise - x0)
                    (loss * len(group) / len(batch) / cfg["grad_accumulation"]).backward()
                    step_loss += loss.item() * len(group) / len(batch) / cfg["grad_accumulation"]
            return step_loss

        for step in range(start_step, steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            # dynamic VRAM's paging blobs live outside the allocator, so an OOM leaves nothing releasable;
            # rewind the samplers and noisemakers and replay the identical step once on an emptied cache
            replay = (generator.get_state(), sampler.rng.getstate(), sampler.epoch, list(sampler.batches), rng.getstate())
            try:
                step_loss = accumulate()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                generator.set_state(replay[0])
                sampler.rng.setstate(replay[1])
                sampler.epoch = replay[2]
                sampler.batches = replay[3]
                rng.setstate(replay[4])
                try:
                    step_loss = accumulate()
                except torch.OutOfMemoryError:
                    if schedule_free:
                        optimizer.eval()
                    if step > start_step:
                        save_lora(state_dict(), f"{save_path}_{step:06d}_interrupted.safetensors", {**metadata, "ss_steps": step})
                        save_state(step)
                    logging.warning("[MOSS Trainer] Out of VRAM twice on one step, the run is saved as interrupted. Train at lower peak resolutions, or close other GPU apps, and queue again")
                    raise
            if cfg["max_grad_norm"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
            else:
                grad_norm = math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in params if p.grad is not None))
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(step_loss)
            ema = step_loss if ema is None else ema * 0.95 + step_loss * 0.05
            pbar.update(1)
            now = time.perf_counter()
            if step == 0 or (step + 1) % cfg["log_every"] == 0 or step + 1 == steps:
                per_step = (now - start) / (step + 1 - start_step)
                vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if torch.cuda.is_available() else 0.0
                # the paging blobs of dynamic VRAM sit outside the allocator, so report what CUDA sees too
                free = torch.cuda.mem_get_info(device)[0] / 1024 ** 3 if torch.cuda.is_available() else 1.0
                logging.info(f"[MOSS Trainer] step {step + 1}/{steps} | epoch {sampler.epoch} | loss {step_loss:.4f} (avg {ema:.4f}) | lr {lr:.2e} | grad norm {grad_norm:.3f} | "
                             f"{per_step:.2f} s/step | elapsed {_fmt_time(now - start)} | ETA {_fmt_time(per_step * (steps - step - 1))} | peak VRAM {vram:.2f} GB | device free {free:.2f} GB")
            evaluate = cfg["validation_every"] > 0 and ((step + 1) % cfg["validation_every"] == 0 or step + 1 == steps)
            save = cfg["save_every"] > 0 and (step + 1) % cfg["save_every"] == 0 and step + 1 < steps
            show = preview["every"] > 0 and ((step + 1) % preview["every"] == 0 or step + 1 == steps)
            if sample_requests:
                # the latest settings win and stay for the rest of the run
                preview.update(sample_requests[-1])
                sample_requests.clear()
                show = True
            if schedule_free and (evaluate or save or show):
                optimizer.eval()
            if evaluate:
                validate(step + 1)
                logging.info(f"[MOSS Trainer] validation loss {validation[-1]:.4f} at step {step + 1} ({(validation[-1] / val_start - 1) * 100:+.1f}% since start)")
            if save:
                save_lora(state_dict(), f"{save_path}_{step + 1:06d}.safetensors", {**metadata, "ss_steps": step + 1})
                save_state(step + 1)
            if show:
                show_preview(step + 1)
            if schedule_free and (evaluate or save or show):
                optimizer.train()
        if schedule_free:
            optimizer.eval()
    except comfy.model_management.InterruptProcessingException:
        if step > start_step:
            if schedule_free:
                optimizer.eval()
            save_lora(state_dict(), f"{save_path}_{step:06d}_interrupted.safetensors", {**metadata, "ss_steps": step})
            save_state(step)
        raise
    finally:
        comfy.model_management.in_training = False
        args.disable_comfy_compiler = disable_compiler
        for block in blocks:
            unpatch(block)
        if cfg["compile"]:
            for block in mp.model.diffusion_model.layers:
                block.__dict__.pop("forward", None)
        for injection in injections:
            injection.eject(mp)
        # the weights were moved outside inference mode, unload so the next prompt reloads them cleanly
        comfy.model_management.unload_model_and_clones(mp)
        # hand the reserved but free segments back, otherwise the process keeps them scored against every later load
        torch.cuda.empty_cache()

    sd = state_dict()
    save_lora(sd, f"{save_path}.safetensors", {**metadata, "ss_steps": steps})
    save_state(steps)
    logging.info(f"[MOSS Trainer] Done: {steps - start_step} steps in {_fmt_time(time.perf_counter() - start)}, final avg loss {ema:.4f}"
                 + (f", validation loss {validation[0]:.4f} -> {validation[-1]:.4f}" if validation else ""))
    return sd, {"loss": loss_history, "validation": validation, "validation_steps": validation_steps}
