import importlib.util
import json
import logging
import math
import os
import random
import time

import safetensors.torch
import torch

import comfy.model_management
from comfy.cli_args import args
import comfy.utils
import node_helpers
from comfy.ldm.modules.attention import attention_pytorch
from comfy.weight_adapter import adapter_maps
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


def make_optimizer(name, params, lr, weight_decay, device):
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=device.type == "cuda")
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


def make_validation_set(items, seed, device, dtype):
    """Up to 4 images spread over the dataset, each with fixed noise levels and noise."""
    n = min(4, len(items))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    val_set = []
    for i in range(n):
        item = items[round(i * (len(items) - 1) / max(1, n - 1))]
        x0 = item["latent"][None].to(device, torch.float32)
        context = item["cond"][None].to(device, dtype)
        for sigma in VALIDATION_SIGMAS:
            val_set.append((x0, context, torch.randn(x0.shape, generator=generator, device=device), torch.tensor([sigma], device=device)))
    return val_set


def validation_loss(model, val_set, dtype, autocast, transformer_options):
    """Velocity MSE on the fixed validation set, so changes come from the adapter and not from sampled noise."""
    total = 0.0
    with torch.no_grad():
        for x0, context, noise, sigma in val_set:
            s = sigma.view(-1, 1, 1, 1)
            with torch.autocast(x0.device.type, dtype=dtype, enabled=autocast):
                pred = model.diffusion_model(((1 - s) * x0 + s * noise).to(dtype), model.model_sampling.timestep(sigma).float(), context=context, transformer_options=transformer_options)
            total += torch.nn.functional.mse_loss(pred.float(), noise - x0).item()
    # hand the cached activation memory back to dynamic VRAM's weight cache
    torch.cuda.empty_cache()
    return total / len(val_set)


def _fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def train(mp, items, empty_cond, cfg, save_path):
    """Trains bypass adapters on the MOSS transformer. Returns (lora state dict, loss map)."""
    device = mp.load_device
    dtype = mp.model.get_dtype_inference()
    autocast = dtype in (torch.bfloat16, torch.float16)

    torch.manual_seed(cfg["seed"])
    adapters, injections = setup_adapters(mp, cfg["algorithm"], cfg["rank"], cfg["alpha"], cfg["target"])
    params = [p for a in adapters.values() for p in a.parameters() if p.requires_grad]
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
    model_sampling = mp.model.model_sampling
    transformer_options = mp.model_options.get("transformer_options", {}).copy()
    transformer_options["prefetch_dynamic_vbars"] = mp.is_dynamic()
    # quantized checkpoints can pin int8 kitchen attention per block, which has no backward
    transformer_options["optimized_attention_override"] = lambda func, *args, **kwargs: attention_pytorch(*args, **kwargs)

    steps = cfg["steps"]
    metadata = {"ss_base_model_version": "moss_image", "ss_network_module": cfg["algorithm"], "ss_network_dim": cfg["rank"],
                "ss_network_alpha": cfg["alpha"], "ez_moss_config": json.dumps(cfg)}
    loss_history = []
    validation = []
    validation_steps = []
    ema = None
    pbar = comfy.utils.ProgressBar(steps)
    logging.info(f"[MOSS Trainer] {len(items)} images, {len(sampler)} batches per epoch, {steps} steps x {cfg['grad_accumulation']} accumulation, batch {cfg['batch_size']}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    step = 0
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(cfg["lr_scheduler"], cfg["warmup_steps"], steps))
        comfy.model_management.in_training = True
        if cfg["validation_every"] > 0:
            val_set = make_validation_set(items, cfg["seed"], device, dtype)
            val_start = validation_loss(mp.model, val_set, dtype, autocast, transformer_options)
            validation.append(val_start)
            validation_steps.append(0)
            logging.info(f"[MOSS Trainer] validation loss {val_start:.4f} before training ({len(val_set) // len(VALIDATION_SIGMAS)} images x {len(VALIDATION_SIGMAS)} noise levels)")
        for step in range(steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            step_loss = 0.0
            for _ in range(cfg["grad_accumulation"]):
                batch = sampler.next()
                conds = [empty_cond if rng.random() < cfg["caption_dropout"] else items[i]["cond"] for i in batch]
                # MOSS has no text mask, rope follows the true length: run each caption length as its own forward
                groups = {}
                for i, cond in zip(batch, conds):
                    groups.setdefault(cond.shape[0], []).append((i, cond))
                for group in groups.values():
                    x0 = torch.stack([items[i]["latent"] for i, _ in group]).to(device, torch.float32)
                    context = torch.stack([c for _, c in group]).to(device, dtype)
                    noise = torch.randn(x0.shape, generator=generator, device=device)
                    sigma = sample_sigmas(len(group), cfg["timestep_distribution"], cfg["timestep_shift"], generator, device)
                    s = sigma.view(-1, 1, 1, 1)
                    xt = (1 - s) * x0 + s * noise
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast):
                        pred = mp.model.diffusion_model(xt.to(dtype), model_sampling.timestep(sigma).float(), context=context, transformer_options=transformer_options)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise - x0)
                    (loss * len(group) / len(batch) / cfg["grad_accumulation"]).backward()
                    step_loss += loss.item() * len(group) / len(batch) / cfg["grad_accumulation"]
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
                per_step = (now - start) / (step + 1)
                vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if torch.cuda.is_available() else 0.0
                logging.info(f"[MOSS Trainer] step {step + 1}/{steps} | epoch {sampler.epoch} | loss {step_loss:.4f} (avg {ema:.4f}) | lr {lr:.2e} | grad norm {grad_norm:.3f} | "
                             f"{per_step:.2f} s/step | elapsed {_fmt_time(now - start)} | ETA {_fmt_time(per_step * (steps - step - 1))} | peak VRAM {vram:.2f} GB")
            if cfg["validation_every"] > 0 and ((step + 1) % cfg["validation_every"] == 0 or step + 1 == steps):
                val = validation_loss(mp.model, val_set, dtype, autocast, transformer_options)
                validation.append(val)
                validation_steps.append(step + 1)
                logging.info(f"[MOSS Trainer] validation loss {val:.4f} at step {step + 1} ({(val / val_start - 1) * 100:+.1f}% since start)")
            if cfg["save_every"] > 0 and (step + 1) % cfg["save_every"] == 0 and step + 1 < steps:
                save_lora(adapter_state_dict(adapters, cfg["save_dtype"]), f"{save_path}_{step + 1:06d}.safetensors", {**metadata, "ss_steps": step + 1})
    except comfy.model_management.InterruptProcessingException:
        if step > 0:
            save_lora(adapter_state_dict(adapters, cfg["save_dtype"]), f"{save_path}_{step:06d}_interrupted.safetensors", {**metadata, "ss_steps": step})
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

    sd = adapter_state_dict(adapters, cfg["save_dtype"])
    save_lora(sd, f"{save_path}.safetensors", {**metadata, "ss_steps": steps})
    logging.info(f"[MOSS Trainer] Done: {steps} steps in {_fmt_time(time.perf_counter() - start)}, final avg loss {ema:.4f}"
                 + (f", validation loss {validation[0]:.4f} -> {validation[-1]:.4f}" if validation else ""))
    return sd, {"loss": loss_history, "validation": validation, "validation_steps": validation_steps}
