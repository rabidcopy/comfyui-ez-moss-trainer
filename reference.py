import logging
import math

import torch

import comfy.ldm.flux.math
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import modulated_rms_norm, rope_split_half
from comfy_api.latest import io

MODES = ("character", "style", "character+style")
# one learned vector per mode, added to that reference's tokens; saved next to the adapter's LoRA weights
MODE_KEY = "ez_moss_reference.mode_embedding"


def rope_scales(transformer_options):
    rope_options = transformer_options.get("rope_options", {})
    return 1.0, 1.0, rope_options.get("scale_y", 1.0), rope_options.get("scale_x", 1.0)


def ids_rope(dit, ids, transformer_options):
    # same NTK axis scaling as MOSSImage.rope
    pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000 * (s or 1.0) ** (-d / (d - 2))) for i, (d, s) in enumerate(zip(dit.rope_axes_dim, rope_scales(transformer_options)))], dim=-3)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def reference_rope(dit, shapes, frame, device, transformer_options):
    """Reference i sits at i + 1 on axis 0 and at frame on axis 1, its rows and columns centered like the target's."""
    scales = rope_scales(transformer_options)
    ids = []
    for i, (h, w) in enumerate(shapes):
        rows = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * float(scales[2] != 0)
        cols = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * float(scales[3] != 0)
        grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).flatten(0, 1)
        ids.append(torch.cat((grid.new_full((h * w, 1), i + 1.0), grid.new_full((h * w, 1), frame), grid), dim=-1))
    return ids_rope(dit, torch.cat(ids)[None], transformer_options)


def run_blocks(dit, x, mod, pe, vec, transformer_options):
    """The MOSSImage block loop with prefetch, the memory compiler and dit block patches."""
    device, dtype = x.device, x.dtype
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(dit.layers)
    transformer_options["block_type"] = "single"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dit.layers), device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(dit.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"])}
            x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
        else:
            x = block(x, mod, pe, transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()
    return x


def split_attention(transformer_options, n, spans, bias):
    """The first n tokens attend to every key, each reference span only to itself. Separate unmasked calls keep SDPA on
    its fused kernels, a mask over the joint sequence does not. bias is the first n tokens' mask, None without one."""
    override = transformer_options.get("optimized_attention_override")

    def attention(func, q, k, v, *args, **kwargs):
        # the caller passes its own mask, ours replaces it: the target's bias or none
        kwargs.pop("mask", None)
        def call(q, k, v, mask=None):
            return override(func, q, k, v, *args, mask=mask, **kwargs) if override is not None else func(q, k, v, *args, mask=mask, **kwargs)
        return torch.cat([call(q[:, :n], k, v, bias)] + [call(q[:, start:end], k[:, start:end], v[:, start:end]) for start, end in spans], dim=1)
    return attention


def forward(dit, x, timestep, context, refs, modes, transformer_options={}):
    """MOSS forward with reference latents appended after the text. refs are (latent, mode index, strength).

    References run at timestep 0 and only attend to themselves, the target and text attend to them with a log(strength) bias."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    text_len = context.shape[1]
    shapes = [latent.shape[-2:] for latent, _, _ in refs]
    n, m = H * W + text_len, sum(h * w for h, w in shapes)
    frame = text_len + transformer_options.get("rope_options", {}).get("shift_t", 0.0)
    pe = torch.cat((dit.rope(H, W, text_len, device, transformer_options), reference_rope(dit, shapes, frame, device, transformer_options)), dim=1)

    vec = dit.time_embedding(timestep_embedding(torch.cat((timestep, torch.zeros_like(timestep))), 256).to(dtype))
    mod = dit.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    mod = [torch.cat((t[:B].expand(-1, n, -1), t[B:].expand(-1, m, -1)), dim=1) for t in mod]
    head_shift, head_scale = dit.video_head_modulation(vec[:B]).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

    mode_embedding = comfy.ops.cast_to_input(modes, x, copy=False)
    ref_tokens = torch.cat([latent.flatten(2).transpose(1, 2).expand(B, -1, -1) for latent, _, _ in refs], dim=1)
    ref_modes = torch.cat([mode_embedding[mode].expand(h * w, -1) for (_, mode, _), (h, w) in zip(refs, shapes)])
    txt = dit.text_embedder(context, pe[:, H * W:n], transformer_options)
    # allocated outside the memory compiler region, the blocks update it in place
    x = torch.cat((dit.video_embedder(x.flatten(2).transpose(1, 2)), txt, dit.video_embedder(ref_tokens) + ref_modes), dim=1)
    del txt, ref_tokens, ref_modes

    spans, start = [], n
    for h, w in shapes:
        spans.append((start, start + h * w))
        start += h * w
    bias = None
    if any(strength != 1 for _, _, strength in refs):
        bias = torch.cat([torch.zeros(n, n, device=device, dtype=dtype)] + [torch.full((n, h * w), math.log(strength), device=device, dtype=dtype) for (_, _, strength), (h, w) in zip(refs, shapes)], dim=1)
    # the text embedder above attends over the text alone, only the blocks split the attention
    transformer_options["optimized_attention_override"] = split_attention(transformer_options, n, spans, bias)
    x = run_blocks(dit, x, mod, pe, vec[:B], transformer_options)

    out = dit.final_linear_video(modulated_rms_norm(x[:, :H * W], dit.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, dit.latent_channels, H, W)


def reference_kv(dit, refs, modes, x, transformer_options):
    """Per block keys and values of the references alone, at timestep 0 and frame 0. Same math as forward, since the
    references never attend to the target or text.

    Each reference runs alone, so the peak is one reference's activations. A reference whose cache does not fit in free
    VRAM, leaving ComfyUI's inference reserve, is cached in pinned RAM and copied in per block. Returns kv[ref][block]."""
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    shapes = [latent.shape[-2:] for latent, _ in refs]
    vec = dit.time_embedding(timestep_embedding(torch.zeros(1, device=device), 256).to(dtype))
    mod = dit.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    mode_embedding = comfy.ops.cast_to_input(modes, x, copy=False)
    pe = reference_rope(dit, shapes, 0.0, device, transformer_options)
    override = transformer_options.get("optimized_attention_override")
    kv = []
    start = 0
    in_ram = 0
    for (latent, mode), (h, w) in zip(refs, shapes):
        size = 2 * len(dit.layers) * h * w * dit.video_embedder.weight.shape[0] * x.element_size()
        offload = device.type != "cpu" and size > comfy.model_management.get_free_memory(device) - comfy.model_management.minimum_inference_memory()
        in_ram += offload
        ref_kv = []

        def capture(func, *args, **kwargs):
            # v is a chunk of the fused qkv output, copy it so the whole qkv is not kept
            ref_kv.append(tuple(torch.empty_like(t, device="cpu", pin_memory=device.type == "cuda").copy_(t) if offload else t.contiguous() for t in args[1:3]))
            return override(func, *args, **kwargs) if override is not None else func(*args, **kwargs)

        transformer_options["optimized_attention_override"] = capture
        tokens = dit.video_embedder(comfy.ops.cast_to_input(latent, x, copy=False).flatten(2).transpose(1, 2)) + mode_embedding[mode]
        run_blocks(dit, tokens, mod, pe[:, start:start + h * w], vec, transformer_options)
        kv.append(ref_kv)
        start += h * w
    if in_ram:
        logging.info(f"[EZ MOSS Reference] {in_ram} of {len(refs)} references cached in RAM, VRAM is full. Sampling is slower, fewer or smaller references avoid it.")
    return kv


class References:
    """Wrappers that add references to sampling. references are (latent, mode index, strength, fidelity), a reference is
    dropped once the fidelity fraction of the sampling steps has run.

    The references' per block keys and values do not depend on the target, timestep or prompt, so they are computed once
    per sampling run and dropped when it ends."""

    def __init__(self, references, modes):
        self.references = references
        self.modes = modes
        self.kv = None

    def outer_sample(self, executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            self.kv = None

    def diffusion_model(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        sample_sigmas = transformer_options["sample_sigmas"].tolist()
        sigma = float(transformer_options["sigmas"].max())
        active = [strength > 0 and sigma > sample_sigmas[round(fidelity * (len(sample_sigmas) - 1))] for _, _, strength, fidelity in self.references]
        if not any(active):
            return executor(x, timestep, context, transformer_options, **kwargs)
        dit = executor.class_obj
        if self.kv is None:
            self.kv = reference_kv(dit, [(latent, mode) for latent, mode, _, _ in self.references], self.modes, x, transformer_options)
        kv = [ref_kv for ref_kv, a in zip(self.kv, active) if a]
        strengths = [(strength, latent.shape[-2] * latent.shape[-1]) for (latent, _, strength, _), a in zip(self.references, active) if a]
        # the cached keys sit at frame 0, the target is at the text length
        frame = context.shape[1] + transformer_options.get("rope_options", {}).get("shift_t", 0.0)
        shift = ids_rope(dit, torch.tensor([[[0.0, frame, 0.0, 0.0]]], device=x.device), transformer_options)
        override = transformer_options.get("optimized_attention_override")
        masks = {}

        def attention(func, q, k, v, heads, *args, **kwargs):
            # the text embedder runs before the blocks set block_index and attends over the text alone
            if "block_index" in kwargs["transformer_options"]:
                i = kwargs["transformer_options"]["block_index"]
                ref_k = torch.cat([ref_kv[i][0].to(q.device, non_blocking=True) for ref_kv in kv], dim=1)
                ref_v = torch.cat([ref_kv[i][1].to(q.device, non_blocking=True) for ref_kv in kv], dim=1)
                ref_k = rope_split_half(ref_k.unflatten(-1, (heads, -1)), shift).flatten(2)
                n = q.shape[1]
                if any(strength != 1 for strength, _ in strengths) and n not in masks:
                    masks[n] = torch.cat([q.new_zeros(n, n)] + [q.new_full((n, size), math.log(strength)) for strength, size in strengths], dim=1)
                if n in masks:
                    kwargs["mask"] = masks[n]
                k = torch.cat((k, ref_k.expand(k.shape[0], -1, -1)), dim=1)
                v = torch.cat((v, ref_v.expand(v.shape[0], -1, -1)), dim=1)
            return override(func, q, k, v, heads, *args, **kwargs) if override is not None else func(q, k, v, heads, *args, **kwargs)

        return executor(x, timestep, context, {**transformer_options, "optimized_attention_override": attention}, **kwargs)


class EZMOSSReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EZMOSSReference",
            display_name="EZ MOSS Reference",
            category="advanced/model_patches",
            description="One reference image for EZ MOSS Apply References. Every image of a batch becomes its own reference with these settings.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=list(MODES), default="character",
                               tooltip="What to take from the image: the character's look, the art style, or both."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.05,
                               tooltip="Scales how much attention the generation pays to this reference. 0 turns it off."),
                io.Float.Input("fidelity", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Fraction of the sampling steps that see this reference. Lower values keep the broad look and leave the details to the prompt."),
            ],
            outputs=[io.Custom("EZ_MOSS_REFERENCE").Output(display_name="reference")],
        )

    @classmethod
    def execute(cls, image, mode, strength, fidelity) -> io.NodeOutput:
        return io.NodeOutput({"image": image, "mode": MODES.index(mode), "strength": strength, "fidelity": fidelity})


class EZMOSSApplyReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EZMOSSApplyReferences",
            display_name="EZ MOSS Apply References",
            category="advanced/model_patches",
            description="Loads a reference adapter trained by the EZ MOSS trainer with reference_training on, "
                        "and makes the model take characters and styles from the connected references.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Combo.Input("adapter", options=folder_paths.get_filename_list("loras"), tooltip="Reference adapter from the loras folder."),
                io.Int.Input("size", default=512, min=64, max=4096, step=16,
                             tooltip="References are scaled to about size x size pixels. The trainer uses its training resolution, 512 by default."),
                io.Autogrow.Input("references", template=io.Autogrow.TemplatePrefix(
                    input=io.Custom("EZ_MOSS_REFERENCE").Input("reference"), prefix="reference_", min=0, max=16)),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, vae, adapter, size, references: io.Autogrow.Type = None) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", adapter), safe_load=True)
        if MODE_KEY not in sd:
            raise ValueError(f"{adapter} is not an EZ MOSS reference adapter, train one with reference_training on.")
        modes = sd.pop(MODE_KEY)
        model = comfy.sd.load_lora_for_models(model, None, sd, 1.0, 0)[0]
        refs = []
        for ref in (references or {}).values():
            image = ref["image"][..., :3].movedim(-1, 1)
            scale = size / math.sqrt(image.shape[2] * image.shape[3])
            image = comfy.utils.common_upscale(image, max(16, round(image.shape[3] * scale / 16) * 16), max(16, round(image.shape[2] * scale / 16) * 16), "area", "disabled")
            for latent in model.model.process_latent_in(vae.encode(image.movedim(1, -1))).split(1):
                refs.append((latent, ref["mode"], ref["strength"], ref["fidelity"]))
        references = References(refs, modes)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "ez_moss_reference", references.outer_sample)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "ez_moss_reference", references.diffusion_model)
        return io.NodeOutput(model)
