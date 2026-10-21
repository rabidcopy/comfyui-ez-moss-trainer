# EZ MOSS-Image LoRA Trainer

One ComfyUI node that trains a LoRA, LoHa, LoKr or OFT adapter on the MOSS-Image transformer. Only the transformer is trained. The text encoder and VAE are used once to fill a disk cache.

It works with bf16 and int8 / int8-convrot MOSS files. The base weights stay quantized and frozen: adapters run in bypass mode, so the int8 GEMM is used in the forward pass and the weight is dequantized per layer in the backward pass. On an RTX 3050 8 GB with the int8 convrot transformer at 512px, LoRA training takes about 3.2 s/step with ComfyUI's default dynamic VRAM (3.7 GB peak), and about 2.8 s/step with legacy loading (`--disable-dynamic-vram`, about 6.5 GB peak).

## EZ Dataset Captioner

Captions every image in a folder (subfolders included) with a vision LLM served behind an OpenAI-compatible API, and writes `image.txt` next to `image.png`. It defaults to LM Studio on `http://127.0.0.1:1234/v1`. Ollama (`:11434/v1`), llama.cpp (`:8080/v1`), vLLM and OpenAI work the same way. Requests only go to the `base_url` you set.

- **list models** fetches `<base_url>/models` and fills the `model` field. Text already in `model` filters the list (type `gemma-4-12b` and press the button to see only matching ids). An empty `model` uses the first model the server lists.
- Presets: `detailed` (default, suits MOSS's LLM text encoder), `medium`, `short`, `tags`, `style LoRA (content only)`, `character LoRA (no identity)`, `text-heavy (OCR)`. `system_prompt` replaces the preset when set. `extra_instructions` is appended to whichever of the two is active, e.g. `Call the woman with silver hair 'kiri'.`
- `prepend` adds a trigger word, e.g. `mytrigger` → `mytrigger, A woman in...`.
- Existing captions are kept unless `overwrite` is on. `<think>` blocks, "Here is a caption:" preambles and leaked chat template markers (a caption starting with "model model ...") are stripped. A reply with nothing left is counted as failed and writes no `.txt`.
- Leave `api_key` empty to use the `OPENAI_API_KEY` environment variable. A key typed in the node is saved in the workflow.

Connect its `folder` output to the trainer's `dataset_paths` input to caption and train in one run. Changed captions are re-encoded automatically.

## Usage

`Load Diffusion Model` (MOSS) + `Load CLIP` (type `moss_image`) + `Load VAE` (Flux2) → **EZ MOSS-Image LoRA Trainer**.

Optional: connect `lora` to the core `Load LoRA Model` node to try the result right away, and `loss_map` to `Plot Loss Graph`.

### Dataset

Put one folder per line in `dataset_paths`. Each image needs a caption in a `.txt` file with the same name:

```
D:/datasets/my_style
D:/datasets/my_character | 3
```

`| 3` repeats that folder 3 times per epoch. Subfolders are included. Images without a caption are trained with the empty prompt.

### Cache

Latents go in `<folder>/.moss_cache/latents` and text encodings go in `<folder>/.moss_cache/text`, unless you set `cache_dir`. The cache key covers the image path, file mtime and size, and the bucket. Text is keyed by the caption. Edited images and captions are re-encoded automatically. Delete `.moss_cache` if you switch to a different text encoder or VAE.

When everything is cached, the text encoder and VAE are not run, and the transformer gets the whole GPU.

### Output

`models/loras/<output_name>.safetensors`, plus `<output_name>_<step>.safetensors` every `save_every` steps. If you cancel the queue, the current weights are saved as `<output_name>_<step>_interrupted.safetensors`. LoRA files trained on fused-qkv MOSS files are saved with split `q/k/v` keys, so they load on both fused and unfused MOSS checkpoints.

### Validation loss

The training loss swings a lot from step to step because every step uses a random noise level. Every `validation_every` steps (default 100, 0 disables), the trainer also measures the loss on up to 4 dataset images at fixed noise levels (0.25, 0.5, 0.75, 0.9) with fixed noise. It does the same once before training as a baseline. Example output:

```
[MOSS Trainer] validation loss 0.1301 before training (4 images x 4 noise levels)
[MOSS Trainer] validation loss 0.1214 at step 300 (-6.7% since start)
```

A steady decline means the LoRA is learning the dataset. A large drop late in the run usually means memorization, so compare the intermediate saves. The values are also in `loss_map` under `validation` and `validation_steps`.

## Defaults

These follow OneTrainer's MOSS LoRA preset (PR #1741) where it has one:

| Setting | Default | Note |
|---|---|---|
| target_layers | attn-mlp | q, k, v (or fused qkv), o, ffn_up_gate, ffn_down in all 42 blocks |
| rank / alpha | 16 / 1.0 | |
| learning_rate | 3e-4 | AdamW, weight decay 0.01, grad clip 1.0 |
| resolution | 512 | Aspect buckets of equal area in 64px steps |
| timesteps | logit-normal, shift 1.0 | x_t = (1-σ)·x0 + σ·ε, MSE on the velocity ε − x0 |
| warmup | 50 steps | |

## VRAM

With dynamic VRAM (the ComfyUI default), weights are paged in per block automatically, so there is nothing to tune. The first step takes about 20 s while weights are paged in. After caching, the node unloads the text encoder and VAE so the transformer has RAM and VRAM to itself. The model compiler is paused during training and restored afterwards.

With legacy loading (`--disable-dynamic-vram`), `vram_reserve_gb` (default 2.5) is the VRAM kept free for activations, gradients and streaming buffers. Transformer weights that do not fit in the rest are streamed from system RAM each step, and those layers are recomputed in the backward pass. If you run out of memory, raise the reserve or lower the resolution. On a 6 GB card, try a reserve of 3–3.5. Streaming relies on pinned memory: with `--disable-pinned-memory`, legacy loading measured about 15 s/step, so stay on dynamic VRAM.

LoHa computes a full-size weight delta per layer, so it is about 3x slower and uses about 1 GB more VRAM than LoRA, LoKr or OFT.

Attention runs through PyTorch SDPA during training, because the Comfy Kitchen int8 attention that quantized MOSS files select has no backward pass. Sampling afterwards still uses the checkpoint's attention.
