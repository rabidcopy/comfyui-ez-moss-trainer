# EZ MOSS-Image LoRA Trainer

One ComfyUI node that trains a LoRA, LoHa, LoKr or OFT adapter on the MOSS-Image transformer. Only the transformer is trained. The text encoder and VAE are used once to fill a disk cache.

It works with bf16 and int8 / int8-convrot MOSS files. The base weights stay quantized and frozen: adapters run in bypass mode, so the int8 GEMM is used in the forward pass and the weight is dequantized per layer in the backward pass. Speed on an RTX 3050 8 GB with the int8 convrot transformer and ComfyUI's default dynamic VRAM, measured as steady-state s/step (peak is PyTorch's allocated peak):

| Resolution, captions | `compile` off | `compile` on (default) | Peak VRAM |
|---|---|---|---|
| 512, short (~15 tokens) | 2.9 s/step | 2.5 s/step | 0.9 GB |
| 512, detailed (~250 tokens) | 3.6 s/step | 3.1 s/step | 1.0 GB |
| 1024, detailed (~250 tokens) | 13.6 s/step | 12.0 s/step | 2.4 GB |

Text tokens go through every transformer block along with the image tokens, so long captions cost time. `--disable-pinned-memory` made no measurable difference with dynamic VRAM.

## EZ Dataset Captioner

Captions every image in a folder (subfolders included) with a vision LLM served behind an OpenAI-compatible API, and writes `image.txt` next to `image.png`. It defaults to LM Studio on `http://127.0.0.1:1234/v1`. Ollama (`:11434/v1`), llama.cpp (`:8080/v1`), vLLM and OpenAI work the same way. Requests only go to the `base_url` you set.

- **list models** fetches `<base_url>/models` and opens a searchable picker at the button. Type to filter (several words all have to match, e.g. `qwen vl`), then click a model or press Enter. An empty `model` uses the first model the server lists. If the button still shows the old menu after updating, hard refresh the browser (Ctrl+F5).
- Presets: `detailed` (default, suits MOSS's LLM text encoder), `medium`, `short`, `tags`, `style LoRA (content only)`, `character LoRA (no identity)`, `text-heavy (OCR)`, `reference fields (JSON)`. `system_prompt` replaces the preset when set.
- `reference fields (JSON)` writes one line of JSON with `character` (permanent look), `outfit`, `scene`, `style` and `visible` (`yes`, `partial` or `no`), so captions for reference training can leave out whatever a reference image supplies. A reply that is not such an object, a refusal for example, counts as failed and is retried on the next run. `prepend` is ignored.
- If an image has a `.json` next to it with a `tags` list, the tags are sent with the image as hints. `extra_instructions` is appended to whichever of the two is active, e.g. `Call the woman with silver hair 'kiri'.`
- `prepend` adds a trigger word, e.g. `mytrigger` → `mytrigger, A woman in...`.
- Existing captions are kept unless `overwrite` is on. `<think>` blocks, "Here is a caption:" preambles and leaked chat template markers (a caption starting with "model model ...") are stripped. A reply with nothing left is counted as failed and writes no `.txt`.
- `max_tokens` defaults to 2048 because thinking models spend tokens reasoning before the caption. A reply cut off at `max_tokens` with no caption fails right away with a hint to raise it.
- Temporary failures (HTTP 408/429/5xx, timeouts, connection errors, empty replies) are retried up to 3 times with 2/4/8 s backoff, or the server's `Retry-After`. Other errors such as 400 or 401 fail right away. Running the node again only captions images that still have no `.txt`.
- API keys: see [API keys](#api-keys).

Connect its `folder` output to the trainer's `dataset_paths` input to caption and train in one run. Changed captions are re-encoded automatically.

### API keys

Keys never need to be in a workflow. Anything typed into a widget is saved in the workflow file and in the metadata of every image you generate, so a shared PNG would leak it.

- Click **set API key** on the captioner or the prompt writer, paste the key into the masked field and press Enter. It is saved per `base_url` in `ComfyUI/user/ez_moss_api_keys.json`, outside workflows, image metadata and this repo. The button reads "API key saved for this URL" once a key is stored. Save an empty key to remove it.
- You can also edit that file directly: `{"https://nano-gpt.com/api/v1": "sk-..."}`. The URL is matched without a trailing slash and ignoring case.
- Lookup order: the `api_key` field, then the saved key for the `base_url`, then the `OPENAI_API_KEY` environment variable.
- A key found in the `api_key` field, from an older workflow or typed in, is moved to the file automatically and the field is cleared. Save the workflow again afterwards so the old copy is gone from it. Images generated before the move still contain the key in their metadata. If any of them were shared, rotate the key at your provider.

## EZ MOSS Prompt Writer

Writes or enhances a MOSS-Image prompt with an LLM behind the same kind of OpenAI-compatible API as the captioner, and outputs it as a string for a CLIP Text Encode node. Connect any number of reference images: a new `image` socket appears each time you connect one, and batches are expanded. Images need a vision model.

| Preset | What it writes |
|---|---|
| `enhance prompt` (default) | Your prompt expanded with concrete subject, setting, composition, lighting and style details. Images act as visual reference. |
| `character from references` | The character's permanent look from the images, placed into your scene. With no prompt, a full-body portrait. |
| `style from references` | The images' art style worked out in detail, applied to your prompt without copying their subjects. |
| `character + style from references` | Both of the above together. |
| `recreate image` | A prompt that reproduces the image. Your prompt lists changes to make. |

Every preset asks for one plain paragraph, with any text to render quoted. `system_prompt` replaces the preset entirely, and `extra_instructions` is appended to whichever is active. `seed` is sent to the API and changes after each run, so queueing again gives a new variation. Reply cleanup, retries, the `max_tokens` handling for thinking models and the **list models** button work the same as in the captioner. The final prompt is also printed to the console.

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

`models/loras/<output_name>.safetensors`, plus `<output_name>_<step>.safetensors` every `save_every` steps. If you cancel the queue, the current weights are saved as `<output_name>_<step>_interrupted.safetensors`. Every save also overwrites `<output_name>.training_state` (fp32 weights, optimizer and sampler state); turn on `resume` with the same settings to continue from it, raising `steps` to train longer. LoRA files trained on fused-qkv MOSS files are saved with split `q/k/v` keys, so they load on both fused and unfused MOSS checkpoints.

### Validation loss

The training loss swings a lot from step to step because every step uses a random noise level. Every `validation_every` steps (default 100, 0 disables), the trainer also measures the loss on up to 4 dataset images at fixed noise levels (0.25, 0.5, 0.75, 0.9) with fixed noise. It does the same once before training as a baseline. Example output:

```
[MOSS Trainer] validation loss 0.1301 before training (4 images x 4 noise levels)
[MOSS Trainer] validation loss 0.1214 at step 300 (-6.7% since start)
```

A steady decline means the LoRA is learning the dataset. A large drop late in the run usually means memorization, so compare the intermediate saves. The values are also in `loss_map` under `validation` and `validation_steps`. With `reference_training` on, each line also gives the loss change when the references are left out (`+0.0123 without references`). A gap that grows over the run means the adapter is learning to use the references.

## Reference adapter

`reference_training` trains an adapter that takes a character's look, an art style or both from reference images at generation time, with no LoRA per character. It needs captions from the `reference fields (JSON)` preset and a `.json` sidecar next to each image with:

- `artist`: the image's one artist.
- `characters`: the named characters in it, which can be an empty list.
- `sheet`: `true` for model and reference sheets.
- `related`: the post ids of the image and of its other versions, including its own id.

Character and artist names only need to be consistent within the dataset.

- Every step pairs an image with one or two references. A character reference shows the same single character by another artist. A style reference is by the same artist with no shared character. A character+style reference shares both. The caption leaves out the fields the references supply, and drops the outfit half the time when the character comes from a reference. One step in ten has no references. Versions of one post and near duplicates are never paired. Model sheets are only used as references.
- To teach the adapter to read references at all, some steps use a random crop (50-80% of each side) of the image itself as its character+style reference: 30% of steps at the start of a run, tapering to 10% at the end. Validation never uses them.
- Each training image gets up to 6 caption variants. All of them are encoded once before training: about 7,800 for 1,666 images, roughly 12.5 GB in `.moss_cache/text`. They are read from disk per step instead of kept in RAM.
- References run through the transformer next to the image at timestep 0. Each one only attends to itself. The image and the prompt attend to all of them. While sampling, each reference's per-block keys and values are computed once and reused every step. That takes about 0.5 GB of VRAM per 512px reference for the length of the run. A reference that doesn't fit in free VRAM is cached in RAM instead and copied in block by block, which is slower but doesn't run out of memory. The file is a normal LoRA plus `ez_moss_reference.mode_embedding`, one learned vector per mode.

To generate, add one **EZ MOSS Reference** per image (`mode`, `strength`, `fidelity`) and connect them all to **EZ MOSS Apply References** together with the model, the VAE and the adapter. `strength` scales the attention paid to that reference. `fidelity` is the fraction of the sampling steps that see it, so lower values keep the broad look and leave the details to the prompt. References are scaled to about `size` x `size` pixels, 512 like the training resolution.

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

## Compile

`compile` (default on) runs `torch.compile` on each transformer block, which fuses the norm, modulation, rope and activation work around the matmuls. Training math is unchanged: losses match uncompiled runs to 3–4 digits. Expect the first step to take about a minute longer. Different buckets and caption lengths do not trigger further recompiles. Streamed layers stay outside the compiled graph, which is split around them. It needs triton (`triton-windows` on Windows); without it, the node logs a warning and trains uncompiled. Forcing the cuDNN, flash or memory-efficient attention backend was not faster than PyTorch's default choice.

## VRAM

With dynamic VRAM (the ComfyUI default), weights are paged in per block automatically, so there is nothing to tune. The first step takes about 20 s while weights are paged in. After caching, the node unloads the text encoder and VAE so the transformer has RAM and VRAM to itself. The model compiler is paused during training and restored afterwards. Each layer is re-run right before its own backward pass, so a weight that dynamic VRAM has already paged out never gets used. Only each block's input is kept between the forward and backward passes, about 25 MB per block at 1024px. A validation pass at 1024px takes about 45 s, so keep `validation_every` around 100 or higher there.

With legacy loading (`--disable-dynamic-vram`), `vram_reserve_gb` (default 2.5) is the VRAM kept free for activations, gradients and streaming buffers. Transformer weights that do not fit in the rest are streamed from system RAM each step, and those layers are recomputed in the backward pass. If you run out of memory, raise the reserve or lower the resolution. On a 6 GB card, try a reserve of 3–3.5. Streaming relies on pinned memory: with `--disable-pinned-memory`, legacy loading measured about 15 s/step, so stay on dynamic VRAM.

LoHa computes a full-size weight delta per layer, so it is about 3x slower and uses about 1 GB more VRAM than LoRA, LoKr or OFT.

Attention runs through PyTorch SDPA during training, because the Comfy Kitchen int8 attention that quantized MOSS files select has no backward pass. Sampling afterwards still uses the checkpoint's attention.
