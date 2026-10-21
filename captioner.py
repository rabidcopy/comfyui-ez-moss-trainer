import asyncio
import logging
import os

import aiohttp
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

from .dataset import IMAGE_EXTENSIONS
from .vlm import chat, encode_image, resolve_model, warn_inline_key

PRESETS = {
    "detailed": "You write captions for text-to-image training data. Describe the image in one detailed paragraph of plain natural language: "
                "the main subject, their appearance, clothing, pose and expression, other objects, the setting and background, the composition and camera angle, "
                "the lighting and colors, and the medium or art style. Quote any visible text exactly. State only what is visible.",
    "medium": "You write captions for text-to-image training data. Describe the image in two or three sentences of plain natural language: "
              "the main subject and what they are doing, the setting, and the art style or medium. Quote any visible text exactly.",
    "short": "You write captions for text-to-image training data. Describe the image in one short sentence naming the main subject, the setting and the medium.",
    "tags": "You write booru-style tags for text-to-image training data. Output a single line of lowercase comma-separated tags: "
            "subject count, character features, clothing, pose, expression, objects, background, composition, art style. No sentences.",
    "style LoRA (content only)": "You write captions for training a style LoRA. Describe what the image shows in one paragraph of plain natural language: "
                                 "subjects, clothing, pose, objects, setting, composition and any visible text quoted exactly. "
                                 "Do not describe the art style, medium, brushwork, line quality, rendering or color palette, the model learns those from the images.",
    "character LoRA (no identity)": "You write captions for training a character LoRA. Describe the image in one paragraph of plain natural language: "
                                    "clothing, accessories, pose, expression, action, setting, composition, lighting and art style. "
                                    "Do not describe the character's permanent traits such as face, hair, eye color, skin, body shape or species, "
                                    "refer to them only as \"the character\".",
    "text-heavy (OCR)": "You write captions for text-to-image training data. First describe the image in two or three sentences of plain natural language, "
                        "then describe every piece of visible text: quote it exactly, and say where it is, its font style, size and color.",
}

class EZCaptionDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EZCaptionDataset",
            display_name="EZ Dataset Captioner (OpenAI-compatible)",
            category="model/training",
            description="Captions every image in a folder with a vision LLM behind an OpenAI-compatible API (LM Studio, llama.cpp, Ollama, vLLM, OpenAI...) "
                        "and writes a .txt with the same name next to each image.",
            is_output_node=True,
            inputs=[
                io.String.Input("folder", default="", tooltip="Folder of images. Subfolders are included."),
                io.String.Input("base_url", default="http://127.0.0.1:1234/v1",
                                tooltip="OpenAI-compatible API root. LM Studio: http://127.0.0.1:1234/v1, Ollama: http://127.0.0.1:11434/v1, llama.cpp: http://127.0.0.1:8080/v1"),
                io.String.Input("model", default="", tooltip="Model id. Use the 'list models' button to pick one, empty uses the first model the server lists."),
                io.String.Input("api_key", default="", tooltip="Leave empty. Use the 'set API key' button to save the key for this base_url outside the workflow. A key typed here is moved there automatically, otherwise OPENAI_API_KEY is used."),
                io.Combo.Input("preset", options=list(PRESETS), default="detailed", tooltip="Caption style."),
                io.String.Input("prepend", default="", tooltip="Text put in front of every caption, e.g. a trigger word. A separating space or comma is added."),
                io.String.Input("system_prompt", multiline=True, default="", tooltip="Replaces the preset's instructions when not empty."),
                io.Boolean.Input("overwrite", default=False, tooltip="Re-caption images that already have a .txt."),
                io.Int.Input("max_image_size", default=1024, min=256, max=4096, step=64, tooltip="Longest side of the image sent to the model."),
                io.Int.Input("max_tokens", default=2048, min=16, max=32768, tooltip="Reply token limit. Thinking models spend tokens reasoning before the caption, so keep this high, non-thinking models stop on their own."),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.05),
                io.Int.Input("concurrency", default=1, min=1, max=16, tooltip="Parallel requests. Local servers usually want 1."),
                io.Int.Input("timeout", default=180, min=5, max=3600, tooltip="Seconds per request."),
                io.String.Input("extra_instructions", multiline=True, default="",
                                tooltip="Appended to the active system prompt (the preset, or system_prompt when set), e.g. \"Call the woman with silver hair 'kiri'.\""),
            ],
            outputs=[io.String.Output(display_name="folder", tooltip="The captioned folder, connect it to the trainer's dataset_paths so training runs after captioning.")],
        )

    @classmethod
    async def execute(cls, folder, base_url, model, api_key, preset, prepend, system_prompt, overwrite, max_image_size, max_tokens, temperature,
                      concurrency, timeout, extra_instructions) -> io.NodeOutput:
        warn_inline_key(api_key, "[EZ Captioner]")
        folder = os.path.abspath(os.path.expanduser(folder.strip().strip('"')))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        images = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, name))
        todo = [p for p in images if overwrite or not os.path.isfile(os.path.splitext(p)[0] + ".txt")]
        logging.info(f"[EZ Captioner] {len(images)} images, {len(todo)} to caption, {len(images) - len(todo)} already captioned")
        if not todo:
            return io.NodeOutput(folder)

        system = system_prompt.strip() or PRESETS[preset]
        if extra_instructions.strip():
            system += "\n\n" + extra_instructions.strip()
        prepend = prepend.strip()
        done = 0
        failed = []
        semaphore = asyncio.Semaphore(concurrency)
        pbar = comfy.utils.ProgressBar(len(todo))

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            model = await resolve_model(session, base_url, model, api_key)
            logging.info(f"[EZ Captioner] Using {model} at {base_url}, preset '{preset if not system_prompt.strip() else 'custom'}'" + (" + extra instructions" if extra_instructions.strip() else ""))

            async def caption(path):
                nonlocal done
                async with semaphore:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    image = await asyncio.to_thread(lambda: encode_image(ImageOps.exif_transpose(Image.open(path)), max_image_size))
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": [{"type": "text", "text": "Caption this image."}, {"type": "image_url", "image_url": {"url": image}}]},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    name = os.path.relpath(path, folder)
                    try:
                        text = await chat(session, base_url, api_key, payload, name, "[EZ Captioner]")
                    except RuntimeError as e:
                        failed.append(path)
                        pbar.update(1)
                        logging.warning(f"[EZ Captioner] {name} failed: {e}")
                        return
                    if prepend:
                        text = f"{prepend}{'' if prepend[-1] in ',.;:' else ','} {text}"
                    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    done += 1
                    pbar.update(1)
                    logging.info(f"[EZ Captioner] {done + len(failed)}/{len(todo)} {name}: {text[:120]}{'...' if len(text) > 120 else ''}")

            await asyncio.gather(*(caption(p) for p in todo))

        logging.info(f"[EZ Captioner] Done: {done} captioned, {len(failed)} failed")
        if failed and not done:
            raise RuntimeError(f"All {len(failed)} captions failed, check base_url, model and that the model supports images.")
        return io.NodeOutput(folder)
