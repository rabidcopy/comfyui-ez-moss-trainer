import asyncio
import base64
import io as pyio
import logging
import os
import re

import aiohttp
from aiohttp import web
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
from comfy_api.latest import io
from server import PromptServer

from .dataset import IMAGE_EXTENSIONS

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

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
RETRIES = 3
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class TransientError(Exception):
    pass


TURN_RE = re.compile(r"<\|?(?:start_of_turn|end_of_turn|im_start|im_end|eot_id|start_header_id|end_header_id)\|?>")


def _api_key(api_key):
    return api_key.strip() or os.environ.get("OPENAI_API_KEY", "")


def _headers(api_key):
    key = _api_key(api_key)
    return {"Authorization": f"Bearer {key}"} if key else {}


async def list_models(session, base_url, api_key):
    async with session.get(base_url.rstrip("/") + "/models", headers=_headers(api_key)) as r:
        r.raise_for_status()
        data = await r.json()
    return [m["id"] for m in data.get("data", [])]


def encode_image(path, max_size):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clean_caption(text):
    text = THINK_RE.sub("", text)
    # some providers leak chat template turn markers, e.g. gemma's "<start_of_turn>model" arrives as "model model ..."
    text = TURN_RE.sub(" ", text).strip()
    text = re.sub(r"^(?:(?:model|assistant)\b[\s:]*)+", "", text)
    text = re.sub(r"^(here is|here's)[^:\n]*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^caption:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())


@PromptServer.instance.routes.post("/ez_moss/caption_models")
async def caption_models_route(request):
    body = await request.json()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            return web.json_response({"models": sorted(await list_models(session, body.get("base_url", ""), body.get("api_key", "")))})
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
        return web.json_response({"error": str(e) or type(e).__name__}, status=502)


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
                io.String.Input("api_key", default="", tooltip="Leave empty to use the OPENAI_API_KEY environment variable. A key typed here is saved in the workflow and image metadata."),
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
        url = base_url.rstrip("/") + "/chat/completions"
        done = 0
        failed = []
        semaphore = asyncio.Semaphore(concurrency)
        pbar = comfy.utils.ProgressBar(len(todo))

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            if not model.strip():
                models = await list_models(session, base_url, api_key)
                if not models:
                    raise ValueError(f"{base_url} lists no models, set the model name.")
                model = models[0]
            logging.info(f"[EZ Captioner] Using {model} at {base_url}, preset '{preset if not system_prompt.strip() else 'custom'}'" + (" + extra instructions" if extra_instructions.strip() else ""))

            async def caption(path):
                nonlocal done
                async with semaphore:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    payload = {
                        "model": model.strip(),
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Caption this image."},
                                {"type": "image_url", "image_url": {"url": await asyncio.to_thread(encode_image, path, max_image_size)}},
                            ]},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    name = os.path.relpath(path, folder)
                    for attempt in range(RETRIES + 1):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        retry_after = None
                        try:
                            async with session.post(url, json=payload, headers=_headers(api_key)) as r:
                                if r.status != 200:
                                    if r.status in RETRY_STATUS:
                                        retry_after = r.headers.get("Retry-After")
                                        raise TransientError(f"HTTP {r.status}: {(await r.text())[:300]}")
                                    raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
                                choice = (await r.json())["choices"][0]
                            text = clean_caption(choice["message"].get("content") or "")
                            if not text:
                                finish = choice.get("finish_reason")
                                reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
                                if finish == "length":
                                    # thinking models can spend the whole budget reasoning, retrying with the same limit rarely helps
                                    raise RuntimeError(f"reply hit max_tokens ({max_tokens}) before any caption" + (" while the model was still reasoning" if reasoning else "")
                                                       + ", raise max_tokens or use a non-thinking model")
                                raise TransientError(f"empty caption (finish_reason {finish}" + (", reasoning only" if reasoning else "") + ")")
                            break
                        except (aiohttp.ClientError, asyncio.TimeoutError, TransientError) as e:
                            error = str(e) or type(e).__name__
                            if attempt < RETRIES:
                                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
                                logging.info(f"[EZ Captioner] {name}: {error}, retrying in {delay:.0f}s ({attempt + 1}/{RETRIES})")
                                await asyncio.sleep(delay)
                                continue
                        except (RuntimeError, KeyError, IndexError) as e:
                            error = str(e) or type(e).__name__
                        failed.append(path)
                        pbar.update(1)
                        logging.warning(f"[EZ Captioner] {name} failed: {error}")
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
