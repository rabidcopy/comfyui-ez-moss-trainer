"""Shared OpenAI-compatible chat client for the captioner and the prompt writer."""
import asyncio
import base64
import io as pyio
import logging
import os
import re

import aiohttp
from aiohttp import web
from PIL import Image

import comfy.model_management
from server import PromptServer

RETRIES = 3
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TURN_RE = re.compile(r"<\|?(?:start_of_turn|end_of_turn|im_start|im_end|eot_id|start_header_id|end_header_id)\|?>")


class TransientError(Exception):
    pass


def _headers(api_key):
    key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


async def list_models(session, base_url, api_key):
    async with session.get(base_url.rstrip("/") + "/models", headers=_headers(api_key)) as r:
        r.raise_for_status()
        data = await r.json()
    return [m["id"] for m in data.get("data", [])]


async def resolve_model(session, base_url, model, api_key):
    if model.strip():
        return model.strip()
    models = await list_models(session, base_url, api_key)
    if not models:
        raise ValueError(f"{base_url} lists no models, set the model name.")
    return models[0]


def encode_image(img, max_size):
    """PIL image -> JPEG data URL with the longest side at most max_size."""
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clean_reply(text):
    text = THINK_RE.sub("", text)
    # some providers leak chat template turn markers, e.g. gemma's "<start_of_turn>model" arrives as "model model ..."
    text = TURN_RE.sub(" ", text).strip()
    text = re.sub(r"^(?:(?:model|assistant)\b[\s:]*)+", "", text)
    text = re.sub(r"^(here is|here's)[^:\n]*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(caption|prompt|enhanced prompt):\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())


async def chat(session, base_url, api_key, payload, label, log_prefix):
    """POST a chat completion and return the cleaned reply, retrying transient failures. Raises RuntimeError on final failure."""
    url = base_url.rstrip("/") + "/chat/completions"
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
            text = clean_reply(choice["message"].get("content") or "")
            if text:
                return text
            finish = choice.get("finish_reason")
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
            if finish == "length":
                # thinking models can spend the whole budget reasoning, retrying with the same limit rarely helps
                raise RuntimeError(f"reply hit max_tokens ({payload['max_tokens']}) before any answer" + (" while the model was still reasoning" if reasoning else "")
                                   + ", raise max_tokens or use a non-thinking model")
            raise TransientError(f"empty reply (finish_reason {finish}" + (", reasoning only" if reasoning else "") + ")")
        except (aiohttp.ClientError, asyncio.TimeoutError, TransientError) as e:
            error = str(e) or type(e).__name__
            if attempt == RETRIES:
                raise RuntimeError(error) from e
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            logging.info(f"{log_prefix} {label}: {error}, retrying in {delay:.0f}s ({attempt + 1}/{RETRIES})")
            await asyncio.sleep(delay)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected response format: {type(e).__name__} {e}") from e


@PromptServer.instance.routes.post("/ez_moss/caption_models")
async def caption_models_route(request):
    body = await request.json()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            return web.json_response({"models": sorted(await list_models(session, body.get("base_url", ""), body.get("api_key", "")))})
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
        return web.json_response({"error": str(e) or type(e).__name__}, status=502)
