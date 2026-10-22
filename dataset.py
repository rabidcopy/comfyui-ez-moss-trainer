import hashlib
import logging
import math
import os
import random

import numpy as np
import safetensors.torch
import torch
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jxl", ".avif"}
BUCKET_STEP = 64  # MOSS latents are 1/16 scale, 64px keeps the bucket count low like OneTrainer


def parse_dataset_paths(text):
    """One folder per line, optionally `folder | repeats`. Lines starting with # are ignored."""
    folders = []
    for line in text.splitlines():
        line = line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        path, _, repeats = line.partition("|")
        path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset folder not found: {path}")
        folders.append((path, max(1, int(repeats.strip() or 1))))
    if not folders:
        raise ValueError("No dataset folders given. Put one folder of image + .txt caption pairs per line.")
    return folders


def scan_folder(folder):
    items = []
    missing = 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            caption_path = os.path.join(root, stem + ".txt")
            if os.path.isfile(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            else:
                caption = ""
                missing += 1
            items.append({"image": os.path.join(root, name), "caption": caption})
    return items, missing


def make_buckets(resolution, enabled):
    if not enabled:
        return [(resolution, resolution)]
    area = resolution * resolution
    lo, hi = max(BUCKET_STEP, resolution // 2 // BUCKET_STEP * BUCKET_STEP), resolution * 2
    buckets = set()
    for w in range(lo, hi + 1, BUCKET_STEP):
        h = area // w // BUCKET_STEP * BUCKET_STEP
        if lo <= h <= hi:
            buckets.add((w, h))
            buckets.add((h, w))
    return sorted(buckets)


def closest_bucket(width, height, buckets):
    aspect = math.log(width / height)
    return min(buckets, key=lambda b: (abs(math.log(b[0] / b[1]) - aspect), -b[0] * b[1]))


def load_image(path, size):
    """Resize to cover the bucket, then center crop. Returns [1, H, W, 3] in 0-1."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    w, h = size
    scale = max(w / img.width, h / img.height)
    rw, rh = max(w, round(img.width * scale)), max(h, round(img.height * scale))
    img = img.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - w) // 2, (rh - h) // 2
    img = img.crop((left, top, left + w, top + h))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]


def _key(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:20]


def _cache_dir(folder, cache_root):
    if cache_root:
        return os.path.join(cache_root, _key(folder))
    return os.path.join(folder, ".moss_cache")


def encode_text(clip, caption):
    tokens = clip.tokenize(caption)
    return clip.encode_from_tokens(tokens)[0].to(torch.bfloat16).cpu().contiguous()


def encode_texts(clip, captions):
    """encode_text for several captions in one padded batch. Each caption is padded at its end with masked pad tokens and
    the text encoder is causal, so the result matches encoding them one at a time."""
    tokens = [clip.tokenize(caption) for caption in captions]
    key = next(iter(tokens[0]))
    sections = [t[key][0] for t in tokens]
    longest = max(len(s) for s in sections)
    pad = getattr(clip.tokenizer, key).pad_token
    cond = clip.encode_from_tokens({key: [s + [(pad, 1.0)] * (longest - len(s)) for s in sections]})[0]
    cond = cond.unflatten(0, (len(sections), longest)).to(torch.bfloat16).cpu()
    return [c[:len(s)].contiguous() for c, s in zip(cond, sections)]


def cache_texts(clip, todo, batch_size):
    """Encodes {path: caption} in batches of similar length and saves each encoding to its path."""
    pbar = comfy.utils.ProgressBar(len(todo))
    todo = sorted(todo.items(), key=lambda x: len(x[1]))
    for n in range(0, len(todo), batch_size):
        comfy.model_management.throw_exception_if_processing_interrupted()
        batch = todo[n:n + batch_size]
        for (path, _), cond in zip(batch, encode_texts(clip, [caption for _, caption in batch])):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            safetensors.torch.save_file({"cond": cond}, path)
        pbar.update(len(batch))
        if (n + len(batch)) // 100 > n // 100:
            logging.info(f"[MOSS Trainer]   text {n + len(batch)}/{len(todo)}")


def text_path(cache, caption):
    return os.path.join(cache, "text", _key(caption) + ".safetensors")


def cached_cond(clip, caption, cache_root, folder):
    path = text_path(_cache_dir(folder, cache_root), caption)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file({"cond": encode_text(clip, caption)}, path)
    return safetensors.torch.load_file(path)["cond"]


def build_dataset(clip, vae, latent_format, folders, resolution, bucketing, cache_root, text_batch, encode_captions=True):
    """Scan the folders, assign buckets and fill the latent / text caches. encode_captions off leaves the .txt captions unencoded."""
    buckets = make_buckets(resolution, bucketing)
    items = []
    for folder, repeats in folders:
        found, missing = scan_folder(folder)
        logging.info(f"[MOSS Trainer] {folder}: {len(found)} images, repeats {repeats}" + (f", {missing} without a .txt caption (trained unconditionally)" if missing else ""))
        cache = _cache_dir(folder, cache_root)
        os.makedirs(os.path.join(cache, "latents"), exist_ok=True)
        os.makedirs(os.path.join(cache, "text"), exist_ok=True)
        for item in found:
            st = os.stat(item["image"])
            with Image.open(item["image"]) as img:
                w, h = ImageOps.exif_transpose(img).size
            item["bucket"] = closest_bucket(w, h, buckets)
            item["repeats"] = repeats
            item["cache"] = cache
            item["latent_path"] = os.path.join(cache, "latents", _key(item["image"], st.st_mtime_ns, st.st_size, *item["bucket"]) + ".safetensors")
            item["text_path"] = text_path(cache, item["caption"])
        items += found
    if not items:
        raise ValueError("The dataset folders contain no images.")

    todo = [i for i in items if not os.path.isfile(i["latent_path"])]
    if todo:
        logging.info(f"[MOSS Trainer] Caching {len(todo)} latents ({len(items) - len(todo)} already cached)")
        pbar = comfy.utils.ProgressBar(len(todo))
        for n, item in enumerate(todo):
            comfy.model_management.throw_exception_if_processing_interrupted()
            latent = latent_format.process_in(vae.encode(load_image(item["image"], item["bucket"])))
            safetensors.torch.save_file({"latent": latent[0].to(torch.bfloat16).cpu().contiguous()}, item["latent_path"])
            pbar.update(1)
            if (n + 1) % 25 == 0:
                logging.info(f"[MOSS Trainer]   latents {n + 1}/{len(todo)}")

    todo = {i["text_path"]: i["caption"] for i in items if encode_captions and not os.path.isfile(i["text_path"])}
    if todo:
        logging.info(f"[MOSS Trainer] Caching {len(todo)} text encodings")
        cache_texts(clip, todo, text_batch)

    for item in items:
        item["latent"] = safetensors.torch.load_file(item["latent_path"])["latent"]
        if encode_captions:
            item["cond"] = safetensors.torch.load_file(item["text_path"])["cond"]

    counts = {}
    for item in items:
        counts[item["bucket"]] = counts.get(item["bucket"], 0) + item["repeats"]
    logging.info("[MOSS Trainer] Buckets (WxH: samples per epoch): " + ", ".join(f"{w}x{h}: {c}" for (w, h), c in sorted(counts.items(), key=lambda x: -x[1])))
    return items


class BucketSampler:
    """Yields batches of item indices that share a bucket, reshuffled every epoch."""

    def __init__(self, items, batch_size, seed):
        self.items = items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.epoch = 0
        self.batches = []

    def _new_epoch(self):
        by_bucket = {}
        for i, item in enumerate(self.items):
            by_bucket.setdefault(item["bucket"], []).extend([i] * item["repeats"])
        self.batches = []
        for idx in by_bucket.values():
            self.rng.shuffle(idx)
            self.batches += [idx[j:j + self.batch_size] for j in range(0, len(idx), self.batch_size)]
        self.rng.shuffle(self.batches)
        self.epoch += 1

    def __len__(self):
        return sum(math.ceil(sum(i["repeats"] for i in self.items if i["bucket"] == b) / self.batch_size) for b in {i["bucket"] for i in self.items})

    def next(self):
        if not self.batches:
            self._new_epoch()
        return self.batches.pop()
