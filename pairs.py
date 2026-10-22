import json
import logging
import os

import safetensors.torch
from PIL import Image

from .dataset import cache_texts, text_path
from .reference import MODES

NO_REFERENCES = 0.1
DROP_OUTFIT = 0.5
# (weight, reference modes), the caption leaves out the fields the references supply
PLANS = ((2, ("character",)), (3, ("style",)), (3, ("character+style",)), (3, ("character", "style")), (1, ("character", "character")), (1, ("style", "style")))
SUPPLIES = {"character": {"character"}, "style": {"style"}, "character+style": {"character", "style"}}
FIELDS = ("character", "outfit", "scene", "style")


def dhash(path):
    with Image.open(path) as img:
        img.draft("L", (64, 64))
        px = img.convert("L").resize((9, 8), Image.BILINEAR).tobytes()
    return sum(1 << i for i in range(64) if px[i + i // 8] > px[i + i // 8 + 1])


def caption(fields, drop):
    return " ".join(fields[k] for k in FIELDS if k not in drop and fields[k] and fields[k].lower().rstrip(".") != "none")


class ReferencePairs:
    """References for the images of tools/e621_pull.py folders captioned with the 'reference fields (JSON)' preset.

    Character references show the same single character by another artist, style references are by the same artist
    with no shared character, character+style references share both. Versions of one post and near duplicates are never
    paired. Model sheets and images without any reference are only used as references."""

    def __init__(self, items):
        self.items = items
        meta = {}
        for i, item in enumerate(items):
            sidecar = os.path.splitext(item["image"])[0] + ".json"
            if not os.path.isfile(sidecar):
                continue
            try:
                fields = json.loads(item["caption"])
            except json.JSONDecodeError:
                continue
            with open(sidecar, "r", encoding="utf-8") as f:
                side = json.load(f)
            meta[i] = {"artist": side["artist"], "characters": side["characters"], "sheet": side["sheet"], "related": set(side["related"]),
                       "fields": fields, "hash": dhash(item["image"])}

        by_artist, by_character = {}, {}
        for i, m in meta.items():
            by_artist.setdefault(m["artist"], []).append(i)
            if len(m["characters"]) == 1 and m["fields"]["visible"] != "no":
                by_character.setdefault(m["characters"][0], []).append(i)

        def distinct(a, b):
            return not meta[a]["related"] & meta[b]["related"] and (meta[a]["hash"] ^ meta[b]["hash"]).bit_count() > 10

        self.fields, self.candidates, self.plans, self.text = {}, {}, {}, {}
        for t, m in meta.items():
            same = by_character.get(m["characters"][0], []) if len(m["characters"]) == 1 and m["fields"]["visible"] != "no" else []
            refs = {"character": [r for r in same if meta[r]["artist"] != m["artist"] and distinct(t, r)],
                    "style": [r for r in by_artist[m["artist"]] if not set(meta[r]["characters"]) & set(m["characters"]) and distinct(t, r)],
                    "character+style": [r for r in same if meta[r]["artist"] == m["artist"] and distinct(t, r)]}
            plans = [(w, modes) for w, modes in PLANS if all(len(refs[mode]) >= modes.count(mode) for mode in modes)]
            if m["sheet"] or not plans:
                continue
            drops = {frozenset()}
            for _, modes in plans:
                drop = set().union(*(SUPPLIES[mode] for mode in modes))
                drops.add(frozenset(drop))
                if "character" in drop:
                    drops.add(frozenset(drop | {"outfit"}))
            self.fields[t] = m["fields"]
            self.candidates[t] = refs
            self.plans[t] = plans
            self.text[t] = {drop: text_path(items[t]["cache"], caption(m["fields"], drop)) for drop in drops}
        if not self.plans:
            raise ValueError("No reference pairs found. reference_training needs folders from tools/e621_pull.py captioned with the 'reference fields (JSON)' preset.")
        for i, item in enumerate(items):
            if i not in self.plans:
                item["repeats"] = 0
        counts = {modes: sum(any(p == modes for _, p in plans) for plans in self.plans.values()) for _, modes in PLANS}
        logging.info(f"[MOSS Trainer] Reference pairs: {len(self.plans)} training images, {len(meta) - len(self.plans)} only used as references, "
                     f"{len(meta)} of {len(items)} images have a sidecar and reference fields. Images per plan: "
                     + ", ".join(f"{' + '.join(mode + ' ref' for mode in modes)}: {c}" for modes, c in counts.items()))

    def cache_text(self, clip, batch_size):
        todo = {path: caption(self.fields[t], drop) for t, variants in self.text.items() for drop, path in variants.items() if not os.path.isfile(path)}
        if not todo:
            return
        logging.info(f"[MOSS Trainer] Caching {len(todo)} reference caption encodings ({len({p for v in self.text.values() for p in v.values()}) - len(todo)} already cached)")
        cache_texts(clip, todo, batch_size)

    def sample(self, i, rng):
        """(text encoding, [(latent, mode index)]) for training image i."""
        plans = self.plans[i]
        modes = () if rng.random() < NO_REFERENCES else rng.choices([p for _, p in plans], weights=[w for w, _ in plans])[0]
        picked = []
        for mode in modes:
            picked.append(rng.choice([r for r in self.candidates[i][mode] if r not in picked]))
        drop = set().union(*(SUPPLIES[mode] for mode in modes))
        if "character" in drop and rng.random() < DROP_OUTFIT:
            drop.add("outfit")
        cond = safetensors.torch.load_file(self.text[i][frozenset(drop)])["cond"]
        return cond, [(self.items[r]["latent"], MODES.index(mode)) for r, mode in zip(picked, modes)]
