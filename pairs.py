import json
import logging
import os

import safetensors.torch
from PIL import Image

from .dataset import cache_texts, text_path
from .reference import MODES

NO_REFERENCES = 0.1
DROP_OUTFIT = 0.5
# chance of a cropped copy of the image as its own character+style reference at the start and the end of a run,
# an easy pairing that teaches the adapter to read references before the harder cross-image ones take over
SELF_REFERENCE = (0.3, 0.1)
SELF_CROP = (0.5, 0.8)
# (weight, reference modes), the caption leaves out the fields the references supply
PLANS = ((2, ("character",)), (3, ("style",)), (3, ("character+style",)), (3, ("character", "style")), (1, ("character", "character")), (1, ("style", "style")))
SUPPLIES = {"character": {"character"}, "style": {"style"}, "character+style": {"character", "style"}}
FIELDS = ("character", "outfit", "scene", "style")
# chance that the caption keeps the field a reference supplies. The character field describes the character's look in
# words, a reference shows it. Leaving it out every time teaches the model that a caption without the field means the
# reference has to be read, and every prompt at inference has one.
KEEP_FIELD = 0.5


def supplied_fields(modes):
    return set().union(*(SUPPLIES[mode] for mode in modes)) if modes else set()


def drop_variants(modes):
    """The caption variants a plan trains on: the fields the references supply are left out, and kept the other half of
    the time. A character reference drops the outfit too, the reference shows the outfit it wants."""
    supplied = supplied_fields(modes)
    variants = {frozenset(supplied)}
    kept = supplied - {"character"} if "character" in supplied else supplied - {"style"}
    variants.add(frozenset(kept))
    if "character" in supplied:
        variants.add(frozenset(supplied | {"outfit"}))
        variants.add(frozenset(kept | {"outfit"}))
    return variants


def dhash(path):
    with Image.open(path) as img:
        img.draft("L", (64, 64))
        px = img.convert("L").resize((9, 8), Image.BILINEAR).tobytes()
    return sum(1 << i for i in range(64) if px[i + i // 8] > px[i + i // 8 + 1])


def caption(fields, drop):
    return " ".join(fields[k] for k in FIELDS if k not in drop and fields[k] and fields[k].lower().rstrip(".") != "none")


def crop(latent, rng):
    """Random crop of SELF_CROP of each side, so a self-reference is neither whole nor aligned with the image."""
    _, h, w = latent.shape
    ch, cw = max(1, round(h * rng.uniform(*SELF_CROP))), max(1, round(w * rng.uniform(*SELF_CROP)))
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    return latent[:, y:y + ch, x:x + cw]


class ReferencePairs:
    """References for images with a .json sidecar (artist, characters, sheet, related) captioned with the 'reference fields (JSON)' preset.

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
        self.meta = meta
        for t, m in meta.items():
            same = by_character.get(m["characters"][0], []) if len(m["characters"]) == 1 and m["fields"]["visible"] != "no" else []
            refs = {"character": [r for r in same if meta[r]["artist"] != m["artist"] and distinct(t, r)],
                    "style": [r for r in by_artist[m["artist"]] if not set(meta[r]["characters"]) & set(m["characters"]) and distinct(t, r)],
                    "character+style": [r for r in same if meta[r]["artist"] == m["artist"] and distinct(t, r)]}
            plans = [(w, modes) for w, modes in PLANS if all(len(refs[mode]) >= modes.count(mode) for mode in modes)]
            if m["sheet"] or not plans:
                continue
            drops = {frozenset(), frozenset(SUPPLIES["character+style"]), frozenset(SUPPLIES["character+style"] | {"outfit"})}
            for _, modes in plans:
                drops |= drop_variants(modes)
            self.fields[t] = m["fields"]
            self.candidates[t] = refs
            self.plans[t] = plans
            self.text[t] = {drop: text_path(items[t]["cache"], caption(m["fields"], drop)) for drop in drops}
        if not self.plans:
            raise ValueError("No reference pairs found. reference_training needs a .json sidecar with artist, characters, sheet and related next to each image, "
                             "and captions from the 'reference fields (JSON)' preset.")
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

    def sample(self, i, rng, progress=None):
        """(text encoding, [(latent, mode index)]) for training image i. progress is the fraction of the run done,
        None leaves out self-references, as validation does."""
        if progress is not None and rng.random() < SELF_REFERENCE[0] + (SELF_REFERENCE[1] - SELF_REFERENCE[0]) * progress:
            return self.self_sample(i, rng)
        plans = self.plans[i]
        modes = () if rng.random() < NO_REFERENCES else rng.choices([p for _, p in plans], weights=[w for w, _ in plans])[0]
        return self.plan_sample(i, modes, rng)

    def has_plan(self, i, modes):
        return any(p == modes for _, p in self.plans.get(i, ()))

    def plan_sample(self, i, modes, rng):
        """sample with references for the given modes, which must be one of image i's plans or empty."""
        picked = []
        for mode in modes:
            picked.append(rng.choice([r for r in self.candidates[i][mode] if r not in picked]))
        refs = [(self.items[r]["latent"], MODES.index(mode)) for r, mode in zip(picked, modes)]
        return self._cond(i, self._drop(modes, rng)), refs

    def self_sample(self, i, rng):
        """sample with a random crop of image i as its own character+style reference."""
        refs = [(crop(self.items[i]["latent"], rng), MODES.index("character+style"))]
        drop = set(SUPPLIES["character+style"])
        if rng.random() < DROP_OUTFIT:
            drop.add("outfit")
        return self._cond(i, drop), refs

    def wrong_reference(self, i, rng):
        """The latent of another character, for the validation entry that tells identity binding from copying. None when
        the dataset has no other character."""
        chars = set(self.meta[i]["characters"])
        related = self.meta[i]["related"]
        pool = [r for r, m in self.meta.items()
                if len(m["characters"]) == 1 and not chars & set(m["characters"]) and not related & m["related"]]
        if not pool:
            return None
        return self.items[rng.choice(pool)]["latent"]

    def _drop(self, modes, rng):
        """The fields the caption leaves out. The fields the references supply are left out most of the time and kept the
        rest, so the caption alone does not tell the model whether it has to read the reference."""
        supplied = supplied_fields(modes)
        if not supplied:
            return set()
        drop = supplied
        if rng.random() < KEEP_FIELD:
            drop = supplied - {"character"} if "character" in supplied else supplied - {"style"}
        if "character" in supplied and rng.random() < DROP_OUTFIT:
            drop.add("outfit")
        return drop

    def _cond(self, i, drop):
        return safetensors.torch.load_file(self.text[i][frozenset(drop)])["cond"]
