"""Whole-image RGB classification and online augmentation; no box/ROI inputs."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

MODALITIES = {"standard", "fluorescence"}
PROFILES = {"baseline", "geometry", "modality"}
CLASS_NAMES = ["normal_candidate", "lesion_candidate"]
CLASS_FOLDERS = ["0_normal", "1_lesion"]


@dataclass(frozen=True)
class AugmentationPolicy:
    version: int = 1
    profile: str = "geometry"
    horizontal_flip_probability: float = 0.5
    rotation_probability: float = 0.5
    rotation_degrees: float = 10.0
    standard_gain_probability: float = 0.3
    standard_gain_range: tuple[float, float] = (.90, 1.10)
    fluorescence_gain_probability: float = 0.0
    fluorescence_gain_range: tuple[float, float] = (.95, 1.05)

    def __post_init__(self):
        if self.version != 1 or self.profile not in PROFILES:
            raise ValueError("Unsupported policy version/profile")
        for name in ("horizontal_flip_probability", "rotation_probability",
                     "standard_gain_probability", "fluorescence_gain_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 <= self.rotation_degrees <= 15:
            raise ValueError("rotation_degrees must be in [0,15]")
        for name, lower, upper in (("standard_gain_range", .90, 1.10),
                                   ("fluorescence_gain_range", .95, 1.05)):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high)
                    and lower <= low <= high <= upper):
                raise ValueError(f"Invalid {name}")

    @classmethod
    def from_yaml(cls, path, profile=None):
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Augmentation YAML must be a mapping")
        if profile is not None:
            values["profile"] = profile
        return cls(**values)


def modality_from_path(path):
    p = Path(path)
    if (p.parent.name not in CLASS_FOLDERS or p.parent.parent.name != "images"
            or p.parent.parent.parent.name not in MODALITIES):
        raise ValueError("Expected .../{modality}/images/{0_normal,1_lesion}/image")
    return p.parent.parent.parent.name


def augment_image(image, modality, policy, training=True, rng=None):
    """Return a complete RGB image and replay; expand rotation canvas, never crop."""
    if modality not in MODALITIES:
        raise ValueError("Unconfirmed modality")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB image")
    rng = rng or random
    image = image.copy()
    replay = {"flip": False, "rotation": 0., "gain": 1., "gain_skipped_clipping": False}
    if not training or policy.profile == "baseline":
        return image, replay
    if rng.random() < policy.horizontal_flip_probability:
        image = np.ascontiguousarray(image[:, ::-1])
        replay["flip"] = True
    if rng.random() < policy.rotation_probability and policy.rotation_degrees:
        angle = rng.uniform(-policy.rotation_degrees, policy.rotation_degrees)
        image = np.asarray(Image.fromarray(image).rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=True,
            fillcolor=(114, 114, 114))).copy()
        replay["rotation"] = angle
    if policy.profile == "modality":
        if rng.random() < getattr(policy, f"{modality}_gain_probability"):
            gain = rng.uniform(*getattr(policy, f"{modality}_gain_range"))
            if float(image.max()) * gain <= 255:
                image = np.rint(image.astype(np.float32) * gain).astype(np.uint8)
                replay["gain"] = gain
            else:
                replay["gain_skipped_clipping"] = True
    return image, replay


def letterbox_full_image(image, imgsz):
    """Preserve the full field of view with aspect-ratio resize and padding."""
    if not isinstance(imgsz, int) or imgsz < 32:
        raise ValueError("imgsz must be an integer >=32")
    h, w = image.shape[:2]
    ratio = imgsz / max(h, w)
    resized = Image.fromarray(image).resize(
        (max(1, round(w * ratio)), max(1, round(h * ratio))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(resized, ((imgsz-resized.width)//2, (imgsz-resized.height)//2))
    return np.asarray(canvas).copy()


def verify_checksums(root):
    root = Path(root).resolve()
    count = 0
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        path = (root / relative.lstrip("*")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Checksum target missing or outside package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {path.name}")
        count += 1
    if not count:
        raise ValueError("Empty checksum manifest")
    return count


def validate_handoff(dataset_json):
    """Check frozen whole-image labels, hashes, inventory and source-group split."""
    path = Path(dataset_json).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Whole-image classification requires dataset.json; YOLO YAML is historical")
    d = json.loads(path.read_text(encoding="utf-8"))
    if (d.get("schema_version") != 1 or d.get("task") != "image_classification"
            or d.get("class_names") != CLASS_NAMES or not d.get("records")):
        raise ValueError("Invalid whole-image classification manifest/class mapping")
    root = path.parent
    seen_ids, group_splits, expected = set(), {}, set()
    counts = {s: {"images": 0, "groups": set(), "classes": Counter(), "modalities": Counter()}
              for s in ("train", "val", "test")}
    for r in d["records"]:
        split, mode, label = r["split"], r["modality"], r["label"]
        if split not in counts or mode not in MODALITIES or type(label) is not int or label not in (0, 1):
            raise ValueError("Invalid split/modality/image label")
        image = (root / r["image_relpath"]).resolve()
        directory = root / split / mode / "images" / CLASS_FOLDERS[label]
        if (not image.is_relative_to(root) or image.parent != directory
                or image.stem != r["image_id"] or not image.is_file()):
            raise ValueError("Image path/class folder/modality differs from manifest")
        if not r["image_id"] or r["image_id"] in seen_ids or image in expected:
            raise ValueError("Duplicate image ID/path")
        seen_ids.add(r["image_id"])
        expected.add(image)
        group = r["group_id"]
        if not group or group_splits.setdefault(group, split) != split:
            raise ValueError("Source-folder group crosses splits or is empty")
        if hashlib.sha256(image.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Image hash mismatch")
        with Image.open(image) as decoded:
            if decoded.getexif():
                raise ValueError("EXIF found; publish a de-identified version before use")
        # PNG getexif may load/close the stream, so verify a fresh handle.
        with Image.open(image) as decoded:
            decoded.verify()
        c = counts[split]
        c["images"] += 1
        c["groups"].add(group)
        c["classes"][label] += 1
        c["modalities"][mode] += 1
    actual = {p.resolve() for split in counts for mode in MODALITIES
              for p in (root / split / mode / "images").rglob("*")
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if actual != expected:
        raise ValueError("Image inventory differs from manifest")
    for split, c in counts.items():
        if not c["images"]:
            raise ValueError(f"Empty {split} split")
        c["groups"] = len(c["groups"])
        c["classes"] = dict(c["classes"])
        c["modalities"] = dict(c["modalities"])
    return {"task": "image_classification", "images": len(seen_ids),
            "dataset_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "split_unit": "inherited source-folder groups; patient independence unconfirmed",
            "class_names": CLASS_NAMES, "splits": counts}


class ImageClassificationDataset:
    """PyTorch map-style dataset returning full-image tensor and scalar label."""
    def __init__(self, dataset_json, split, policy=None, imgsz=640, modalities=None):
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        self.root = Path(dataset_json).resolve().parent
        data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
        if data.get("task") != "image_classification" or data.get("class_names") != CLASS_NAMES:
            raise ValueError("Use whole-image classification manifest")
        modes = set(modalities or MODALITIES)
        if not modes or not modes <= MODALITIES:
            raise ValueError("Invalid modality filter")
        self.records = [r for r in data["records"] if r["split"] == split
                        and r["modality"] in modes]
        if not self.records:
            raise ValueError(f"No {split} samples for selected modality")
        self.training = split == "train"
        self.policy = policy or AugmentationPolicy()
        self.imgsz = imgsz

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        r = self.records[index]
        with Image.open(self.root / r["image_relpath"]) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
        pixels, _ = augment_image(pixels, r["modality"], self.policy, training=self.training)
        pixels = letterbox_full_image(pixels, self.imgsz)
        array = pixels.astype(np.float32).transpose(2, 0, 1) / 255.
        # Channel normalization only; never use a pretrained center-crop transform.
        array = ((array - np.array([.485, .456, .406], np.float32)[:, None, None])
                 / np.array([.229, .224, .225], np.float32)[:, None, None])
        return {"image": torch.from_numpy(np.ascontiguousarray(array)),
                "label": int(r["label"]), "image_id": r["image_id"], "modality": r["modality"]}
