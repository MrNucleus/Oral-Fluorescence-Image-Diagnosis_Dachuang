"""Versioned whole-image cache. Test images are never materialized here."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .online_augmentation import ImageClassificationDataset, augment_image, letterbox_full_image

VERSION = "resize-before-augmentation-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash():
    return sha256(__file__) + ":" + sha256(Path(__file__).with_name("online_augmentation.py"))


def build_cache(data_json, destination, imgsz=640):
    data_json, destination = Path(data_json).resolve(), Path(destination).resolve()
    if destination.is_relative_to(data_json.parent):
        raise ValueError("Cache must be outside frozen image package")
    if imgsz < 64:
        raise ValueError("imgsz must be >=64")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=VERSION, imgsz=imgsz, dataset_sha256=sha256(data_json),
                    implementation_hash=implementation_hash(), records=[])
    for split in ("train", "val"):
        dataset = ImageClassificationDataset(data_json, split, imgsz=imgsz)
        for index, row in enumerate(dataset.records):
            source = data_json.parent / row["image_relpath"]
            source_hash = sha256(source)
            if source_hash != row["sha256"]:
                raise ValueError("Source hash mismatch; preserve source and investigate")
            if split == "val":
                array = dataset[index]["image"].numpy()
            else:
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    ratio = min(1., imgsz / max(image.size))
                    array = np.asarray(image.resize(
                        tuple(max(1, round(side * ratio)) for side in image.size),
                        Image.Resampling.BILINEAR)).copy()
            relative = Path(split) / row["modality"] / f"{index:05d}.npy"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, array, allow_pickle=False)
            manifest["records"].append(dict(image_id=row["image_id"], split=split,
                modality=row["modality"], label=row["label"], source_sha256=source_hash,
                path=relative.as_posix(), sha256=sha256(target), shape=list(array.shape),
                dtype=str(array.dtype)))
            if (index + 1) % 100 == 0:
                print(f"Cache {split}: {index + 1}/{len(dataset)}", flush=True)
    # Written last: an interrupted build never masquerades as a ready cache.
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class CachedImageClassificationDataset(ImageClassificationDataset):
    def __init__(self, data_json, split, policy=None, imgsz=640, modalities=None, cache_dir=None):
        if split not in ("train", "val"):
            raise ValueError("Cache only supports train/val")
        super().__init__(data_json, split, policy, imgsz, modalities)
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_manifest_sha256 = sha256(self.cache_dir / "manifest.json")
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["version"] != VERSION or manifest["imgsz"] != imgsz
                or manifest["dataset_sha256"] != sha256(data_json)
                or manifest["implementation_hash"] != implementation_hash()):
            raise ValueError("Stale or incompatible cache; build a new version")
        indexed = {(r["split"], r["image_id"]): r for r in manifest["records"]}
        if len(indexed) != len(manifest["records"]):
            raise ValueError("Duplicate cache records")
        self.cached = []
        for row in self.records:
            entry = indexed[(split, row["image_id"])]
            if (entry["label"] != row["label"] or entry["modality"] != row["modality"]
                    or entry["source_sha256"] != row["sha256"]):
                raise ValueError("Cache labels or source identity differ")
            path = (self.cache_dir / entry["path"]).resolve()
            if not path.is_relative_to(self.cache_dir) or sha256(path) != entry["sha256"]:
                raise ValueError("Cache payload hash or path invalid")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_dtype = np.uint8 if split == "train" else np.float32
            valid_shape = (array.ndim == 3 and array.shape[2] == 3
                           and max(array.shape[:2]) <= imgsz) if split == "train" else array.shape == (3, imgsz, imgsz)
            if array.dtype != expected_dtype or not valid_shape or list(array.shape) != entry["shape"]:
                raise ValueError("Invalid cached array")
            self.cached.append(path)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        pixels = np.load(self.cached[index], mmap_mode="r", allow_pickle=False)
        if self.training:
            pixels, _ = augment_image(pixels, row["modality"], self.policy, training=True)
            pixels = letterbox_full_image(pixels, self.imgsz)
            array = pixels.astype(np.float32).transpose(2, 0, 1) / 255.
            array = ((array - np.array([.485, .456, .406], np.float32)[:, None, None])
                     / np.array([.229, .224, .225], np.float32)[:, None, None])
        else:
            array = pixels
        return dict(image=torch.from_numpy(np.array(array, copy=True, order="C")),
                    label=int(row["label"]), image_id=row["image_id"], modality=row["modality"])
