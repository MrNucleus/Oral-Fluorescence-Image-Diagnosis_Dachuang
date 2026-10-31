import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/oral_v7_resnet18_baseline_20261026/code/src"))
from oral_fluorescence_detection.image_cache import (
    build_cache, CachedImageClassificationDataset, sha256,
)
from oral_fluorescence_detection.online_augmentation import (
    ImageClassificationDataset, AugmentationPolicy,
)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        package = self.root / "source"
        package.mkdir()
        rows = []
        pixels = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)
        for split in ("train", "val", "test"):
            target = package / f"{split}.jpg"
            Image.fromarray(pixels).save(target)
            rows.append(dict(image_id=split, split=split, modality="standard", label=1,
                             image_relpath=target.name, sha256=sha256(target)))
        self.data = package / "dataset.json"
        self.data.write_text(json.dumps(dict(task="image_classification",
            class_names=["normal_candidate", "lesion_candidate"], records=rows)), encoding="utf-8")
        self.cache = self.root / "cache"
        build_cache(self.data, self.cache, imgsz=64)

    def test_validation_exact_no_test_payload_and_no_jpg_dependency(self):
        cached = CachedImageClassificationDataset(self.data, "val", imgsz=64, cache_dir=self.cache)
        original = ImageClassificationDataset(self.data, "val", imgsz=64)
        expected = original[0]["image"]
        (self.data.parent / "val.jpg").unlink()
        self.assertTrue(torch.equal(cached[0]["image"], expected))
        self.assertFalse((self.cache / "test").exists())
        with self.assertRaises(ValueError):
            CachedImageClassificationDataset(self.data, "test", imgsz=64, cache_dir=self.cache)

    def test_corruption_and_stale_manifest_rejected(self):
        payload = next((self.cache / "train").rglob("*.npy"))
        with payload.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaises(ValueError):
            CachedImageClassificationDataset(self.data, "train", imgsz=64, cache_dir=self.cache)
        self.data.write_text(self.data.read_text() + " ")
        with self.assertRaises(ValueError):
            CachedImageClassificationDataset(self.data, "val", imgsz=64, cache_dir=self.cache)

    def test_train_augmentation_is_live_and_reproducible(self):
        policy = AugmentationPolicy(horizontal_flip_probability=1., rotation_probability=1.)
        cached = CachedImageClassificationDataset(self.data, "train", policy, imgsz=64, cache_dir=self.cache)
        random.seed(42)
        first = cached[0]["image"]
        second = cached[0]["image"]
        self.assertFalse(torch.equal(first, second))
        random.seed(42)
        self.assertTrue(torch.equal(first, cached[0]["image"]))
        self.assertEqual(tuple(first.shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
