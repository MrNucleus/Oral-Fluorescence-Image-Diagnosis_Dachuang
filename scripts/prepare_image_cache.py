"""Build/verify a private cache and benchmark loading without training or test access."""
import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments/oral_v7_resnet18_baseline_20261026/code"
sys.path.insert(0, str(CODE / "src"))

from oral_fluorescence_detection.image_cache import build_cache, CachedImageClassificationDataset
from oral_fluorescence_detection.online_augmentation import ImageClassificationDataset
from oral_fluorescence_detection.image_classification import seed_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import psutil
    from torch.utils.data import DataLoader, Subset
    torch.set_num_threads(2)
    if args.build:
        build_cache(args.data, args.cache_dir)
    cached = {s: CachedImageClassificationDataset(args.data, s, cache_dir=args.cache_dir)
              for s in ("train", "val")}
    original = {s: ImageClassificationDataset(args.data, s) for s in ("train", "val")}
    equal = 0
    for i in range(len(original["val"])):
        a, b = original["val"][i], cached["val"][i]
        if not torch.equal(a["image"], b["image"]):
            raise ValueError("Validation tensor differs from original pipeline")
        equal += 1
    checked = 0
    for split in ("train", "val"):
        if original[split].records != cached[split].records:
            raise ValueError("Dataset membership changed")
        for i in range(len(cached[split])):
            sample = cached[split][i]
            if sample["image"].shape != (3, 640, 640) or not torch.isfinite(sample["image"]).all():
                raise ValueError("Invalid cached output")
            checked += 1
    indices = random.Random(42).sample(range(len(original["train"])),
                                     min(args.samples, len(original["train"])))
    results = []
    for repetition in range(2):
        for name, dataset in (("original", original["train"]), ("cached_v1", cached["train"])):
            for workers in (0, 2):
                random.seed(42)
                np.random.seed(42)
                loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=workers,
                    worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
                observations = []
                stop = threading.Event()
                def observe():
                    process = psutil.Process()
                    while not stop.is_set():
                        rss = process.memory_info().rss
                        for child in process.children(recursive=True):
                            try:
                                rss += child.memory_info().rss
                            except psutil.Error:
                                pass
                        observations.append((rss, psutil.virtual_memory().available))
                        stop.wait(.2)
                monitor = threading.Thread(target=observe, daemon=True)
                monitor.start()
                start = time.perf_counter()
                first_batch_seconds = None
                count = 0
                for batch in loader:
                    if first_batch_seconds is None:
                        first_batch_seconds = time.perf_counter() - start
                    count += len(batch["label"])
                elapsed = time.perf_counter() - start
                stop.set()
                monitor.join()
                result = dict(pipeline=name, workers=workers, repetition=repetition+1,
                    images=count, seconds=elapsed, images_per_second=count/elapsed,
                    first_batch_seconds=first_batch_seconds,
                    sampled_peak_process_tree_rss_MiB=max(x[0] for x in observations)/2**20,
                    sampled_min_system_available_MiB=min(x[1] for x in observations)/2**20)
                results.append(result)
                print(json.dumps(result), flush=True)
    report = dict(created=datetime.now().astimezone().isoformat(),
        scope="CPU loader only, concurrent formal training; no GPU model or test evaluation",
        validation_bitwise_equal=equal, checked_outputs=checked,
        cache_manifest_sha256=cached["train"].cache_manifest_sha256,
        cache_bytes=sum(p.stat().st_size for p in Path(args.cache_dir).rglob("*.npy")),
        benchmark=results)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verification and benchmark saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
