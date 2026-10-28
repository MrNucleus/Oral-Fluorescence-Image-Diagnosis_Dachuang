"""Check frozen complete-image inputs and classification loader; never train a model."""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oral_fluorescence_detection.online_augmentation import (  # noqa: E402
    AugmentationPolicy,
    ImageClassificationDataset,
    validate_handoff,
    verify_checksums,
)


def inventory(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="完整图像分类包的 dataset.json")
    parser.add_argument("--augmentation", type=Path,
                        default=PROJECT_ROOT / "configs" / "online_augmentation.yaml")
    parser.add_argument("--profile", choices=["baseline", "geometry", "modality"])
    parser.add_argument("--modalities", nargs="+", choices=["standard", "fluorescence"],
                        help="仅检查指定模态；不改变包内图像标签或划分")
    parser.add_argument("--all", action="store_true", help="读取全部图像；否则每split每模态2张")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, help="新的报告目录，必须位于数据包外")
    args = parser.parse_args()
    data_json = args.data.resolve()
    root = data_json.parent
    if data_json.name != "dataset.json":
        raise ValueError("Use the current whole-image classification dataset.json; YOLO YAML is retired")
    output = (args.output or PROJECT_ROOT / "outputs" / "whole_image_augmentation_checks" /
              datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Output must be outside the frozen dataset")
    # Validate before creating a report directory; failures do not leave partial PASS reports.
    before = inventory(root)
    checked_before = verify_checksums(root)
    summary = validate_handoff(data_json)
    config = json.loads(data_json.read_text(encoding="utf-8-sig"))
    policy = AugmentationPolicy.from_yaml(args.augmentation, args.profile)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    audited = {}
    selected_modes = set(args.modalities or ("standard", "fluorescence"))
    for split in ("train", "val", "test"):
        expected = [row for row in config["records"] if row["split"] == split
                    and row["modality"] in selected_modes]
        dataset = ImageClassificationDataset(data_json, split, policy=policy,
                                             imgsz=args.imgsz, modalities=args.modalities)
        if len(dataset) != len(expected):
            raise ValueError("Classification loader removed input images")
        modes = {mode: 0 for mode in sorted(selected_modes)}
        labels = {0: 0, 1: 0}
        checked = 0
        for index, row in enumerate(expected):
            mode = row["modality"]
            if not args.all and modes[mode] >= 2:
                continue
            item = dataset[index]
            image, label = item["image"], item["label"]
            if (item["image_id"] != row["image_id"] or item["modality"] != mode
                    or isinstance(label, bool) or not isinstance(label, int)
                    or label != row["label"]):
                raise ValueError("Whole-image identity, modality or scalar class changed")
            if any(key in item for key in ("boxes", "bboxes", "cls")):
                raise ValueError("Detection targets are forbidden in the classification loader")
            if (image.dtype != torch.float32 or image.shape != (3, args.imgsz, args.imgsz)
                    or not torch.isfinite(image).all()):
                raise ValueError("Invalid complete-image classification tensor")
            if split != "train":
                again = dataset[index]
                if label != again["label"] or not torch.equal(image, again["image"]):
                    raise ValueError("Validation/test transform is stochastic")
            checked += 1
            modes[mode] += 1
            labels[label] += 1
        audited[split] = dict(images_loaded=checked, scalar_labels_loaded=checked,
                              class_counts=labels, modalities=modes,
                              augmentation_enabled=split == "train" and policy.profile != "baseline",
                              test_used_for_training=False)
    checked_after = verify_checksums(root)
    if before != inventory(root) or checked_before != checked_after:
        raise ValueError("Dataset file inventory changed")
    record = dict(status="PASS", task="image_classification", training_executed=False,
                  profile=policy.profile, modalities=sorted(selected_modes),
                  input_checksums_verified_before_and_after=checked_after,
                  input_inventory_unchanged=True, preflight=summary, loader=audited,
                  all_images_loaded=args.all, imgsz=args.imgsz,
                  scope="Whole-image structural/software audit; no clinical label confirmation")
    output.mkdir(parents=True, exist_ok=False)
    report = output / "validation.json"
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"验证报告：{report}")


if __name__ == "__main__":
    main()
