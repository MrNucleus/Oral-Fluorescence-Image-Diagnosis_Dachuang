"""Manual deterministic check for epoch-boundary resume without pytest."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oral_fluorescence_detection import image_classification  # noqa: E402


def make_handoff(root: Path):
    records = []
    for split in ("train", "val", "test"):
        for index in range(4):
            label = index % 2
            modality = "standard" if index < 2 else "fluorescence"
            stem = f"{split}_{index}"
            relative = (
                f"{split}/{modality}/images/"
                f"{label}_{'normal' if label == 0 else 'lesion'}/{stem}.png"
            )
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((48, 80, 3), [30, 70, 110], dtype=np.uint8)
            pixels[8:36, 12:50] = [60 + 40 * label, 90, 140]
            Image.fromarray(pixels).save(path)
            records.append({
                "image_id": stem,
                "group_id": f"group_{split}_{index}",
                "split": split,
                "modality": modality,
                "label": label,
                "image_relpath": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "label_basis": "synthetic resume fixture",
            })
    dataset = root / "dataset.json"
    dataset.write_text(json.dumps({
        "schema_version": 1,
        "task": "image_classification",
        "class_names": ["normal_candidate", "lesion_candidate"],
        "records": records,
    }), encoding="utf-8")
    return dataset


def write_config(path: Path, epochs: int):
    path.write_text(yaml.safe_dump({
        "model": "resnet18",
        "pretrained": False,
        "epochs": epochs,
        "imgsz": 64,
        "batch": 2,
        "workers": 0,
        "device": "cpu",
        "seed": 31,
        "lr0": 0.001,
        "optimizer": "SGD",
        "patience": 5,
    }), encoding="utf-8")


def values_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="oral-resume-check-") as temporary:
        base = Path(temporary)
        dataset = make_handoff(base / "input")
        config = base / "train.yaml"

        image_classification.EXPERIMENTS_DIR = base / "runs_full"
        write_config(config, 2)
        full_best = image_classification.train(dataset, config_yaml=config, name="full")
        full_run = full_best.parent

        image_classification.EXPERIMENTS_DIR = base / "runs_resume"
        write_config(config, 1)
        first_best = image_classification.train(dataset, config_yaml=config, name="resume")
        first_run = first_best.parent
        write_config(config, 2)
        image_classification.train(
            dataset,
            config_yaml=config,
            resume=first_run / "last.pt",
        )

        full_history = json.loads((full_run / "history.json").read_text(encoding="utf-8"))
        resumed_history = json.loads((first_run / "history.json").read_text(encoding="utf-8"))
        full_state = torch.load(full_run / "last.pt", map_location="cpu", weights_only=False)
        resumed_state = torch.load(first_run / "last.pt", map_location="cpu", weights_only=False)

        result = {
            "history_equal": full_history == resumed_history,
            "model_equal": values_equal(
                full_state["model_state_dict"], resumed_state["model_state_dict"]
            ),
            "optimizer_equal": values_equal(
                full_state["optimizer_state_dict"], resumed_state["optimizer_state_dict"],
            ),
            "completed_epochs": resumed_state["epoch"],
            "continue_script_exists": (first_run / "continue.ps1").is_file(),
        }
        print(json.dumps(result, ensure_ascii=False))
        if not all((
            result["history_equal"],
            result["model_equal"],
            result["optimizer_equal"],
            result["completed_epochs"] == 2,
            result["continue_script_exists"],
        )):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
