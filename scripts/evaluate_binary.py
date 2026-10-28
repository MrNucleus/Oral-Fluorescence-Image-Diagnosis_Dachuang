"""Evaluate a saved binary checkpoint and select its threshold on validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_binary import (  # noqa: E402
    build_model,
    load_config,
    predict_loader,
    save_json,
)
from src.classification.data import (  # noqa: E402
    BinaryImageDataset,
    build_image_transforms,
)
from src.classification.metrics import (  # noqa: E402
    evaluate_predictions,
    select_operating_threshold,
)
from src.utils.paths import PROJECT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved binary checkpoint")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "binary.yaml"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest_path = PROJECT_ROOT / config["manifest"]
    experiment_dir = PROJECT_ROOT / config["experiment_dir"]
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else experiment_dir / "best.pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])

    val_dataset = BinaryImageDataset(
        manifest_path, "val", build_image_transforms(image_size, train=False)
    )
    test_dataset = BinaryImageDataset(
        manifest_path, "test", build_image_transforms(image_size, train=False)
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    model = build_model(pretrained=False, dropout=float(config["model"]["dropout"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_labels, val_probabilities, val_patient_indices = predict_loader(
        model, val_loader, device, use_amp
    )
    val_result = evaluate_predictions(
        val_labels,
        val_probabilities,
        val_patient_indices,
        val_dataset.patient_ids,
        threshold=0.5,
        bootstrap=0,
    )
    patients = val_result["patients"]
    threshold_result = select_operating_threshold(
        [row["label"] for row in patients],
        [row["mean_probability"] for row in patients],
    )
    selected_threshold = threshold_result["selected"]["threshold"]

    test_labels, test_probabilities, test_patient_indices = predict_loader(
        model, test_loader, device, use_amp
    )
    test_default = evaluate_predictions(
        test_labels,
        test_probabilities,
        test_patient_indices,
        test_dataset.patient_ids,
        threshold=0.5,
        bootstrap=int(config["evaluation"]["bootstrap"]),
    )
    test_selected = evaluate_predictions(
        test_labels,
        test_probabilities,
        test_patient_indices,
        test_dataset.patient_ids,
        threshold=selected_threshold,
        bootstrap=int(config["evaluation"]["bootstrap"]),
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "selection_split": "val",
        "selection_scope": "patient-level mean predicted probability",
        "threshold_selection": threshold_result,
        "test_at_default_threshold": test_default,
        "test_at_selected_threshold": test_selected,
    }
    save_json(experiment_dir / "threshold_evaluation.json", payload)
    print(
        json.dumps(
            {
                "selected_threshold": selected_threshold,
                "validation_selected": threshold_result["selected"],
                "test_default_image": test_default["image_level"],
                "test_default_patient": test_default["patient_level"],
                "test_selected_image": test_selected["image_level"],
                "test_selected_patient": test_selected["patient_level"],
                "test_selected_patient_ci95": test_selected["patient_bootstrap_ci95"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

