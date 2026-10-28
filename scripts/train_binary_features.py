"""Classical color/texture baseline for the binary l/n classification task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_binary import load_config, save_json  # noqa: E402
from src.classification.data import (  # noqa: E402
    build_binary_manifest,
)
from src.classification.metrics import (  # noqa: E402
    evaluate_predictions,
    select_operating_threshold,
)
from src.utils.paths import PROJECT_ROOT  # noqa: E402


def extract_features(relative_path: str) -> np.ndarray:
    path = PROJECT_ROOT / relative_path
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").resize((128, 128))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    values: list[float] = []
    for channel in (rgb[..., 0], rgb[..., 1], rgb[..., 2], hsv[..., 0], hsv[..., 1], hsv[..., 2], gray):
        values.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(np.percentile(channel, 10)),
                float(np.percentile(channel, 50)),
                float(np.percentile(channel, 90)),
            ]
        )

    eps = 1e-6
    values.extend(
        [
            float((rgb[..., 0].mean() + eps) / (rgb[..., 1].mean() + eps)),
            float((rgb[..., 0].mean() + eps) / (rgb[..., 2].mean() + eps)),
            float((rgb[..., 1].mean() + eps) / (rgb[..., 2].mean() + eps)),
            float(np.abs(np.diff(gray, axis=0)).mean()),
            float(np.abs(np.diff(gray, axis=1)).mean()),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def read_split(manifest_path: Path, split: str) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def split_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    paths = [row["image"] for row in rows]
    with ThreadPoolExecutor(max_workers=8) as executor:
        features = list(executor.map(extract_features, paths))
    patient_ids = sorted({row["patient_id"] for row in rows})
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(patient_ids)
    }
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    patient_indices = [patient_to_index[row["patient_id"]] for row in rows]
    return np.vstack(features), labels, patient_indices, patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train color/texture binary baseline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "binary_annotated.yaml"))
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "experiments" / "binary_logreg_color_v1"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest_path = PROJECT_ROOT / config["manifest"]
    if args.rebuild_manifest or not manifest_path.exists():
        build_binary_manifest(
            manifest_path,
            PROJECT_ROOT / config["summary"],
            ratios=tuple(config["split"]["ratios"]),
            seed=int(config["seed"]),
            annotated_only=bool(config["data"].get("annotated_only", False)),
        )

    train_rows = read_split(manifest_path, "train")
    val_rows = read_split(manifest_path, "val")
    test_rows = read_split(manifest_path, "test")
    x_train, y_train, _, _ = split_arrays(train_rows)
    x_val, y_val, val_patient_indices, val_patient_ids = split_arrays(val_rows)
    x_test, y_test, test_patient_indices, test_patient_ids = split_arrays(test_rows)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=int(config["seed"]),
        ),
    )
    model.fit(x_train, y_train)
    val_probabilities = model.predict_proba(x_val)[:, 1].tolist()
    test_probabilities = model.predict_proba(x_test)[:, 1].tolist()

    val_default = evaluate_predictions(
        y_val.tolist(),
        val_probabilities,
        val_patient_indices,
        val_patient_ids,
        threshold=0.5,
        bootstrap=0,
    )
    threshold_result = select_operating_threshold(
        [row["label"] for row in val_default["patients"]],
        [row["mean_probability"] for row in val_default["patients"]],
    )
    selected_threshold = threshold_result["selected"]["threshold"]
    test_default = evaluate_predictions(
        y_test.tolist(),
        test_probabilities,
        test_patient_indices,
        test_patient_ids,
        threshold=0.5,
        bootstrap=1000,
    )
    test_selected = evaluate_predictions(
        y_test.tolist(),
        test_probabilities,
        test_patient_indices,
        test_patient_ids,
        threshold=selected_threshold,
        bootstrap=1000,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "StandardScaler + LogisticRegression",
        "features": "RGB/HSV percentiles, RGB ratios, gradient energy",
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "n_features": int(x_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "threshold_selection": threshold_result,
        "selection_split": "val",
        "test_at_default_threshold": test_default,
        "test_at_selected_threshold": test_selected,
    }
    save_json(output / "metrics.json", payload)
    print(
        json.dumps(
            {
                "n_features": payload["n_features"],
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

