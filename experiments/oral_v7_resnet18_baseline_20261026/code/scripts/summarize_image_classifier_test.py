"""Validate frozen test predictions and summarize them by site and modality."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oral_fluorescence_detection.image_classification import binary_metrics  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_unique(records: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        value = record.get(key)
        if not value:
            raise ValueError(f"{label} contains an empty {key}")
        if value in indexed:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        indexed[value] = record
    return indexed


def metric_row(dimension: str, group: str, records: list[dict]) -> dict:
    labels = [int(record["label"]) for record in records]
    probabilities = [float(record["lesion_probability"]) for record in records]
    metrics = binary_metrics(labels, probabilities)
    return {
        "dimension": dimension,
        "group": group,
        "images": metrics["images"],
        "normal_candidate_images": sum(label == 0 for label in labels),
        "lesion_candidate_images": sum(label == 1 for label in labels),
        "source_groups": len({record["group_id"] for record in records}),
        **{key: value for key, value in metrics.items() if key != "images"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-sites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    test_metrics_path = args.test_metrics.resolve()
    dataset_path = args.dataset.resolve()
    sites_path = args.image_sites.resolve()
    output = args.output.resolve()

    dataset = load_json(dataset_path)
    sites = load_json(sites_path)
    predictions = load_json(predictions_path)
    test_result = load_json(test_metrics_path)

    dataset_hash = sha256(dataset_path)
    sites_hash = sha256(sites_path)
    expected_dataset_hash = test_result.get("dataset_json_sha256")
    if dataset_hash != expected_dataset_hash:
        raise ValueError("Dataset hash differs from the frozen test result")
    if sites.get("dataset_json_sha256") != dataset_hash:
        raise ValueError("Site table belongs to a different dataset manifest")

    dataset_records = index_unique(dataset["records"], "image_id", "dataset")
    site_records = index_unique(sites["records"], "image_id", "image_sites")
    prediction_records = index_unique(predictions, "image_id", "predictions")
    expected_ids = {
        image_id
        for image_id, record in dataset_records.items()
        if record.get("split") == "test"
    }
    if set(prediction_records) != expected_ids:
        missing = sorted(expected_ids - set(prediction_records))
        extra = sorted(set(prediction_records) - expected_ids)
        raise ValueError(f"Prediction IDs do not match frozen test IDs; missing={missing}, extra={extra}")

    joined = []
    for image_id in sorted(expected_ids):
        source = dataset_records[image_id]
        site = site_records.get(image_id)
        prediction = prediction_records[image_id]
        if site is None:
            raise ValueError(f"Missing site record for test image: {image_id}")
        if site.get("split") != "test" or int(site.get("label")) != int(source["label"]):
            raise ValueError(f"Site record split or label mismatch: {image_id}")
        if int(prediction["label"]) != int(source["label"]):
            raise ValueError(f"Prediction label mismatch: {image_id}")
        probability = float(prediction["lesion_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid lesion probability: {image_id}")
        joined.append({
            "image_id": image_id,
            "group_id": source["group_id"],
            "label": int(source["label"]),
            "lesion_probability": probability,
            "modality": source["modality"],
            "site_group": site["site_group_main"],
            "site_group_cn": site.get("site_group_main_cn", ""),
            "site_multi_flag": bool(site.get("site_multi_flag", False)),
        })

    overall = metric_row("overall", "all_test", joined)
    frozen_metrics = test_result["metrics"]
    if overall["confusion_matrix"] != frozen_metrics["confusion_matrix"]:
        raise ValueError("Recomputed overall confusion matrix differs from frozen test result")
    for key in ("accuracy", "sensitivity", "specificity", "balanced_accuracy", "auroc"):
        if abs(float(overall[key]) - float(frozen_metrics[key])) > 1e-12:
            raise ValueError(f"Recomputed overall {key} differs from frozen test result")

    rows = [overall]
    for dimension, key in (("site", "site_group"), ("modality", "modality")):
        grouped = defaultdict(list)
        for record in joined:
            grouped[record[key]].append(record)
        for group in sorted(grouped):
            row = metric_row(dimension, group, grouped[group])
            if dimension == "site":
                row["group_cn"] = next(
                    record["site_group_cn"] for record in grouped[group]
                )
                row["multisite_images"] = sum(
                    record["site_multi_flag"] for record in grouped[group]
                )
            rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    document = {
        "task": "whole_image_test_stratification",
        "threshold": 0.5,
        "positive_class": "1 = lesion_candidate",
        "test_images": len(joined),
        "test_source_groups": len({record["group_id"] for record in joined}),
        "dataset_json_sha256": dataset_hash,
        "image_sites_json_sha256": sites_hash,
        "test_predictions_sha256": sha256(predictions_path),
        "test_metrics_sha256": sha256(test_metrics_path),
        "validation": {
            "prediction_ids_match_test_ids": True,
            "unique_prediction_ids": len(prediction_records),
            "unique_image_site_ids": len(site_records),
            "overall_metrics_match_frozen_result": True,
        },
        "limitations": [
            "Labels are candidate labels, not confirmed clinical diagnoses.",
            "Source-folder groups are not confirmed independent patients.",
            "Confidence intervals were not pre-specified.",
            "Subgroup estimates are descriptive and should not be used for ranking claims.",
        ],
        "rows": rows,
    }
    (output / "stratified_metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "dimension", "group", "group_cn", "images", "normal_candidate_images",
        "lesion_candidate_images", "source_groups", "auroc", "sensitivity",
        "specificity", "balanced_accuracy", "accuracy", "confusion_matrix",
        "multisite_images",
    ]
    with (output / "stratified_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["confusion_matrix"] = json.dumps(
                row["confusion_matrix"], separators=(",", ":")
            )
            writer.writerow(rendered)

    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
