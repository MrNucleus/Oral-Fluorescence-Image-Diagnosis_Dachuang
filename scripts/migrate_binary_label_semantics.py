"""Create v2 manifests with n=0/negative and l=1/positive."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = (
    (
        PROJECT_ROOT / "data" / "processed" / "binary_v1" / "manifest.csv",
        PROJECT_ROOT / "data" / "processed" / "binary_v2" / "manifest.csv",
        PROJECT_ROOT / "data" / "processed" / "binary_v2" / "split_summary.json",
    ),
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "binary_annotated_v1"
        / "manifest.csv",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "binary_annotated_v2"
        / "manifest.csv",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "binary_annotated_v2"
        / "split_summary.json",
    ),
)


def migrate(source: Path, destination: Path, summary_path: Path) -> dict:
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {source}")
    fields = list(rows[0])
    for row in rows:
        old_label = int(row["label"])
        if old_label not in (0, 1):
            raise ValueError(f"Unexpected label {old_label} in {source}")
        new_label = 1 - old_label
        row["label"] = str(new_label)
        row["label_name"] = "l" if new_label == 1 else "n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "label_map": {"n": 0, "l": 1},
        "label_semantics": "l=positive, n=negative",
        "confirmed_on": "2026-10-19",
        "source_manifest": source.relative_to(PROJECT_ROOT).as_posix(),
        "split": {},
        "assumptions": [
            "Only the label encoding was changed; images, patients and splits were preserved.",
            "n is the negative class and l is the positive class.",
        ],
    }
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        patient_labels = {
            row["patient_id"]: row["label"] for row in split_rows
        }
        summary["split"][split] = {
            "images": len(split_rows),
            "patients": len(patient_labels),
            "images_by_label": dict(Counter(row["label_name"] for row in split_rows)),
            "patients_by_label": dict(Counter(patient_labels.values())),
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    for source, destination, summary in MIGRATIONS:
        result = migrate(source, destination, summary)
        print(f"{source.relative_to(PROJECT_ROOT)} -> {destination.relative_to(PROJECT_ROOT)}")
        print(json.dumps(result["split"], ensure_ascii=False))


if __name__ == "__main__":
    main()

