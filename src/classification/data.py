"""Patient-safe binary manifest construction and image datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..data.build_dataset import discover_samples
from ..data.voc import parse_voc
from ..utils.paths import PROJECT_ROOT


LABEL_TO_INDEX = {"n": 0, "l": 1}
INDEX_TO_LABEL = {0: "n", 1: "l"}
MANIFEST_FIELDS = (
    "split",
    "patient_id",
    "image",
    "label",
    "label_name",
    "has_annotation",
)


def anonymous_patient_id(patient_key: str) -> str:
    """Return a stable pseudonym without persisting the source identifier."""
    digest = hashlib.sha256(f"oral-binary-v1::{patient_key}".encode("utf-8")).hexdigest()
    return f"P{digest[:10].upper()}"


def _relative_image_path(image_path: str) -> str:
    path = Path(image_path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image is outside the project root: {path}") from exc


def _patient_labels(samples: Iterable[dict]) -> dict[str, int]:
    patient_classes: dict[str, set[str]] = defaultdict(set)
    patient_ids: dict[str, str] = {}

    for sample in samples:
        source_id = sample["patient"]
        patient_ids[source_id] = anonymous_patient_id(source_id)
        if not sample["xml"]:
            continue
        annotation = parse_voc(sample["xml"])
        for box in annotation.boxes:
            if box.cls_name in LABEL_TO_INDEX:
                patient_classes[source_id].add(box.cls_name)

    missing = sorted(patient_ids[p] for p in patient_ids if not patient_classes[p])
    ambiguous = sorted(
        patient_ids[p] for p in patient_ids if len(patient_classes[p]) > 1
    )
    if missing:
        raise ValueError(
            "Patients without a usable binary label: " + ", ".join(missing)
        )
    if ambiguous:
        raise ValueError(
            "Patients with conflicting binary labels: " + ", ".join(ambiguous)
        )

    return {
        source_id: LABEL_TO_INDEX[next(iter(classes))]
        for source_id, classes in patient_classes.items()
    }


def _has_usable_annotation(sample: dict) -> bool:
    if not sample["xml"]:
        return False
    annotation = parse_voc(sample["xml"])
    return any(box.cls_name in LABEL_TO_INDEX for box in annotation.boxes)


def _stratified_patient_split(
    patient_labels: dict[str, int],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, set[str]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("Split ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")

    by_label: dict[int, list[str]] = defaultdict(list)
    for patient_key, label in patient_labels.items():
        by_label[label].append(anonymous_patient_id(patient_key))

    result = {"train": set(), "val": set(), "test": set()}
    for label, patients in sorted(by_label.items()):
        if len(patients) < 3:
            raise ValueError(f"Not enough patients for label {label}")
        patients = sorted(patients)
        random.Random(seed + 1009 * (label + 1)).shuffle(patients)
        n_total = len(patients)
        n_val = max(1, int(round(n_total * ratios[1])))
        n_test = max(1, int(round(n_total * ratios[2])))
        n_train = n_total - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Not enough training patients for label {label}")
        result["train"].update(patients[:n_train])
        result["val"].update(patients[n_train:n_train + n_val])
        result["test"].update(patients[n_train + n_val:])

    overlaps = {
        "train_val": result["train"] & result["val"],
        "train_test": result["train"] & result["test"],
        "val_test": result["val"] & result["test"],
    }
    if any(overlaps.values()):
        raise RuntimeError("Patient leakage detected while constructing splits")
    return result


def _split_summary(
    records: list[dict],
    seed: int,
    ratios: tuple[float, float, float],
    annotated_only: bool,
) -> dict:
    summary: dict = {
        "seed": seed,
        "ratios": list(ratios),
        "label_map": LABEL_TO_INDEX,
        "label_source": "VOC object class per patient; image inherits patient label",
        "label_semantics": "n=negative (0), l=positive (1)",
        "annotated_only": annotated_only,
        "split": {},
        "assumptions": [
            "Each patient has exactly one valid binary class across all annotated images.",
            "Included images inherit the patient label; unannotated images are excluded when annotated_only is true.",
            "Patient identifiers in the manifest are one-way pseudonyms.",
            "The clinical meaning of l and n is not asserted.",
        ],
    }
    for split in ("train", "val", "test"):
        rows = [record for record in records if record["split"] == split]
        patient_labels = {
            record["patient_id"]: record["label"] for record in rows
        }
        summary["split"][split] = {
            "images": len(rows),
            "patients": len(patient_labels),
            "images_by_label": dict(Counter(record["label_name"] for record in rows)),
            "patients_by_label": dict(Counter(patient_labels.values())),
        }
    return summary


def build_binary_manifest(
    output_path: Path | str,
    summary_path: Path | str | None = None,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    annotated_only: bool = False,
) -> dict:
    """Build a patient-level binary manifest without writing source identifiers."""
    samples = discover_samples()
    source_labels = _patient_labels(samples)
    if annotated_only:
        samples = [sample for sample in samples if _has_usable_annotation(sample)]
    split_patients = _stratified_patient_split(source_labels, ratios, seed)

    patient_to_split = {
        patient: split
        for split, patients in split_patients.items()
        for patient in patients
    }
    records: list[dict] = []
    for sample in samples:
        source_id = sample["patient"]
        patient_id = anonymous_patient_id(source_id)
        label = source_labels[source_id]
        records.append(
            {
                "split": patient_to_split[patient_id],
                "patient_id": patient_id,
                "image": _relative_image_path(sample["image"]),
                "label": label,
                "label_name": INDEX_TO_LABEL[label],
                "has_annotation": int(bool(sample["xml"])),
            }
        )

    records.sort(key=lambda row: (row["split"], row["patient_id"], row["image"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    summary = _split_summary(records, seed, ratios, annotated_only)
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def build_image_transforms(image_size: int = 224, train: bool = False):
    """Use whole-image resizing so peripheral lesions are not cropped away."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.08, contrast=0.08, saturation=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class BinaryImageDataset(Dataset):
    """Image dataset backed by an anonymized manifest."""

    def __init__(self, manifest_path: Path | str, split: str, transform=None):
        self.transform = transform
        rows: list[dict] = []
        with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    rows.append(row)
        if not rows:
            raise ValueError(f"No rows found for split={split!r}")

        self.records = rows
        self.patient_ids = sorted({row["patient_id"] for row in rows})
        self.patient_to_index = {
            patient_id: index for index, patient_id in enumerate(self.patient_ids)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = PROJECT_ROOT / record["image"]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(record["label"]), dtype=torch.long)
        patient_index = torch.tensor(
            self.patient_to_index[record["patient_id"]], dtype=torch.long
        )
        return image, label, patient_index

