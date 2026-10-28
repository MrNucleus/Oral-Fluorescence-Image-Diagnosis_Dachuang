"""Train the patient-split ResNet18 binary classification baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.data import (  # noqa: E402
    BinaryImageDataset,
    build_binary_manifest,
    build_image_transforms,
)
from src.classification.metrics import evaluate_predictions  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_model(pretrained: bool, dropout: float) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
    return model


def set_trainable_stage(model: nn.Module, stage: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if stage >= 2:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def optimizer_for_trainable(model: nn.Module, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "n": total,
    }


@torch.no_grad()
def predict_loader(model, loader, device, use_amp):
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    patient_indices: list[int] = []
    for images, batch_labels, batch_patients in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        labels.extend(int(value) for value in batch_labels.tolist())
        probabilities.extend(float(value) for value in batch_probabilities.cpu().tolist())
        patient_indices.extend(int(value) for value in batch_patients.tolist())
    return labels, probabilities, patient_indices


def validation_metrics(model, loader, device, use_amp, threshold, bootstrap=0):
    labels, probabilities, patient_indices = predict_loader(
        model, loader, device, use_amp
    )
    return evaluate_predictions(
        labels,
        probabilities,
        patient_indices,
        loader.dataset.patient_ids,
        threshold=threshold,
        bootstrap=bootstrap,
    )


def selection_score(metrics: dict) -> float:
    patient_auc = metrics["patient_level"]["auroc"]
    if patient_auc is not None:
        return float(patient_auc)
    return float(metrics["patient_level"]["balanced_accuracy"])


def save_checkpoint(path: Path, model: nn.Module, epoch: int, stage: int, metrics: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "validation_metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary l/n classifier")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "binary.yaml"))
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    seed = int(config["seed"])
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    if args.stage1_epochs is not None:
        config["training"]["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None:
        config["training"]["stage2_epochs"] = args.stage2_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    manifest_path = PROJECT_ROOT / config["manifest"]
    summary_path = PROJECT_ROOT / config["summary"]
    if args.rebuild_manifest or not manifest_path.exists():
        summary = build_binary_manifest(
            manifest_path,
            summary_path,
            ratios=tuple(config["split"]["ratios"]),
            seed=seed,
            annotated_only=bool(config["data"].get("annotated_only", False)),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    experiment_dir = PROJECT_ROOT / config["experiment_dir"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "config_resolved.json", config)

    image_size = int(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    train_dataset = BinaryImageDataset(
        manifest_path, "train", build_image_transforms(image_size, train=True)
    )
    val_dataset = BinaryImageDataset(
        manifest_path, "val", build_image_transforms(image_size, train=False)
    )
    test_dataset = BinaryImageDataset(
        manifest_path, "test", build_image_transforms(image_size, train=False)
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    model = build_model(
        pretrained=bool(config["model"]["pretrained"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    train_labels = [int(record["label"]) for record in train_dataset.records]
    counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    threshold = float(config["evaluation"]["threshold"])

    run_info = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images": len(test_dataset),
        "train_patients": len(train_dataset.patient_ids),
        "val_patients": len(val_dataset.patient_ids),
        "test_patients": len(test_dataset.patient_ids),
        "train_class_counts": dict(Counter(train_labels)),
        "class_weights": class_weights.detach().cpu().tolist(),
    }
    save_json(experiment_dir / "run_info.json", run_info)
    print(json.dumps(run_info, ensure_ascii=False, indent=2))

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = -1
    best_stage = -1
    patience_counter = 0

    stage_specs = [
        (1, int(config["training"]["stage1_epochs"]), float(config["training"]["stage1_lr"])),
        (2, int(config["training"]["stage2_epochs"]), float(config["training"]["stage2_lr"])),
    ]
    for stage, epochs, learning_rate in stage_specs:
        if epochs <= 0:
            continue
        patience_counter = 0
        set_trainable_stage(model, stage)
        optimizer = optimizer_for_trainable(
            model,
            learning_rate=learning_rate,
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
        print(
            f"Stage {stage}: epochs={epochs}, lr={learning_rate}, "
            f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        for stage_epoch in range(1, epochs + 1):
            started = time.time()
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp
            )
            val_metrics = validation_metrics(
                model, val_loader, device, use_amp, threshold, bootstrap=0
            )
            scheduler.step()
            score = selection_score(val_metrics)
            epoch_record = {
                "stage": stage,
                "stage_epoch": stage_epoch,
                "epoch": len(history) + 1,
                **train_metrics,
                "val_loss_proxy": 1.0 - val_metrics["image_level"]["balanced_accuracy"],
                "val_image_balanced_accuracy": val_metrics["image_level"]["balanced_accuracy"],
                "val_patient_balanced_accuracy": val_metrics["patient_level"]["balanced_accuracy"],
                "val_patient_auroc": val_metrics["patient_level"]["auroc"],
                "seconds": round(time.time() - started, 2),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False))

            if score > best_score:
                best_score = score
                best_epoch = len(history)
                best_stage = stage
                patience_counter = 0
                save_checkpoint(
                    experiment_dir / "best.pt",
                    model,
                    best_epoch,
                    best_stage,
                    val_metrics,
                    config,
                )
            else:
                patience_counter += 1
            save_checkpoint(
                experiment_dir / "last.pt",
                model,
                len(history),
                stage,
                val_metrics,
                config,
            )
            if patience_counter >= int(config["training"]["patience"]):
                print(f"Early stopping at stage {stage}, epoch {stage_epoch}")
                break
        if patience_counter >= int(config["training"]["patience"]):
            break

    history_path = experiment_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        experiment_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = validation_metrics(
        model,
        test_loader,
        device,
        use_amp,
        threshold,
        bootstrap=int(config["evaluation"]["bootstrap"]),
    )
    final = {
        "status": "completed",
        "experiment": "binary_resnet18_v1",
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "selection_metric": "validation patient-level AUROC",
        "threshold": threshold,
        "test": test_metrics,
        "run_info": run_info,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(experiment_dir / "test_metrics.json", final)
    save_json(
        experiment_dir / "run_summary.json",
        {
            "status": final["status"],
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "training_epochs": len(history),
            "test_image_metrics": test_metrics["image_level"],
            "test_patient_metrics": test_metrics["patient_level"],
            "test_patient_bootstrap_ci95": test_metrics["patient_bootstrap_ci95"],
        },
    )
    print("Final test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

