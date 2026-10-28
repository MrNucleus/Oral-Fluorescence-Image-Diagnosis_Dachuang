"""Train binary l/n models using the model parameters in the project guide."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import sys
import time
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.data import BinaryImageDataset  # noqa: E402
from src.classification.metrics import (  # noqa: E402
    evaluate_predictions,
)
from src.utils.paths import PROJECT_ROOT  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class GaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def build_plan_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.20, contrast=0.15, saturation=0.10
                ),
                transforms.ToTensor(),
                GaussianNoise(std=0.01),
                normalize,
                transforms.RandomErasing(p=0.3),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class PlanClassifier(nn.Module):
    def __init__(self, backbone_name: str, dropout: float, pretrained: bool):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.stage2_modules = [backbone.layer3, backbone.layer4]
        elif backbone_name == "efficientnet_b3":
            weights = (
                models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            )
            backbone = models.efficientnet_b3(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.stage2_modules = [backbone.features[-2], backbone.features[-1]]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, tuple):
            features = features[0]
        return self.head(features)

    def set_stage(self, stage: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        if stage >= 2:
            for module in self.stage2_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if stage >= 3:
            for parameter in self.parameters():
                parameter.requires_grad = True


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def metric_pack(labels, probabilities, patient_indices, patient_ids) -> dict:
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    macro = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average="macro", zero_division=0
    )
    result = evaluate_predictions(
        labels,
        probabilities,
        patient_indices,
        patient_ids,
        threshold=0.5,
        bootstrap=0,
    )
    result["image_level"]["macro_f1"] = float(macro[2])
    result["image_level"]["kappa"] = float(cohen_kappa_score(labels, predictions))
    return result


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
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, use_amp) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    labels: list[int] = []
    probabilities: list[float] = []
    patient_indices: list[int] = []
    for images, batch_labels, batch_patients in loader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, batch_labels)
        total_loss += float(loss.item()) * batch_labels.size(0)
        total += batch_labels.size(0)
        batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        labels.extend(int(value) for value in batch_labels.tolist())
        probabilities.extend(float(value) for value in batch_probabilities.cpu().tolist())
        patient_indices.extend(int(value) for value in batch_patients.tolist())
    metrics = metric_pack(
        labels, probabilities, patient_indices, loader.dataset.patient_ids
    )
    return {
        "loss": total_loss / total,
        "metrics": metrics,
        "labels": labels,
        "probabilities": probabilities,
    }


def make_loader(manifest: Path, split: str, config: dict, shuffle: bool):
    image_size = int(config["data"]["image_size"])
    dataset = BinaryImageDataset(
        manifest,
        split,
        build_plan_transforms(image_size, train=shuffle),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


def selection_key(metrics: dict) -> tuple:
    image = metrics["metrics"]["image_level"]
    return (
        image["accuracy"],
        image["macro_f1"],
        image["auroc"] if image["auroc"] is not None else -1,
    )


def train_model(
    model_name: str,
    config: dict,
    train_loader,
    val_loader,
    device,
    experiment_dir: Path,
) -> dict:
    pretrained = next(
        item["pretrained"] for item in config["models"] if item["name"] == model_name
    )
    model = PlanClassifier(
        model_name,
        dropout=float(config["training"]["dropout"]),
        pretrained=bool(pretrained),
    ).to(device)
    train_labels = [int(row["label"]) for row in train_loader.dataset.records]
    counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config["training"]["label_smoothing"]),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(config["training"]["amp"]) and device.type == "cuda"
    )

    history: list[dict] = []
    best_key = (-1.0, -1.0, -1.0)
    best_metrics = None
    best_epoch = None
    best_stage = None
    global_epoch = 0
    dashboard_opened = False

    for stage_index, stage in enumerate(config["training"]["stages"], start=1):
        model.set_stage(stage_index)
        epochs = int(stage["epochs"])
        learning_rate = float(stage["learning_rate"])
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
        patience_counter = 0
        best_stage_loss = float("inf")
        print(
            f"[{model_name}] stage={stage['name']} epochs={epochs} "
            f"lr={learning_rate} trainable="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            started = time.time()
            train_loss, train_accuracy = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp=bool(config["training"]["amp"]) and device.type == "cuda",
            )
            validation = evaluate_loader(
                model,
                val_loader,
                criterion,
                device,
                use_amp=bool(config["training"]["amp"]) and device.type == "cuda",
            )
            scheduler.step()
            image_metrics = validation["metrics"]["image_level"]
            row = {
                "model": model_name,
                "stage": stage_index,
                "stage_name": stage["name"],
                "stage_epoch": stage_epoch,
                "epoch": global_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": validation["loss"],
                "val_accuracy": image_metrics["accuracy"],
                "val_macro_f1": image_metrics["macro_f1"],
                "val_auroc": image_metrics["auroc"],
                "val_kappa": image_metrics["kappa"],
                "seconds": round(time.time() - started, 2),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))

            key = selection_key(validation)
            if key > best_key:
                best_key = key
                best_metrics = validation["metrics"]
                best_epoch = global_epoch
                best_stage = stage_index
                checkpoint_dir = experiment_dir / model_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stage": stage_index,
                        "epoch": global_epoch,
                        "validation_metrics": best_metrics,
                        "model_name": model_name,
                    },
                    checkpoint_dir / "best.pt",
                )

            if validation["loss"] < best_stage_loss:
                best_stage_loss = validation["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= int(config["training"]["patience"]):
                print(
                    f"[{model_name}] early stop at stage {stage_index}, "
                    f"stage epoch {stage_epoch}"
                )
                break

            if bool(config.get("visualization", {}).get("live_dashboard", False)):
                write_live_outputs(
                    experiment_dir,
                    model_name,
                    history,
                    config,
                    best_stage or stage_index,
                    best_epoch or global_epoch,
                    best_metrics,
                    open_dashboard=(
                        bool(config.get("visualization", {}).get("open_dashboard", False))
                        and not dashboard_opened
                    ),
                )
                dashboard_opened = True

        if bool(config.get("visualization", {}).get("live_dashboard", False)):
            write_live_outputs(
                experiment_dir,
                model_name,
                history,
                config,
                best_stage or stage_index,
                best_epoch or global_epoch,
                best_metrics,
                open_dashboard=False,
            )

    model_dir = experiment_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(
        model_dir / "best_validation.json",
        {
            "model": model_name,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "metrics": best_metrics,
        },
    )
    return {
        "model_name": model_name,
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint": str(model_dir / "best.pt"),
        "history": history,
    }


def plot_history(model_result: dict, output_path: Path) -> None:
    history = model_result["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    fig.patch.set_facecolor("#f7f6f2")
    axes[0].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[1].set_title("Loss")
    axes[2].plot(epochs, [row["val_macro_f1"] for row in history], label="Macro F1")
    axes[2].plot(epochs, [row["val_auroc"] for row in history], label="AUROC")
    axes[2].set_title("Validation ranking metrics")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{model_result['model_name']} | best stage {model_result['best_stage']} "
        f"epoch {model_result['best_epoch']}",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_live_outputs(
    experiment_dir: Path,
    model_name: str,
    history: list[dict],
    config: dict,
    best_stage: int,
    best_epoch: int,
    best_metrics: dict | None,
    open_dashboard: bool,
) -> Path:
    model_dir = experiment_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "history_live.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    plot_history(
        {
            "model_name": model_name,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "history": history,
        },
        model_dir / "training_curves_live.png",
    )

    latest = history[-1]
    best_image = (
        best_metrics["image_level"]
        if best_metrics is not None
        else {"accuracy": latest["val_accuracy"], "macro_f1": latest["val_macro_f1"], "auroc": latest["val_auroc"]}
    )
    total_epochs = sum(int(stage["epochs"]) for stage in config["training"]["stages"])
    refresh = int(config.get("visualization", {}).get("refresh_seconds", 5))
    dashboard_path = model_dir / "live_dashboard.html"
    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(model_name)} live training</title>
  <style>
    body {{ margin: 0; background: #f2f0ea; color: #202124;
           font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: #5f6368; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #d9d6ce; border-radius: 8px;
             padding: 14px; }}
    .card b {{ display: block; font-size: 22px; color: #1f5a94; margin-top: 5px; }}
    .bar {{ height: 12px; background: #ddd9d0; border-radius: 8px; overflow: hidden;
            margin: 18px 0; }}
    .bar > i {{ display: block; height: 100%; background: #2d7f73;
                width: {100.0 * latest['epoch'] / total_epochs:.1f}%; }}
    img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #d9d6ce;
           background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin-top: 16px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e3e0d8; text-align: left; }}
    th {{ background: #e8eef2; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(model_name)} 实时训练</h1>
  <div class="sub">epoch {latest['epoch']} / {total_epochs} · stage {latest['stage']} · page refreshes every {refresh}s</div>
  <div class="bar"><i></i></div>
  <div class="grid">
    <div class="card">当前训练准确率<b>{latest['train_accuracy']:.3f}</b></div>
    <div class="card">当前验证准确率<b>{latest['val_accuracy']:.3f}</b></div>
    <div class="card">当前验证macro F1<b>{latest['val_macro_f1']:.3f}</b></div>
    <div class="card">当前验证AUROC<b>{latest['val_auroc'] if latest['val_auroc'] is not None else 0:.3f}</b></div>
    <div class="card">最佳验证准确率<b>{best_image['accuracy']:.3f}</b></div>
    <div class="card">最佳macro F1<b>{best_image['macro_f1']:.3f}</b></div>
    <div class="card">最佳AUROC<b>{(best_image['auroc'] or 0):.3f}</b></div>
    <div class="card">最佳epoch<b>{best_epoch}</b></div>
  </div>
  <p>最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <img src="training_curves_live.png?t={time.time_ns()}" alt="live training curves">
</main>
</body>
</html>
"""
    dashboard_path.write_text(dashboard, encoding="utf-8")
    if open_dashboard:
        webbrowser.open(dashboard_path.resolve().as_uri())
    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Guide-compliant binary training")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "binary_plan.yaml")
    )
    parser.add_argument("--max-stage-epochs", type=int, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_stage_epochs is not None:
        for stage in config["training"]["stages"]:
            stage["epochs"] = min(int(stage["epochs"]), args.max_stage_epochs)
    if args.models:
        requested = {name.strip() for name in args.models.split(",")}
        config["models"] = [
            item for item in config["models"] if item["name"] in requested
        ]
    if args.output_dir:
        config["output_dir"] = args.output_dir

    set_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = PROJECT_ROOT / config["manifest"]
    experiment_dir = PROJECT_ROOT / config["output_dir"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "config_resolved.json", config)
    train_loader = make_loader(manifest, "train", config, shuffle=True)
    val_loader = make_loader(manifest, "val", config, shuffle=False)
    test_loader = make_loader(manifest, "test", config, shuffle=False)

    model_results = []
    for model_cfg in config["models"]:
        model_results.append(
            train_model(
                model_cfg["name"],
                config,
                train_loader,
                val_loader,
                device,
                experiment_dir,
            )
        )
        plot_history(
            model_results[-1],
            experiment_dir / model_cfg["name"] / "training_curves.png",
        )

    selected = max(
        model_results,
        key=lambda result: (
            result["best_metrics"]["image_level"]["accuracy"],
            result["best_metrics"]["image_level"]["macro_f1"],
            result["best_metrics"]["image_level"]["auroc"]
            if result["best_metrics"]["image_level"]["auroc"] is not None
            else -1,
        ),
    )
    selection = {
        "rule": config["selection"],
        "candidates": [
            {
                "model": result["model_name"],
                "best_stage": result["best_stage"],
                "best_epoch": result["best_epoch"],
                "validation": result["best_metrics"]["image_level"],
            }
            for result in model_results
        ],
        "selected_model": selected["model_name"],
    }
    save_json(experiment_dir / "model_selection.json", selection)

    model = PlanClassifier(
        selected["model_name"],
        dropout=float(config["training"]["dropout"]),
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(
        selected["checkpoint"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    train_labels = [int(row["label"]) for row in train_loader.dataset.records]
    counts = Counter(train_labels)
    total = len(train_labels)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [total / (2 * counts[0]), total / (2 * counts[1])],
            dtype=torch.float32,
            device=device,
        ),
        label_smoothing=float(config["training"]["label_smoothing"]),
    )
    test_result = evaluate_loader(
        model,
        test_loader,
        criterion,
        device,
        use_amp=bool(config["training"]["amp"]) and device.type == "cuda",
    )
    final = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "selected_test_metrics": test_result["metrics"],
        "deviations_from_guide": [
            "Available labels are binary l/n rather than five clinical classes.",
            "The existing patient-level 70/15/15 split is reused instead of rebuilding 7:2:1.",
        ],
    }
    save_json(experiment_dir / "final_results.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

