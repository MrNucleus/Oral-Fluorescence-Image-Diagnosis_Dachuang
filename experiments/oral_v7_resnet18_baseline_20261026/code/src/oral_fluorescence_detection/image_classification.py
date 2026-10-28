"""Whole-image classification training; validation selects models, test stays separate."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .online_augmentation import (
    CLASS_NAMES, MODALITIES, AugmentationPolicy, ImageClassificationDataset, validate_handoff,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs/image_classifier_runs"


def build_model(model_name, pretrained=True):
    """Initialize a classifier; pretrained tensors never imply cropping images."""
    from torch import nn
    from torchvision import models
    choices = {"resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
               "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
               "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT)}
    if model_name not in choices:
        raise ValueError("Use resnet18, resnet34 or resnet50 for whole-image classification")
    constructor, weights = choices[model_name]
    model = constructor(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def resolve_device(value):
    import torch
    text = str(value)
    if text == "cpu":
        return torch.device("cpu")
    if text == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(text)
    raise ValueError(f"Device {value!r} unavailable; choose cpu, mps or an available CUDA index")


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _new_run(root, name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Experiment name must be a single directory name")
    root.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        path = root / (name if index == 0 else f"{name}_{index+1}")
        try:
            path.mkdir(mode=0o700)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate new experiment directory")


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o600)


def _atomic_torch_save(value, path):
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    path.chmod(0o600)


def _capture_rng_state(torch, loader_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch, loader_generator, state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_continue_script(run):
    script = r"""$ErrorActionPreference = 'Stop'
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = (Resolve-Path (Join-Path $runDir '..\..\..')).Path
$workspace = (Resolve-Path (Join-Path $codeDir '..\..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$trainer = Join-Path $codeDir 'scripts\train_image_classifier.py'
$data = Join-Path $workspace 'data\raw\oral_binary_2026-10-23\口腔整图分类_完整交接包_2026-10-21\01_图片\dataset.json'
$resume = Join-Path $runDir 'last.pt'
if (-not (Test-Path -LiteralPath $resume)) {
    $resume = Join-Path $runDir 'best.pt'
}
$log = Join-Path $runDir 'training.continue.log'
& $python -B $trainer --data $data --resume $resume --epochs 100 2>&1 |
    Tee-Object -FilePath $log -Append
exit $LASTEXITCODE
"""
    path = run / "continue.ps1"
    if not path.exists():
        path.write_text(script, encoding="utf-8")
        path.chmod(0o600)


def _history_derived_state(history):
    if not history:
        return float("inf"), 0
    losses = [float(item["val_metrics"]["loss"]) for item in history]
    best_loss = min(losses)
    best_index = losses.index(best_loss)
    return best_loss, len(losses) - best_index - 1


def binary_metrics(labels, probabilities):
    """Fixed 0.5 decision threshold; missing classes produce null metrics, not zero."""
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    predicted = (score >= .5).astype(np.int64)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(y, predicted):
        matrix[truth, pred] += 1
    tn, fp, fn, tp = map(int, matrix.ravel())
    sensitivity = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    auc = None
    n1, n0 = int(y.sum()), len(y)-int(y.sum())
    if n0 and n1:
        # Mann-Whitney ranks with tie averaging; avoids quadratic pair matrices.
        order = np.argsort(score, kind="stable")
        sorted_score = score[order]
        ranks = np.empty(len(y), dtype=float)
        i = 0
        while i < len(y):
            j = i+1
            while j < len(y) and sorted_score[j] == sorted_score[i]:
                j += 1
            ranks[order[i:j]] = (i+1+j)/2
            i = j
        auc = float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))
    return {"images": len(y), "accuracy": (tn+tp)/len(y), "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity+specificity)/2
            if sensitivity is not None and specificity is not None else None,
            "auroc": auc, "confusion_matrix": matrix.tolist(), "threshold": .5}


def evaluate_model(model, loader, device):
    import torch
    from torch import nn
    model.eval()
    loss_sum, labels, scores, ids = 0., [], [], []
    with torch.inference_mode():
        for batch in loader:
            images, target = batch["image"].to(device), batch["label"].to(device)
            logits = model(images)
            loss_sum += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
            labels.extend(target.cpu().tolist())
            scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
            ids.extend(batch["image_id"])
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    metrics = {**binary_metrics(labels, scores), "loss": loss_sum/len(labels)}
    if not np.isfinite(metrics["loss"]) or not np.isfinite(scores).all():
        raise ValueError("Nonfinite classifier output")
    return metrics, [{"image_id": i, "label": y, "lesion_probability": p}
                     for i, y, p in zip(ids, labels, scores)]


def train(data_json, config_yaml=None, name=None, overrides=None,
          augmentation_yaml=None, profile=None, check_only=False, modalities=None,
          resume=None):
    config_yaml = Path(config_yaml or CONFIGS_DIR / "train_image_classifier.yaml")
    cfg = yaml.safe_load((CONFIGS_DIR / "train_image_classifier.yaml").read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load(config_yaml.read_text(encoding="utf-8")))
    cfg.update(overrides or {})
    allowed = {"model", "epochs", "imgsz", "batch", "lr0", "optimizer", "patience",
               "device", "workers", "seed", "pretrained", "name", "augmentation_config",
               "weight_decay"}
    if set(cfg)-allowed:
        raise ValueError(f"Unknown classification configuration: {sorted(set(cfg)-allowed)}")
    configured_policy = cfg.pop("augmentation_config", "configs/online_augmentation.yaml")
    policy_path = Path(augmentation_yaml or configured_policy)
    if not policy_path.is_absolute():
        policy_path = PROJECT_ROOT / policy_path
    policy = AugmentationPolicy.from_yaml(policy_path, profile=profile)
    modes = sorted(set(modalities or MODALITIES))
    if not set(modes) <= MODALITIES or not modes:
        raise ValueError("Invalid modality selection")
    summary = validate_handoff(data_json)
    # No model constructor/import/download or test loading in this preflight.
    for split in ("train", "val"):
        rows = ImageClassificationDataset(data_json, split, policy, cfg["imgsz"], modes).records
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"Both candidate classes required in selected {split} samples")
    if cfg["model"] not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError("This whole-image trainer requires a classification ResNet model")
    if (cfg["epochs"] < 1 or cfg["batch"] < 2 or cfg["workers"] < 0
            or cfg["imgsz"] < 64 or cfg["lr0"] <= 0 or cfg["patience"] < 1):
        raise ValueError("Invalid whole-image training hyperparameters")
    if check_only:
        result = {"dataset": summary, "augmentation": asdict(policy),
                  "modalities": modes, "task": "image_classification", "training_executed": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(cfg["device"])
    datasets = {s: ImageClassificationDataset(data_json, s, policy, cfg["imgsz"], modes)
                for s in ("train", "val")}
    loader_generators = {s: torch.Generator().manual_seed(seed) for s in ("train", "val")}
    loaders = {s: DataLoader(ds, batch_size=cfg["batch"], shuffle=s == "train",
                            num_workers=cfg["workers"], worker_init_fn=seed_worker,
                            generator=loader_generators[s], drop_last=False)
               for s, ds in datasets.items()}
    resume_path = Path(resume).resolve() if resume else None
    model = build_model(cfg["model"], bool(cfg["pretrained"]) and resume_path is None).to(device)
    if cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr0"],
                                      weight_decay=cfg.get("weight_decay", .0001))
    elif cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr0"], momentum=.9,
                                    weight_decay=cfg.get("weight_decay", .0001))
    else:
        raise ValueError("optimizer must be AdamW or SGD")
    if resume_path is None:
        run = _new_run(EXPERIMENTS_DIR, name or cfg["name"])
    else:
        if not resume_path.is_file():
            raise ValueError(f"Resume checkpoint does not exist: {resume_path}")
        run = resume_path.parent
    metadata = {"task": "image_classification", "config": cfg, "augmentation": asdict(policy),
                "dataset": summary, "modalities": modes, "python": platform.python_version(),
                "torch": torch.__version__, "label_unit": "whole_image",
                "selection": "minimum validation cross-entropy; no test evaluation",
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(__file__).with_name(
                                      "online_augmentation.py"))}}
    if resume_path is None:
        _write_json(run / "run_config.json", metadata)
    _write_continue_script(run)

    best_loss, stale, history = float("inf"), 0, []
    start_epoch = 0
    optimizer_state_preserved = False
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("task") != "image_classification":
            raise ValueError("Resume checkpoint is not an image classification checkpoint")
        if state.get("class_names") != CLASS_NAMES or state.get("model_name") != cfg["model"]:
            raise ValueError("Resume checkpoint model or class mapping differs from configuration")
        if state.get("dataset_json_sha256") != summary["dataset_json_sha256"]:
            raise ValueError("Resume checkpoint belongs to a different dataset manifest")
        if int(state.get("imgsz", -1)) != int(cfg["imgsz"]):
            raise ValueError("Resume checkpoint image size differs from configuration")
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
            optimizer_state_preserved = True
        history = state.get("history")
        if not isinstance(history, list):
            history_file = run / "history.json"
            history = (
                json.loads(history_file.read_text(encoding="utf-8"))
                if history_file.is_file()
                else []
            )
        derived_loss, derived_stale = _history_derived_state(history)
        best_loss = float(state.get("best_loss", derived_loss))
        stale = int(state.get("stale", derived_stale))
        start_epoch = int(state.get("epoch", len(history)))
        if start_epoch > int(cfg["epochs"]):
            raise ValueError("Resume checkpoint is beyond the configured epoch budget")
        _restore_rng_state(torch, loader_generators["train"], state.get("rng_state"))
        _write_json(run / "resume_info.json", {
            "resumed_at": datetime.now().astimezone().isoformat(),
            "checkpoint": str(resume_path),
            "completed_epoch": start_epoch,
            "optimizer_state_preserved": optimizer_state_preserved,
            "history_entries": len(history),
        })

    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        train_loss, samples = 0., 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = nn.functional.cross_entropy(logits, batch["label"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite classification loss")
            loss.backward()
            optimizer.step()
            samples += len(batch["label"])
            train_loss += float(loss.detach())*len(batch["label"])
        metrics, _ = evaluate_model(model, loaders["val"], device)
        history.append({"epoch": epoch+1, "train_images": samples,
                        "train_loss": train_loss/samples, "val_metrics": metrics})
        print(json.dumps(history[-1], ensure_ascii=False))
        if metrics["loss"] < best_loss:
            best_loss, stale = metrics["loss"], 0
            checkpoint = {"task": "image_classification", "model_name": cfg["model"],
                          "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "class_names": CLASS_NAMES, "imgsz": cfg["imgsz"], "epoch": epoch+1,
                          "val_metrics": metrics, "augmentation": asdict(policy), "seed": seed,
                          "modalities": modes, "dataset_json_sha256": summary["dataset_json_sha256"]}
            _atomic_torch_save(checkpoint, run / "best.pt")
        else:
            stale += 1
        _write_json(run / "history.json", history)
        last_state = {
            "task": "image_classification",
            "model_name": cfg["model"],
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "imgsz": cfg["imgsz"],
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "stale": stale,
            "history": history,
            "augmentation": asdict(policy),
            "seed": seed,
            "modalities": modes,
            "dataset_json_sha256": summary["dataset_json_sha256"],
            "rng_state": _capture_rng_state(torch, loader_generators["train"]),
            "complete_epoch_boundary": True,
        }
        _atomic_torch_save(last_state, run / "last.pt")
        best_entry = min(history, key=lambda item: float(item["val_metrics"]["loss"]))
        complete = epoch + 1 >= int(cfg["epochs"]) or stale >= cfg["patience"]
        _write_json(run / "continue_state.json", {
            "status": "complete" if complete else "running",
            "last_completed_epoch": epoch + 1,
            "next_epoch": epoch + 2,
            "max_epochs": int(cfg["epochs"]),
            "best_epoch": best_entry["epoch"],
            "best_val_loss": best_entry["val_metrics"]["loss"],
            "best_val_auroc": best_entry["val_metrics"]["auroc"],
            "best_balanced_accuracy": best_entry["val_metrics"]["balanced_accuracy"],
            "resume_checkpoint": str(run / "last.pt"),
            "checkpoint_scope": "completed_epoch_boundary",
            "optimizer_state_preserved": True,
            "history": str(run / "history.json"),
            "continue_script": str(run / "continue.ps1"),
            "dataset_json_sha256": summary["dataset_json_sha256"],
        })
        if stale >= cfg["patience"]:
            break
    if start_epoch >= int(cfg["epochs"]):
        print(f"Resume checkpoint is already at the configured epoch limit: {run / 'best.pt'}")
        return run / "best.pt"
    print(f"整图分类训练完成：{run / 'best.pt'}；test 未参与训练或模型选择")
    return run / "best.pt"


def evaluate_checkpoint(checkpoint_path, data_json, output, device="cpu", batch=16, workers=0):
    """Explicit final test evaluation. Frozen checkpoint supplies preprocessing/modality."""
    import torch
    from torch.utils.data import DataLoader
    summary = validate_handoff(data_json)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("task") != "image_classification"
            or checkpoint.get("class_names") != CLASS_NAMES):
        raise ValueError("Whole-image classification checkpoint required")
    if checkpoint.get("dataset_json_sha256") != summary["dataset_json_sha256"]:
        raise ValueError("Dataset differs from trained frozen manifest")
    destination = Path(output).resolve()
    if destination.is_relative_to(Path(data_json).resolve().parent):
        raise ValueError("Write evaluation outside the frozen input package")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    target_device = resolve_device(device)
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    dataset = ImageClassificationDataset(data_json, "test", imgsz=checkpoint["imgsz"],
                                         modalities=checkpoint["modalities"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
    metrics, predictions = evaluate_model(model, loader, target_device)
    _write_json(destination / "test_metrics.json", {"task": "image_classification",
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "dataset_json_sha256": summary["dataset_json_sha256"], "metrics": metrics})
    _write_json(destination / "test_predictions.json", predictions)
    return metrics
