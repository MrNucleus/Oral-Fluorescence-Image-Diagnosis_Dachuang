"""Visualization helpers for the binary classification experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sklearn.metrics import roc_curve


STAGE_COLORS = {1: "#d9e8f5", 2: "#f7e4c6"}
TRAIN_COLOR = "#1f5a94"
VAL_COLOR = "#c25b2a"
ACCENT = "#2d7f73"


def load_history(path: Path | str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _history_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    numeric = [
        "epoch",
        "stage",
        "loss",
        "accuracy",
        "val_image_balanced_accuracy",
        "val_patient_balanced_accuracy",
        "val_patient_auroc",
    ]
    return {
        key: np.asarray(
            [float(row[key]) if row[key] not in ("", None) else np.nan for row in history]
        )
        for key in numeric
    }


def _stage_regions(ax, values: dict[str, np.ndarray]) -> None:
    stages = values["stage"].astype(int)
    epochs = values["epoch"]
    for stage in sorted(set(stages.tolist())):
        selected = epochs[stages == stage]
        if len(selected) == 0:
            continue
        ax.axvspan(
            selected.min() - 0.45,
            selected.max() + 0.45,
            color=STAGE_COLORS.get(stage, "#eeeeee"),
            alpha=0.8,
            zorder=0,
        )


def plot_training_curves(
    history: list[dict],
    output_path: Path | str,
    best_epoch: int | None = None,
) -> Path:
    values = _history_arrays(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("#f7f6f2")
    for ax in axes.ravel():
        ax.set_facecolor("#ffffff")
        _stage_regions(ax, values)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 0]
    ax.plot(values["epoch"], values["loss"], marker="o", color=TRAIN_COLOR, label="Train CE")
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross entropy")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(
        values["epoch"],
        values["accuracy"],
        marker="o",
        color=TRAIN_COLOR,
        label="Train accuracy",
    )
    ax.plot(
        values["epoch"],
        values["val_image_balanced_accuracy"],
        marker="s",
        color=VAL_COLOR,
        label="Val image balanced acc.",
    )
    ax.plot(
        values["epoch"],
        values["val_patient_balanced_accuracy"],
        marker="^",
        color=ACCENT,
        label="Val patient balanced acc.",
    )
    ax.set_ylim(0, 1.03)
    ax.set_title("Accuracy across stages")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(
        values["epoch"],
        values["val_patient_auroc"],
        marker="o",
        color=ACCENT,
        linewidth=2.2,
    )
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1, label="Chance")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("Validation patient AUROC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for stage in sorted(set(values["stage"].astype(int).tolist())):
        mask = values["stage"].astype(int) == stage
        ax.plot(
            values["epoch"][mask],
            values["val_image_balanced_accuracy"][mask],
            marker="o",
            linewidth=2,
            label=f"Stage {stage}",
        )
    ax.set_ylim(0, 1.03)
    ax.set_title("Validation image balanced accuracy by stage")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False)

    if best_epoch is not None:
        for ax in axes.ravel():
            ax.axvline(best_epoch, color="#8b1e3f", linestyle="--", linewidth=1.2)
            ax.text(
                best_epoch,
                ax.get_ylim()[1],
                " best",
                color="#8b1e3f",
                fontsize=8,
                va="top",
            )

    fig.suptitle("Binary classification training process", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def animate_training_process(
    history: list[dict],
    output_path: Path | str,
    best_epoch: int | None = None,
) -> Path:
    values = _history_arrays(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = values["epoch"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#ffffff")

    def update(frame_index: int):
        ax.clear()
        ax.set_facecolor("#ffffff")
        upto = frame_index + 1
        _stage_regions(ax, {key: value[:upto] for key, value in values.items()})
        ax.plot(
            epochs[:upto],
            values["accuracy"][:upto],
            marker="o",
            color=TRAIN_COLOR,
            label="Train accuracy",
        )
        ax.plot(
            epochs[:upto],
            values["val_image_balanced_accuracy"][:upto],
            marker="s",
            color=VAL_COLOR,
            label="Val image balanced acc.",
        )
        ax.plot(
            epochs[:upto],
            values["val_patient_balanced_accuracy"][:upto],
            marker="^",
            color=ACCENT,
            label="Val patient balanced acc.",
        )
        ax.plot(
            epochs[:upto],
            values["val_patient_auroc"][:upto],
            marker="D",
            color="#6f4e7c",
            linestyle="--",
            label="Val patient AUROC",
        )
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
        if best_epoch is not None and best_epoch <= epochs[frame_index]:
            ax.axvline(best_epoch, color="#8b1e3f", linestyle="--", linewidth=1.2)
        ax.set_xlim(0.5, epochs[-1] + 0.5)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"Training process, epoch {epochs[frame_index]}")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        return (ax,)

    animation = FuncAnimation(fig, update, frames=len(history), interval=500)
    animation.save(output_path, writer=PillowWriter(fps=2))
    plt.close(fig)
    return output_path


def plot_test_overview(
    image_labels: list[int],
    image_probabilities: list[float],
    test_metrics: dict,
    threshold: float,
    output_path: Path | str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.patch.set_facecolor("#f7f6f2")
    for ax in axes.ravel():
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)

    patients = test_metrics["patients"]
    labels = np.asarray([row["label"] for row in patients], dtype=int)
    probabilities = np.asarray([row["mean_probability"] for row in patients], dtype=float)
    positions = np.asarray([0 if label == 0 else 1 for label in labels], dtype=float)
    rng = np.random.default_rng(42)
    positions = positions + rng.uniform(-0.08, 0.08, size=len(positions))

    ax = axes[0, 0]
    colors = np.where(labels == 0, "#4f7cac", "#c25b2a")
    ax.scatter(probabilities, positions, c=colors, s=80, edgecolor="white", linewidth=0.8)
    ax.axvline(threshold, color="#8b1e3f", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_yticks([0, 1], ["n (negative)", "l (positive)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean predicted probability (positive l)")
    ax.set_title("Patient-level test probabilities")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[0, 1]
    image_labels_array = np.asarray(image_labels, dtype=int)
    image_prob_array = np.asarray(image_probabilities, dtype=float)
    for label, color, name in (
        (0, "#4f7cac", "n negative"),
        (1, "#c25b2a", "l positive"),
    ):
        mask = image_labels_array == label
        fpr, tpr, _ = roc_curve(mask, image_prob_array)
        auc = test_metrics["image_level"]["auroc"]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name}, image AUROC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#777777")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Image-level ROC")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    for ax, level, title in (
        (axes[1, 0], "image_level", "Image-level confusion matrix"),
        (axes[1, 1], "patient_level", "Patient-level confusion matrix"),
    ):
        matrix = np.asarray(test_metrics[level]["confusion_matrix"], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > matrix.max() * 0.55 else "#222222",
                    fontsize=14,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1], ["Pred l", "Pred n"])
        ax.set_yticks([0, 1], ["True l", "True n"])
        ax.set_title(title)
        ax.set_xlabel(f"Balanced acc. {test_metrics[level]['balanced_accuracy']:.3f}")

    patient_ba = test_metrics["patient_level"]["balanced_accuracy"]
    patient_auc = test_metrics["patient_level"]["auroc"]
    fig.suptitle(
        f"Test overview | patient balanced acc. {patient_ba:.3f} | "
        f"patient AUROC {patient_auc:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_dashboard(
    output_path: Path | str,
    experiment_name: str,
    summary: dict,
    image_paths: dict[str, Path],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patient = summary["test"]["patient_level"]
    image = summary["test"]["image_level"]
    relative = {key: Path(path).name for key, path in image_paths.items()}
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{experiment_name} training dashboard</title>
  <style>
    body {{ margin: 0; background: #f2f0ea; color: #202124;
           font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    p {{ color: #5f6368; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                margin: 24px 0; }}
    .metric {{ background: white; border: 1px solid #d9d6ce; border-radius: 8px;
               padding: 16px; }}
    .metric strong {{ display: block; font-size: 24px; color: #1f5a94; margin-top: 6px; }}
    section {{ background: white; border: 1px solid #d9d6ce; border-radius: 8px;
               padding: 16px; margin: 16px 0; }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ background: #edeae3; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 760px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>{experiment_name}</h1>
  <p>Patient-level split. Threshold selected on validation only.</p>
  <div class="metrics">
    <div class="metric">Patient AUROC<strong>{patient['auroc']:.3f}</strong></div>
    <div class="metric">Patient balanced acc.<strong>{patient['balanced_accuracy']:.3f}</strong></div>
    <div class="metric">Sensitivity<strong>{patient['sensitivity']:.3f}</strong></div>
    <div class="metric">Specificity<strong>{patient['specificity']:.3f}</strong></div>
  </div>
  <section><img src="{relative['training']}" alt="Training curves"></section>
  <section><img src="{relative['overview']}" alt="Test overview"></section>
  <p>Image-level AUROC: <code>{image['auroc']:.3f}</code>.
     Animation: <a href="{relative['animation']}">training_process.gif</a>.</p>
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def load_metrics(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

