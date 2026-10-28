"""Create report figures for the ResNet-50 whole-image baseline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    PROJECT_ROOT
    / "experiments"
    / "oral_v7_resnet18_baseline_20261026"
    / "code"
    / "outputs"
    / "image_classifier_runs"
)
RESNET50_RUN = RUNS / "resnet50_whole_image_geometry_20261027_pycharm"
RESNET18_RUN = RUNS / "v7_resnet18_original_20261026"
OUTPUT = PROJECT_ROOT / "reports" / "figures" / "resnet50_baseline_20261027"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AEB8BE")
    axis.spines["bottom"].set_color("#AEB8BE")
    axis.tick_params(colors="#42535B", labelsize=9)
    axis.grid(axis="y", color="#E4E8EA", linewidth=0.8)
    axis.set_axisbelow(True)


def training_curves():
    history = read_json(RESNET50_RUN / "history.json")
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_metrics"]["loss"] for entry in history]
    auroc = [entry["val_metrics"]["auroc"] for entry in history]
    balanced = [entry["val_metrics"]["balanced_accuracy"] for entry in history]
    best_epoch = 35

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    axes[0].plot(epochs, train_loss, color="#1F5A7A", linewidth=2, label="Train")
    axes[0].plot(epochs, val_loss, color="#B54A3A", linewidth=2, label="Validation")
    axes[0].axvline(best_epoch, color="#73848C", linewidth=1, linestyle="--", alpha=0.8)
    axes[0].set_title("Loss", loc="left", fontsize=12, fontweight="bold", color="#17252B")
    axes[0].set_xlabel("Epoch", color="#42535B")
    axes[0].set_ylabel("Cross-entropy", color="#42535B")
    axes[0].legend(frameon=False, fontsize=9)
    style_axis(axes[0])

    axes[1].plot(epochs, auroc, color="#2F6B4F", linewidth=2, marker="o", markersize=2.5, label="AUROC")
    axes[1].plot(epochs, balanced, color="#B07123", linewidth=2, marker="o", markersize=2.5, label="Balanced accuracy")
    axes[1].axvline(best_epoch, color="#73848C", linewidth=1, linestyle="--", alpha=0.8)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_title("Validation discrimination", loc="left", fontsize=12, fontweight="bold", color="#17252B")
    axes[1].set_xlabel("Epoch", color="#42535B")
    axes[1].set_ylabel("Score", color="#42535B")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    style_axis(axes[1])
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.savefig(OUTPUT / "training_curves.png", dpi=220, facecolor="white")
    plt.close(fig)


def test_comparison():
    resnet18 = read_json(RESNET18_RUN / "training_record_20261027.json")["frozen_test"]
    resnet50 = read_json(RESNET50_RUN / "training_record_20261027.json")["frozen_test"]
    labels = ["AUROC", "Sensitivity", "Specificity", "Balanced accuracy"]
    keys = ["auroc", "sensitivity", "specificity", "balanced_accuracy"]
    x = list(range(len(labels)))
    width = 0.34

    fig, axis = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    bars18 = axis.bar(
        [value - width / 2 for value in x],
        [resnet18[key] for key in keys],
        width,
        color="#5C7F9B",
        label="ResNet18",
    )
    bars50 = axis.bar(
        [value + width / 2 for value in x],
        [resnet50[key] for key in keys],
        width,
        color="#B07123",
        label="ResNet50",
    )
    for bars in (bars18, bars50):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#25343B",
            )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score", color="#42535B")
    axis.set_title("Frozen test comparison", loc="left", fontsize=12, fontweight="bold", color="#17252B")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(axis)
    fig.savefig(OUTPUT / "test_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def stratified_auroc():
    strat = read_json(
        RESNET50_RUN / "final_test_20261027_1345" / "stratified_metrics.json"
    )
    label_map = {
        "tongue_floor_broad": "Tongue/floor",
        "gingivopalatal_broad": "Gingival/palatal",
        "buccal_labial_vestibular_broad": "Buccal/labial/vestibular",
        "fluorescence": "Fluorescence",
        "standard": "Standard",
    }
    rows = [row for row in strat["rows"] if row["dimension"] in {"site", "modality"}]
    labels = [f'{label_map[row["group"]]} (n={row["images"]})' for row in rows]
    values = [row["auroc"] for row in rows]
    colors = ["#315F78" if row["dimension"] == "site" else "#9C6B2E" for row in rows]

    fig, axis = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    axis.axvline(0.5, color="#73848C", linewidth=1, linestyle="--", alpha=0.8)
    axis.set_xlim(0.4, 1.0)
    axis.set_xlabel("AUROC", color="#42535B")
    axis.set_title("ResNet50 frozen-test AUROC by subgroup", loc="left", fontsize=12, fontweight="bold", color="#17252B")
    axis.grid(axis="x", color="#E4E8EA", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.annotate(
            f"{bar.get_width():.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#25343B",
        )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AEB8BE")
    axis.spines["bottom"].set_color("#AEB8BE")
    axis.tick_params(colors="#42535B", labelsize=9)
    fig.savefig(OUTPUT / "stratified_auroc.png", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training_curves()
    test_comparison()
    stratified_auroc()
    print(OUTPUT)


if __name__ == "__main__":
    main()
