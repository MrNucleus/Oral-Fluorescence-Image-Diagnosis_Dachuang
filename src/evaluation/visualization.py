import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_curves(y_true, y_prob, class_names, save_path=None, figsize=(12, 10)):
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_classification_report(metrics, class_names, save_path=None, figsize=(10, 6)):
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.25
    precision = [metrics.get(f"precision_class_{i}", 0) for i in range(n_classes)]
    recall = [metrics.get(f"recall_class_{i}", 0) for i in range(n_classes)]
    f1 = [metrics.get(f"f1_class_{i}", 0) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.set_title("Per-Class Performance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
