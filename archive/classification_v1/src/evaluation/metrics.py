import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=5):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes)
    )
    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = precision[i]
        metrics[f"recall_class_{i}"] = recall[i]
        metrics[f"f1_class_{i}"] = f1[i]
    metrics["f1_macro"] = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )[2]
    if y_prob is not None and num_classes > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y_true_onehot = np.eye(num_classes)[y_true]
                metrics["auc_macro"] = roc_auc_score(
                    y_true_onehot, y_prob, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["auc_macro"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return metrics
