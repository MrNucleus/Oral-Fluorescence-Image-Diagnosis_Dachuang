import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss：缓解类别不平衡问题"""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        log_pt = F.log_softmax(preds, dim=1)
        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        loss = F.nll_loss(log_pt * focal_weight, targets, reduction="none")
        if self.alpha is not None:
            alpha = self.alpha.to(preds.device)
            alpha_weight = alpha[targets]
            loss = loss * alpha_weight
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(name, num_classes=None, device=None):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "focal":
        return FocalLoss(alpha=None, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss: {name}")
