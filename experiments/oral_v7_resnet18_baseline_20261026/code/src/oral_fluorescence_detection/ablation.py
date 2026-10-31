"""Feature-map attention and training-only loss variants for controlled ablations."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODELS = ("resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3", "densenet121")
ATTENTIONS = ("none", "se", "cbam")
LOSSES = ("ce", "weighted_ce", "label_smoothing", "focal")

class ChannelAttention(nn.Module):
    def __init__(self, channels, use_max=False):
        super().__init__()
        self.use_max = use_max
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(1, channels//16), 1), nn.ReLU(),
                                 nn.Conv2d(max(1, channels//16), channels, 1))
    def forward(self, x):
        score = self.mlp(F.adaptive_avg_pool2d(x, 1))
        if self.use_max:
            score = score + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * score.sigmoid()

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels, True)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        x = self.channel(x)
        summary = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * self.spatial(summary).sigmoid()

def attach_attention(model, name, attention):
    if attention not in ATTENTIONS:
        raise ValueError("Unknown attention")
    if attention == "none":
        return model
    if name.startswith("resnet"):
        channels, location = model.fc.in_features, "layer4"
    elif name.startswith("efficientnet"):
        channels, location = model.classifier[-1].in_features, "features"
    else:
        channels, location = model.classifier.in_features, "features"
    module = ChannelAttention(channels) if attention == "se" else CBAM(channels)
    # After the last feature stage, before the backbone's own pooling/classifier.
    setattr(model, location, nn.Sequential(getattr(model, location), module))
    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        return ((1 - torch.exp(-ce)).pow(self.gamma) * ce).mean()

def build_loss(name, train_labels, smoothing=0.1, gamma=2.0):
    if name not in LOSSES or not 0 <= smoothing < 1 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid loss configuration")
    counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=2)
    if len(counts) != 2 or (counts == 0).any():
        raise ValueError("Training labels must contain both binary classes")
    weights = len(train_labels) / (2 * counts.float())
    if name == "focal":
        criterion = FocalLoss(gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights if name == "weighted_ce" else None,
                                       label_smoothing=smoothing if name == "label_smoothing" else 0)
    return criterion, {"name": name, "class_counts": counts.tolist(),
                       "class_weights": weights.tolist() if name == "weighted_ce" else None,
                       "smoothing": smoothing if name == "label_smoothing" else 0,
                       "gamma": gamma if name == "focal" else 0}
