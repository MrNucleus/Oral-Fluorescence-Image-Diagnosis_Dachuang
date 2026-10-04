import torch.nn as nn
import torchvision.models as models


def build_backbone(backbone_name="resnet50", pretrained=True):
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif backbone_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return model, in_features


class OralClassificationModel(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=5, dropout=0.3,
                 pretrained=True, use_attention=True, attention_type="cbam"):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, in_features = build_backbone(backbone_name, pretrained)

        self.use_attention = use_attention
        if use_attention:
            if attention_type == "cbam":
                from .attention import CBAM
                self.attention = CBAM(in_features)
            elif attention_type == "se":
                from .attention import SEModule
                self.attention = SEModule(in_features)
            else:
                self.attention = nn.Identity()
        else:
            self.attention = nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if "efficientnet" in backbone_name else nn.Identity(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.attention(x)
        return self.classifier(x)


def create_model(cfg):
    model = OralClassificationModel(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.data.num_classes,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        use_attention=cfg.model.use_attention,
        attention_type=cfg.model.attention_type,
    )
    return model
