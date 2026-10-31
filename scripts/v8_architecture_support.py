"""Architecture-specific stage definitions; fixed section 4.3 training conditions."""
import torch
from torch import nn
from torchvision import models

WEIGHTS = {
    'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
    'efficientnet_b3': models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    'densenet121': models.DenseNet121_Weights.IMAGENET1K_V1,
}
HEADS = {'resnet50': 'fc', 'efficientnet_b3': 'classifier', 'densenet121': 'classifier'}
LAST_STAGES = {
    'resnet50': ('layer3', 'layer4'),
    # Final two MBConv stages plus their terminal projection.
    'efficientnet_b3': ('features.6', 'features.7', 'features.8'),
    # Final two dense blocks plus connecting transition and terminal BN.
    'densenet121': ('features.denseblock3', 'features.transition3', 'features.denseblock4', 'features.norm5'),
}


def build_classifier(name, pretrained=True):
    if name not in WEIGHTS:
        raise ValueError('Unsupported architecture')
    model = models.get_model(name, weights=WEIGHTS[name] if pretrained else None)
    if name == 'resnet50':
        model.fc = nn.Sequential(nn.Dropout(.5), nn.Linear(model.fc.in_features, 2))
    elif name == 'efficientnet_b3':
        model.classifier = nn.Sequential(nn.Dropout(.5), nn.Linear(model.classifier[-1].in_features, 2))
    else:
        model.classifier = nn.Sequential(nn.Dropout(.5), nn.Linear(model.classifier.in_features, 2))
    return model


def configure_stage(model, optimizer, name, stage):
    if stage not in (1, 2, 3):
        raise ValueError('Unknown stage')
    active = (HEADS[name],) + (LAST_STAGES[name] if stage == 2 else ())
    if stage == 3:
        model.train()
    else:
        # Freeze stochastic depth and BN buffers throughout inactive branches too.
        model.eval()
        for prefix in active:
            model.get_submodule(prefix).train()
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(stage == 3 or any(key == p or key.startswith(p+'.') for p in active))
        parameter.grad = None
    for group in optimizer.param_groups:
        group['lr'] = (1e-3,1e-4,1e-5)[stage-1]
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
