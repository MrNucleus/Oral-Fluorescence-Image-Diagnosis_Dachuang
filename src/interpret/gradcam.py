import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
        activations = self.activation[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap


def visualize_gradcam(model, image_tensor, device, class_names, target_layer, save_path=None):
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)
    img_array = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_array = img_array * np.array([0.229, 0.224, 0.225])
    img_array = img_array + np.array([0.485, 0.456, 0.406])
    img_array = np.clip(img_array, 0, 1)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    superimposed = heatmap_colored * 0.4 + img_array * 0.6
    superimposed = np.clip(superimposed, 0, 1)
    with torch.no_grad():
        pred_idx = model(input_tensor).argmax(dim=1).item()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay (Pred: {class_names[pred_idx]})")
    axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def find_target_layer(model, backbone_name="resnet50"):
    if backbone_name == "resnet50":
        return model.backbone.layer4[-1]
    elif "efficientnet" in backbone_name:
        return model.backbone.features[-1]
    return model.backbone.layer4[-1]
