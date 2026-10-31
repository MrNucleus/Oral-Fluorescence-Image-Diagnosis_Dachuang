import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/oral_v7_resnet18_baseline_20261026/code/src"))
from oral_fluorescence_detection.ablation import build_loss
from oral_fluorescence_detection.image_classification import build_model

class AblationTests(unittest.TestCase):
    def test_focal_gamma_zero_equals_ce(self):
        x = torch.tensor([[2., -1.], [-.5, .3]], requires_grad=True)
        y = torch.tensor([0, 1])
        loss, _ = build_loss("focal", y.tolist(), gamma=0)
        self.assertTrue(torch.allclose(loss(x,y), F.cross_entropy(x,y)))
        loss(x,y).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_weights_use_training_counts(self):
        _, meta = build_loss("weighted_ce", [0, 0, 0, 1])
        self.assertAlmostEqual(meta["class_weights"][0], 2/3, places=6)
        self.assertEqual(meta["class_weights"][1], 2)

    def test_model_attention_backward_and_reload(self):
        torch.set_num_threads(2)
        for name in ("resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3", "densenet121"):
            for attention in ("none", "se", "cbam"):
                with self.subTest(name=name, attention=attention):
                    model = build_model(name, False, attention).eval()
                    x = torch.randn(2,3,64,64)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (2,2))
                    for kind in ("ce", "weighted_ce", "label_smoothing", "focal"):
                        criterion, _ = build_loss(kind, [0,0,1])
                        value = criterion(logits, torch.tensor([0,1]))
                        self.assertTrue(torch.isfinite(value))
                    value.backward()
                    if attention != "none":
                        params = [p for n,p in model.named_parameters() if "mlp" in n]
                        self.assertTrue(params)
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                    clone = build_model(name, False, attention).eval()
                    clone.load_state_dict(model.state_dict())
                    with torch.no_grad():
                        self.assertTrue(torch.allclose(logits, clone(x)))
