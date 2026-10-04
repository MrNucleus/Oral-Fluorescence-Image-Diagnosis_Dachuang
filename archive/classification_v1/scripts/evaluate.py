import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import json

from src.config import Config
from src.data.dataset import OralFluorescenceDataset
from src.data.transforms import get_val_transforms
from src.models.backbone import create_model
from src.evaluation.metrics import evaluate, compute_metrics
from src.evaluation.visualization import plot_confusion_matrix, plot_roc_curves, plot_classification_report
from src.utils.seed import set_seed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="evaluation")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded: {args.checkpoint} (acc: {checkpoint['accuracy']:.2f}%)")

    transform = get_val_transforms(config.data.img_size)
    test_dataset = OralFluorescenceDataset(
        config.data.data_root,
        os.path.join(config.data.split_dir, config.data.test_split),
        transform=transform, class_names=config.data.class_names,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                             shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, config.data.num_classes)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    if "auc_macro" in metrics:
        print(f"AUC Macro: {metrics['auc_macro']:.4f}")

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"{args.exp_name}_metrics.json")
    serializable = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved: {report_path}")

    class_names = config.data.class_names
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          save_path=os.path.join("reports", f"{args.exp_name}_confusion.png"))
    plot_roc_curves(y_true, y_prob, class_names,
                    save_path=os.path.join("reports", f"{args.exp_name}_roc.png"))
    plot_classification_report(metrics, class_names,
                               save_path=os.path.join("reports", f"{args.exp_name}_per_class.png"))
    print("Figures saved to reports/")


if __name__ == "__main__":
    main()
