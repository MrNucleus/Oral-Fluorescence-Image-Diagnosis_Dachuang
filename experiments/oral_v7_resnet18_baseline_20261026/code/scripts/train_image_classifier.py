"""Whole-image classification entry point (no detection/box labels)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="口腔完整图片二分类训练")
    parser.add_argument("--data", required=True, help="新版整图分类包 dataset.json")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                               / "configs/train_image_classifier.yaml"))
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50"])
    for key in ("epochs", "batch", "imgsz", "workers"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--augmentation")
    parser.add_argument("--profile", choices=["baseline", "geometry", "modality"])
    parser.add_argument("--modalities", nargs="+", choices=["standard", "fluorescence"])
    parser.add_argument("--no-pretrained", action="store_true", help="随机初始化，不下载权重")
    parser.add_argument("--check-only", action="store_true", help="只预检，不构建模型或训练")
    parser.add_argument("--resume", help="从 last.pt 或 best.pt 继续训练")
    args = parser.parse_args()
    from oral_fluorescence_detection.image_classification import train
    overrides = {k: getattr(args, k) for k in ("model", "epochs", "batch", "imgsz", "device", "workers")
                 if getattr(args, k) is not None}
    if args.no_pretrained:
        overrides["pretrained"] = False
    train(args.data, config_yaml=args.config, name=args.name, overrides=overrides,
          augmentation_yaml=args.augmentation, profile=args.profile,
          check_only=args.check_only, modalities=args.modalities, resume=args.resume)


if __name__ == "__main__":
    main()
