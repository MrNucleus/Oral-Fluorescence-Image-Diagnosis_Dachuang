"""训练入口：python scripts/train.py [--epochs 50] [--model yolov8s.pt] [--name my_run]"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.yolo_train import train
from src.utils.paths import DATASETS_DIR, CONFIGS_DIR


def main():
    parser = argparse.ArgumentParser(description="YOLO 训练")
    parser.add_argument("--data", default=str(DATASETS_DIR / "oral_fluorescence" / "data.yaml"))
    parser.add_argument("--config", default=str(CONFIGS_DIR / "train.yaml"))
    parser.add_argument("--model", default=None, help="覆盖配置中的模型，如 yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    overrides = {}
    if args.model: overrides["model"] = args.model
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.batch is not None: overrides["batch"] = args.batch
    if args.imgsz is not None: overrides["imgsz"] = args.imgsz
    if args.device is not None: overrides["device"] = args.device

    train(data_yaml=args.data, config_yaml=args.config, name=args.name, overrides=overrides)


if __name__ == "__main__":
    main()
