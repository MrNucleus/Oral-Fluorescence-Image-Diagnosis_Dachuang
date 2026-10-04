"""评估入口：python scripts/evaluate.py --weights experiments/xxx/weights/best.pt --split test"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.yolo_eval import evaluate
from src.utils.paths import DATASETS_DIR


def main():
    parser = argparse.ArgumentParser(description="YOLO 评估")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--data", default=str(DATASETS_DIR / "oral_fluorescence" / "data.yaml"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--name", default="eval")
    parser.add_argument("--device", default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    evaluate(args.weights, args.data, split=args.split, name=args.name,
             device=args.device, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
