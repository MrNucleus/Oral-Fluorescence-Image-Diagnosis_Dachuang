"""推理入口：python scripts/predict.py --weights best.pt --source path/to/image.jpg"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infer.predict import predict


def main():
    parser = argparse.ArgumentParser(description="YOLO 推理")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--source", required=True, help="图片/目录路径")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    predict(args.weights, args.source, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, name=args.name)


if __name__ == "__main__":
    main()
