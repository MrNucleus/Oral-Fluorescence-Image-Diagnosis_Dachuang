"""Evaluate a frozen whole-image checkpoint on test, separately from training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="冻结整图分类模型的最终测试")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="包外尚未存在的结果目录")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    from oral_fluorescence_detection.image_classification import evaluate_checkpoint
    metrics = evaluate_checkpoint(args.checkpoint, args.data, args.output,
                                  args.device, args.batch, args.workers)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
