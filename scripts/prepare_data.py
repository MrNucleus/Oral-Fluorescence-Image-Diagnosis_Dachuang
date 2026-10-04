"""构建 YOLO 数据集：python scripts/prepare_data.py [--no-background] [--link-mode copy]"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.build_dataset import build
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="VOC -> YOLO 数据集构建")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "dataset.yaml"))
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink"], default=None)
    parser.add_argument("--no-background", action="store_true",
                        help="不把无标注图片作为背景样本")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    class_map = cfg["class_map"]
    split_cfg = cfg.get("split", {})
    ratios = tuple(split_cfg.get("ratios", [0.7, 0.15, 0.15]))
    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    include_background = (not args.no_background) and split_cfg.get("include_background", True)
    link_mode = args.link_mode or cfg.get("link_mode", "hardlink")
    output = PROJECT_ROOT / cfg.get("output", "datasets/oral_fluorescence")

    build(class_map, names, ratios=ratios, seed=seed, output_dir=output,
          link_mode=link_mode, include_background=include_background)


if __name__ == "__main__":
    main()
