"""PyCharm entry point for live EfficientNet-B3 training."""

from __future__ import annotations

import sys
from pathlib import Path

from train_binary_plan import main


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / "configs" / "binary_lpos_effnet40_live.yaml"
    sys.argv = [
        "train_binary_plan.py",
        "--config",
        str(config),
    ]
    main()

