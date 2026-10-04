"""YOLO 训练封装：读取训练配置，调用 ultralytics 进行训练。

依赖：pip install ultralytics（会自动带 torch/torchvision）。
训练产物默认保存在 experiments/<name>/ 下，便于每次实验留档。
"""

import shutil
from pathlib import Path

from ..utils.paths import CONFIGS_DIR, EXPERIMENTS_DIR, ensure_dir
from ..utils.seed import set_seed


def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(data_yaml, config_yaml=None, name=None, overrides=None):
    """执行一次 YOLO 训练。

    参数：
      data_yaml   : 生成的 data.yaml 路径（datasets/.../data.yaml）
      config_yaml : 训练超参配置（默认 configs/train.yaml）
      name        : 本次实验名（默认取配置里的 name）
      overrides   : dict，覆盖任意 ultralytics 超参，如 {"epochs": 50}
    """
    from ultralytics import YOLO

    config_yaml = Path(config_yaml) if config_yaml else (CONFIGS_DIR / "train.yaml")
    cfg = load_yaml(config_yaml)
    overrides = dict(overrides or {})
    run_name = name or cfg.get("name", "yolo_run")

    set_seed(int(cfg.get("seed", 42)))

    # model 允许被 overrides 覆盖
    model_name = overrides.pop("model", cfg.get("model", "yolov8n.pt"))
    model = YOLO(model_name)

    train_args = {
        "data": str(data_yaml),
        "epochs": cfg.get("epochs", 100),
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 16),
        "lr0": cfg.get("lr0", 0.01),
        "optimizer": cfg.get("optimizer", "auto"),
        "patience": cfg.get("patience", 20),
        "device": cfg.get("device", 0),
        "workers": cfg.get("workers", 8),
        "project": str(EXPERIMENTS_DIR),
        "name": run_name,
        "seed": int(cfg.get("seed", 42)),
        "pretrained": cfg.get("pretrained", True),
    }
    if overrides:
        train_args.update(overrides)

    print(f"开始训练：model={model_name} epochs={train_args['epochs']} "
          f"imgsz={train_args['imgsz']} batch={train_args['batch']} device={train_args['device']}")
    model.train(**train_args)

    # ultralytics 默认把结果放在 project/name 下；把最佳权重位置打印出来
    best = EXPERIMENTS_DIR / run_name / "weights" / "best.pt"
    print(f"训练完成，最佳权重：{best}")
    return best
