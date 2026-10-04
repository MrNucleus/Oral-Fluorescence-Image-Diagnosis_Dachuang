"""YOLO 评估封装：在 val/test 集上计算 mAP、P、R 等指标并保存。"""

import json
from pathlib import Path

from ..utils.paths import REPORTS_DIR, ensure_dir


def evaluate(weights, data_yaml, split="test", name="eval", device=0, imgsz=640):
    """在指定数据集划分上评估模型。

    返回包含整体指标的 dict，并保存到 reports/<name>_metrics.json。
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                        device=device, name=name, project=str(REPORTS_DIR / "val_runs"))

    metrics = {
        "split": split,
        "weights": str(weights),
        "mAP50": float(getattr(results.box, "map50", float("nan"))),
        "mAP50-95": float(getattr(results.box, "map", float("nan"))),
        "precision": float(getattr(results.box, "mp", float("nan"))),
        "recall": float(getattr(results.box, "mr", float("nan"))),
    }

    ensure_dir(REPORTS_DIR)
    out = REPORTS_DIR / f"{name}_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== 评估结果 ==========")
    for k, v in metrics.items():
        print(f"{k:10s}: {v}")
    print(f"已保存：{out}")
    return metrics
