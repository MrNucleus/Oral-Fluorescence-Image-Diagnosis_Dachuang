"""YOLO 推理封装：对单张图/目录/整个数据集做预测并保存可视化结果。"""

from pathlib import Path

from ..utils.paths import REPORTS_DIR


def predict(weights, source, conf=0.25, iou=0.45, imgsz=640, device=0,
            save=True, save_dir=None, name="predict"):
    """对 source 执行推理。

    source 可以是图片路径、目录，或自定义格式的路径列表。
    返回 ultralytics Results 列表。
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    project = Path(save_dir) if save_dir else (REPORTS_DIR / "predictions")
    results = model.predict(source=str(source), conf=conf, iou=iou, imgsz=imgsz,
                            device=device, save=save, project=str(project), name=name)
    print(f"推理完成，共 {len(results)} 张，结果保存于：{project / name}")
    return results
