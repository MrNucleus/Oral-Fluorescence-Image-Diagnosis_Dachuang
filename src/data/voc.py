"""解析 PASCAL VOC 格式的 XML 标注，并转换为 YOLO 标签行。"""

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class VOCBox:
    cls_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VOCAnnotation:
    filename: str
    width: int
    height: int
    boxes: list = field(default_factory=list)


def parse_voc(xml_path) -> VOCAnnotation:
    """解析一个 VOC XML 文件。"""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    filename = (root.findtext("filename") or "").strip()

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or not name:
            continue
        boxes.append(VOCBox(
            cls_name=name,
            xmin=float(bb.findtext("xmin", "0")),
            ymin=float(bb.findtext("ymin", "0")),
            xmax=float(bb.findtext("xmax", "0")),
            ymax=float(bb.findtext("ymax", "0")),
        ))

    return VOCAnnotation(filename=filename, width=width, height=height, boxes=boxes)


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def to_yolo_lines(ann: VOCAnnotation, class_map: dict) -> list:
    """将标注框转换为 YOLO 格式行：<cls> <xc> <yc> <w> <h>（归一化）。

    忽略 class_map 中不存在的类别，并丢弃非法（零面积）框。
    """
    lines = []
    if ann.width <= 0 or ann.height <= 0:
        return lines

    for b in ann.boxes:
        if b.cls_name not in class_map:
            continue
        x1, x2 = sorted((b.xmin, b.xmax))
        y1, y2 = sorted((b.ymin, b.ymax))
        bw = (x2 - x1) / ann.width
        bh = (y2 - y1) / ann.height
        if bw <= 0 or bh <= 0:
            continue
        xc = ((x1 + x2) / 2.0) / ann.width
        yc = ((y1 + y2) / 2.0) / ann.height
        lines.append(
            f"{class_map[b.cls_name]} {clip01(xc):.6f} {clip01(yc):.6f} "
            f"{clip01(bw):.6f} {clip01(bh):.6f}"
        )
    return lines
