"""原始数据探索：统计类别、样本量、图像尺寸、每患者分布等。

只读操作，不改动原始数据；结果打印并写入 reports/。
"""

import json
from collections import Counter, defaultdict

from .voc import parse_voc
from .build_dataset import discover_samples, patient_key
from ..utils.paths import REPORTS_DIR, ensure_dir

try:
    from PIL import Image
except Exception:
    Image = None


def analyze(save=True, sample_size_for_size=60):
    samples = discover_samples()
    n_images = len(samples)
    n_annotated = sum(1 for s in samples if s["xml"])

    cls_counter = Counter()
    objs_per_img = []
    per_patient = defaultdict(lambda: {"images": 0, "annotated": 0, "objects": 0})
    sizes = []

    for i, s in enumerate(samples):
        p = s["patient"]
        per_patient[p]["images"] += 1
        if s["xml"]:
            ann = parse_voc(s["xml"])
            per_patient[p]["annotated"] += 1
            per_patient[p]["objects"] += len(ann.boxes)
            objs_per_img.append(len(ann.boxes))
            for b in ann.boxes:
                cls_counter[b.cls_name] += 1

        if Image is not None and i < sample_size_for_size and s["xml"]:
            try:
                with Image.open(s["image"]) as im:
                    sizes.append(im.size)
            except Exception:
                pass

    report = {
        "n_images": n_images,
        "n_annotated": n_annotated,
        "n_background": n_images - n_annotated,
        "n_patients": len(per_patient),
        "class_instances": dict(cls_counter),
        "objects_per_image": {
            "min": min(objs_per_img) if objs_per_img else 0,
            "max": max(objs_per_img) if objs_per_img else 0,
            "mean": round(sum(objs_per_img) / len(objs_per_img), 2) if objs_per_img else 0,
            "n_multi_object_xml": sum(1 for x in objs_per_img if x > 1),
        },
        "image_sizes_sample": sorted({f"{w}x{h}" for w, h in sizes}),
        "per_patient": dict(per_patient),
    }

    print("========== 数据探索 ==========")
    print(f"图片总数      : {report['n_images']}")
    print(f"有标注图片    : {report['n_annotated']}")
    print(f"无标注(背景)  : {report['n_background']}")
    print(f"患者数        : {report['n_patients']}")
    print(f"类别实例分布  : {report['class_instances']}")
    print(f"每图目标数    : {report['objects_per_image']}")
    print(f"图像尺寸(样本): {report['image_sizes_sample']}")

    if save:
        ensure_dir(REPORTS_DIR)
        out = REPORTS_DIR / "data_analysis.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存：{out}")

    return report
