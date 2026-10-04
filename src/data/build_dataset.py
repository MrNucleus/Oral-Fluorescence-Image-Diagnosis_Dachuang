"""把 VOC 标注的原始数据构建为 YOLO 训练数据集。

关键点：
1. 按【患者】划分 train/val/test，避免同一患者的多张图分散到不同子集造成数据泄露。
2. 无标注的图片（如 IMG_*.JPG）默认作为背景样本（空标签），可用 --no-background 关闭。
3. 图片可采用 硬链接/软链接/复制 三种方式落盘，硬链接不额外占空间。
"""

import json
import re
import shutil
from pathlib import Path

from .voc import parse_voc, to_yolo_lines
from ..utils.paths import (DATA_IMAGES, DATA_ANNOTATIONS, DATASETS_DIR, ensure_dir)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def patient_key(folder_name: str) -> str:
    """从文件夹名推导患者标识，用于按患者分组。

    规则：纯数字（001）直接作为标识；否则取开头到第一个空格/数字/括号之前的中文姓名。
    例如 "程福全 牙龈 20250703（+）" -> "程福全"；"李景涛20250520（-）" -> "李景涛"。
    """
    name = folder_name.strip()
    if re.fullmatch(r"\d+", name):
        return name
    m = re.match(r"^[\u4e00-\u9fffA-Za-z]+", name)
    return m.group(0) if m else name


def discover_samples():
    """扫描原始目录，返回样本列表。"""
    samples = []
    if not DATA_IMAGES.exists():
        raise FileNotFoundError(f"找不到图片目录：{DATA_IMAGES}")

    for folder in sorted(p for p in DATA_IMAGES.iterdir() if p.is_dir()):
        key = patient_key(folder.name)
        ann_dir = DATA_ANNOTATIONS / folder.name
        for img in sorted(folder.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            xml = None
            if ann_dir.exists():
                cand = ann_dir / (img.stem + ".xml")
                if cand.exists():
                    xml = cand
            samples.append({
                "patient": key,
                "folder": folder.name,
                "image": str(img),
                "xml": str(xml) if xml else None,
            })
    return samples


def split_patients(samples, ratios, seed, names):
    """按患者划分，返回 {split: [samples]} 与划分记录。"""
    import random
    patients = sorted({s["patient"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    groups = {names[0]: [], names[1]: [], names[2]: []}
    for s in samples:
        p = s["patient"]
        if p in train_p:
            groups[names[0]].append(s)
        elif p in val_p:
            groups[names[1]].append(s)
        else:
            groups[names[2]].append(s)

    record = {
        "train_patients": sorted(train_p),
        "val_patients": sorted(val_p),
        "test_patients": sorted(test_p),
    }
    return groups, record


def _place(src: Path, dst: Path, link_mode: str):
    """把图片放到目标位置，支持 copy/symlink/hardlink（失败回退 copy）。"""
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if link_mode == "symlink":
            dst.symlink_to(src)
        elif link_mode == "hardlink":
            import os
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(class_map, names, ratios=(0.7, 0.15, 0.15), seed=42,
          output_dir=None, link_mode="hardlink", include_background=True,
          verbose=True):
    """构建 YOLO 数据集，返回输出目录路径。"""
    split_names = ("train", "val", "test")
    output_dir = Path(output_dir) if output_dir else (DATASETS_DIR / "oral_fluorescence")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    samples = discover_samples()
    groups, record = split_patients(samples, ratios, seed, split_names)

    stats = {"skipped_background": 0, "objects": {}}
    manifest = []

    for split in split_names:
        img_dir = ensure_dir(output_dir / "images" / split)
        lbl_dir = ensure_dir(output_dir / "labels" / split)
        for s in groups[split]:
            src_img = Path(s["image"])
            stem = src_img.stem
            # 文件名加患者前缀，避免不同患者目录下同名文件冲突
            safe_stem = f"{s['folder']}__{stem}".replace(" ", "_")
            dst_img = img_dir / (safe_stem + src_img.suffix.lower())
            _place(src_img, dst_img, link_mode)

            lines = []
            if s["xml"]:
                ann = parse_voc(s["xml"])
                lines = to_yolo_lines(ann, class_map)
                for cls in [b.cls_name for b in ann.boxes]:
                    stats["objects"][cls] = stats["objects"].get(cls, 0) + 1

            if not lines and not include_background:
                # 无目标且不保留背景 -> 跳过
                dst_img.unlink(missing_ok=True)
                stats["skipped_background"] += 1
                continue

            (lbl_dir / (safe_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            manifest.append({
                "split": split, "patient": s["patient"], "folder": s["folder"],
                "image": str(dst_img), "n_objects": len(lines),
            })

    # 写 data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    data_yaml = (
        f"path: {output_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    (output_dir / "split_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"数据集输出：{output_dir}")
        for split in split_names:
            n_img = len(list((output_dir / 'images' / split).iterdir()))
            n_lbl = len(list((output_dir / 'labels' / split).iterdir()))
            print(f"  {split}: {n_img} 图 / {n_lbl} 标签, 患者数={len(record[split + '_patients'])}")
        print(f"  类别实例数：{stats['objects']}")
        if include_background:
            n_bg = sum(1 for m in manifest if m['n_objects'] == 0)
            print(f"  背景图（无目标）：{n_bg}")

    return output_dir
