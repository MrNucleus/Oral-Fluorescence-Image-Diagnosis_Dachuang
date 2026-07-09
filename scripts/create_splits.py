"""数据集划分脚本：数据到手后运行此脚本生成 train/val/test 划分文件"""
import os
import random
from glob import glob
from collections import defaultdict


def main():
    base_dir = "data/raw"
    split_dir = "data/splits"
    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    random.seed(42)

    # 自动检测类别（子目录名 = 类别名）
    class_names = sorted([d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d))])
    if not class_names:
        print("No class directories found in data/raw/")
        print("Please organize data as: data/raw/<class_name>/<images>")
        return

    print(f"Detected classes: {class_names}")

    # 收集每个类别的所有图像路径
    class_samples = defaultdict(list)
    for cls in class_names:
        cls_dir = os.path.join(base_dir, cls)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            class_samples[cls].extend(
                glob(os.path.join(cls_dir, ext), recursive=False)
            )
        print(f"  {cls}: {len(class_samples[cls])} samples")

    # 划分并写入
    os.makedirs(split_dir, exist_ok=True)

    splits = {"train.txt": [], "val.txt": [], "test.txt": []}
    for cls in class_names:
        samples = class_samples[cls]
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, sample in enumerate(samples):
            rel_path = os.path.relpath(sample, base_dir)
            if i < n_train:
                splits["train.txt"].append(f"{rel_path} {cls}")
            elif i < n_train + n_val:
                splits["val.txt"].append(f"{rel_path} {cls}")
            else:
                splits["test.txt"].append(f"{rel_path} {cls}")

    for filename, lines in splits.items():
        path = os.path.join(split_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"  {filename}: {len(lines)} samples")

    print("Done! Splits saved to data/splits/")


if __name__ == "__main__":
    main()
