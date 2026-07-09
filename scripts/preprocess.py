import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from glob import glob
from tqdm import tqdm
from src.data.preprocessing import quality_score, correct_illumination


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--quality-threshold", type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    print(f"Found {len(files)} images")

    for img_path in tqdm(files, desc="Preprocessing"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = quality_score(img_rgb)
        if q["is_blurry"] or q["is_overexposed"] or q["is_underexposed"]:
            print(f"  Low quality: {os.path.basename(img_path)} ({q['sharpness']:.0f})")
        corrected = correct_illumination(img_rgb)
        rel = os.path.relpath(img_path, args.input_dir)
        out = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
