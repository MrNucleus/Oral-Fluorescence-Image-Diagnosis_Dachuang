"""PyCharm live console for local ablation validation results; does not train."""
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    print("Ablation monitor: train/validation only. Ctrl+C to close.", flush=True)
    seen = set()
    print("Waiting for completed validation epochs. Existing pilot may still be training.", flush=True)
    while True:
        for path in sorted((ROOT / "reports/ablation").glob("*/validation_summary.json")):
            for row in json.loads(path.read_text(encoding="utf-8")):
                key = (str(path), row["variant"])
                if key in seen:
                    continue
                seen.add(key)
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        time.sleep(3)
