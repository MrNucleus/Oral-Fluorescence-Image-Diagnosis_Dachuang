"""Run controlled whole-image ablations on frozen train/val; never evaluates test."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments/oral_v7_resnet18_baseline_20261026/code"
sys.path.insert(0, str(CODE / "src"))
from oral_fluorescence_detection.image_classification import train

VARIANTS = [
    ("resnet50", "none", "ce"),
    ("efficientnet_b3", "none", "ce"),
    ("densenet121", "none", "ce"),
    ("resnet50", "se", "ce"),
    ("resnet50", "cbam", "ce"),
    ("resnet50", "none", "weighted_ce"),
    ("resnet50", "none", "label_smoothing"),
    ("resnet50", "none", "focal"),
]

def reuse_baseline(source, data, cache_dir, seed, workers):
    """Reuse a completed formal baseline only when its inputs match."""
    source = Path(source).resolve()
    cfg = yaml.safe_load((source / "variant_0.yaml").read_text(encoding="utf-8"))
    expected = dict(model="resnet50", attention="none", loss="ce", epochs=100,
                    imgsz=640, batch=4, seed=seed, workers=workers, lr0=.001, patience=20)
    if any(cfg.get(k) != value for k, value in expected.items()):
        raise ValueError("Existing baseline configuration differs")
    if Path(cfg["cache_dir"]).resolve() != Path(cache_dir).resolve():
        raise ValueError("Existing baseline uses a different cache")
    run = CODE / "outputs/image_classifier_runs" / cfg["name"]
    meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    state = json.loads((run / "continue_state.json").read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (state["status"] != "complete" or not (run / "best.pt").is_file()
            or meta["dataset"]["dataset_json_sha256"] != digest(data)
            or meta["preprocessing"] != dict(version="resize-before-augmentation-v1",
                cache_manifest_sha256=digest(Path(cache_dir) / "manifest.json"))
            or meta["source_hashes"]["online_augmentation.py"] != digest(
                CODE / "src/oral_fluorescence_detection/online_augmentation.py")):
        raise ValueError("Existing baseline is incomplete or has different data/preprocessing")
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    selected = min(history, key=lambda row: row["val_metrics"]["loss"])
    result = dict(variant=0, model="resnet50", attention="none", loss="ce", seed=seed,
                  pilot=False, imgsz=640, best_epoch=selected["epoch"],
                  validation=selected["val_metrics"], run=str(run.relative_to(ROOT)),
                  reused_from=str(source.relative_to(ROOT)))
    return cfg, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pilot", action="store_true", help="1 epoch, 320 full-field input; software pilot only")
    parser.add_argument("--variants", nargs="*", type=int, help="Variant indices 0..7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", help="Optional versioned cache; changes training preprocessing")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--reuse-baseline", help="Reuse compatible completed cache baseline; run remaining seven")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    indices = args.variants if args.variants is not None else list(range(len(VARIANTS)))
    if any(i < 0 or i >= len(VARIANTS) for i in indices):
        parser.error("variant index must be 0..7")
    baseline = None
    if args.reuse_baseline:
        if args.pilot or not args.cache_dir or args.variants is not None:
            parser.error("--reuse-baseline requires cache formal all-variant mode")
        baseline = reuse_baseline(args.reuse_baseline, args.data, args.cache_dir, args.seed, args.workers)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    report = ROOT / "reports/ablation" / stamp
    report.mkdir(parents=True)
    (report / "plan.json").write_text(json.dumps(dict(variants=indices, cache=bool(args.cache_dir),
        reused_baseline=args.reuse_baseline)), encoding="utf-8")
    print(f"Experiment report: {report}", flush=True)
    if baseline:
        cfg, result = baseline
        (report / "variant_0.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        results.append(result)
        (report / "validation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Reusing completed variant 0 from {args.reuse_baseline}", flush=True)
    for i in indices:
        if baseline and i == 0:
            continue
        model, attention, loss = VARIANTS[i]
        cfg = dict(model=model, attention=attention, loss=loss,
                   label_smoothing=.1, focal_gamma=2., seed=args.seed,
                   epochs=1 if args.pilot else 100, imgsz=320 if args.pilot else 640,
                   batch=4, workers=args.workers, lr0=.001, patience=20,
                   name=f"ablation_{stamp}_{i}_{model}_{attention}_{loss}" + ("_cache_v1" if args.cache_dir else ""))
        if args.cache_dir:
            cfg["cache_dir"] = str(Path(args.cache_dir).resolve())
        (report / f"variant_{i}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        print(f"Starting variant {i}: {model}/{attention}/{loss}", flush=True)
        best = train(args.data, overrides=cfg)
        history = json.loads((best.parent / "history.json").read_text(encoding="utf-8"))
        selected = min(history, key=lambda row: row["val_metrics"]["loss"])
        results.append(dict(variant=i, model=model, attention=attention, loss=loss,
                            seed=args.seed, pilot=args.pilot, imgsz=cfg["imgsz"],
                            best_epoch=selected["epoch"], validation=selected["val_metrics"],
                            run=str(best.parent.relative_to(ROOT))))
        (report / "validation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Validation-only results: {report}", flush=True)

if __name__ == "__main__":
    main()
