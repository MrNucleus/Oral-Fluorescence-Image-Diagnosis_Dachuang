"""Read-only, loopback-only dashboard for the current restarted pilot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/ablation"
RESTART = REPORTS / "restart_20261028_5_to_7"
VARIANTS = [
    ("resnet50", "none", "ce"), ("efficientnet_b3", "none", "ce"),
    ("densenet121", "none", "ce"), ("resnet50", "se", "ce"),
    ("resnet50", "cbam", "ce"), ("resnet50", "none", "weighted_ce"),
    ("resnet50", "none", "label_smoothing"), ("resnet50", "none", "focal"),
]


def read_pilot_status():
    completed = {}
    # Keep only this ResNet50-based pilot cohort; do not mix formal or historical runs.
    for folder in sorted(REPORTS.iterdir()):
        if not folder.name.isdigit() and "_" not in folder.name:
            continue
        if folder.name < "20261028_134000":
            continue
        path = folder / "validation_summary.json"
        if not path.is_file():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # Writer may be replacing a partially written summary.
        for row in rows:
            i = row.get("variant", -1)
            if (isinstance(i, int) and 0 <= i < 8 and row.get("pilot") is True
                    and row.get("imgsz") == 320 and row.get("seed") == 42
                    and tuple(row.get(k) for k in ("model", "attention", "loss")) == VARIANTS[i]):
                completed[i] = {**row, "source": str(path.relative_to(ROOT))}

    log = RESTART / "training.stdout.log"
    errors = RESTART / "training.stderr.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    current = None
    points = []
    phase = "等待日志"
    for line in text.splitlines():
        if line.startswith("Starting variant "):
            try:
                current = int(line.split()[2].rstrip(":"))
                points = []
                phase = "准备数据与模型"
            except ValueError:
                pass
        elif line.startswith("{"):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("phase") == "train":
                points.append(item)
                phase = "训练中" if item["batch"] < item["batches"] else "训练批次结束，等待验证"
            elif "val_metrics" in item:
                phase = "该组验证完成"
    if current in completed:
        phase = "该组验证完成"
    if len(completed) == 8:
        phase = "8组试跑全部完成"
    error_text = errors.read_text(encoding="utf-8", errors="replace") if errors.exists() else ""
    if "Traceback (most recent call last)" in error_text:
        phase = "检测到错误，请查看错误日志"
    now = datetime.now(timezone.utc).timestamp()
    updated = log.stat().st_mtime if log.exists() else None
    return {
        "phase": phase, "current": current, "points": points,
        "completed": [completed[i] for i in sorted(completed)],
        "variants": [dict(index=i, model=m, attention=a, loss=l) for i, (m, a, l) in enumerate(VARIANTS)],
        "log_age_seconds": round(now - updated) if updated else None,
        "updated": datetime.fromtimestamp(updated).astimezone().isoformat(timespec="seconds") if updated else None,
        "tail": text.splitlines()[-10:], "error_tail": error_text.splitlines()[-8:],
        "note": "320像素、每组1轮、seed 42；仅训练与验证，不计算测试集性能。日志更新不等同于进程存活检测。",
    }


FORMAL_REPORT = None
REPEAT_SINCE = None


def read_status():
    if REPEAT_SINCE is not None:
        return read_repeat_status()
    if FORMAL_REPORT is None:
        return read_pilot_status()
    return read_formal_status(ROOT / FORMAL_REPORT)


def read_formal_status(report):
    plan_path = report / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    planned = plan.get("variants", list(range(len(VARIANTS))))
    title = "缓存优化训练" if plan.get("cache") else "正式训练"
    completed, runs = [], []
    updated = None
    for i, (model, attention, loss) in enumerate(VARIANTS):
        if i not in planned:
            continue
        config = report / f"variant_{i}.yaml"
        if not config.is_file():
            continue
        import yaml
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (cfg["imgsz"] != 640 or cfg["epochs"] != 100
                or (cfg["model"], cfg["attention"], cfg["loss"]) != (model, attention, loss)):
            raise ValueError("Formal configuration identity mismatch")
        run = ROOT / "experiments/oral_v7_resnet18_baseline_20261026/code/outputs/image_classifier_runs" / cfg["name"]
        def read_json(name, fallback):
            try:
                return json.loads((run / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return fallback
        history = read_json("history.json", [])
        state = read_json("continue_state.json", {})
        entry = {"index": i, "history": history, "state": state, "config": cfg}
        if history:
            entry["best"] = min(history, key=lambda row: row["val_metrics"]["loss"])
            updated = max(updated or 0, (run / "history.json").stat().st_mtime)
        else:
            updated = max(updated or 0, config.stat().st_mtime)
        runs.append(entry)
        if state.get("status") == "complete" and history:
            completed.append({"variant": i, "model": model, "attention": attention, "loss": loss,
                              "validation": entry["best"]["val_metrics"], "best_epoch": entry["best"]["epoch"]})
    active = max(runs, key=lambda entry: (report / f"variant_{entry['index']}.yaml").stat().st_mtime) if runs else None
    history = active["history"] if active else []
    last = history[-1] if history else None
    current = active["index"] if active else None
    phase = f"{title} · 等待首轮验证结果"
    if last:
        phase = f"{title} · 已完成第{last['epoch']}轮，等待下一轮结果"
    if len(completed) == len(planned):
        phase = f"{len(planned)}组{title}全部完成"
    now = datetime.now(timezone.utc).timestamp()
    return {"mode": "formal", "phase": phase, "current": current,
            "points": [], "history": history, "completed": completed,
            "variants": [dict(index=i, model=m, attention=a, loss=l) for i, (m, a, l) in enumerate(VARIANTS) if i in planned],
            "best": active.get("best") if active else None,
            "stale": active["state"].get("stale") if active else None,
            "epoch": last["epoch"] if last else 0, "max_epochs": 100,
            "updated": datetime.fromtimestamp(updated).astimezone().isoformat(timespec="seconds") if updated else None,
            "log_age_seconds": round(now-updated) if updated else None,
            "tail": [json.dumps(row, ensure_ascii=False) for row in history[-4:]], "error_tail": [],
            "note": "640像素、batch 4、最多100轮、早停20轮；只读取每轮训练历史。逐批日志仍在PyCharm Run窗口；未计算测试集性能。"}


def read_repeat_status():
    import yaml
    variants = [dict(index=i, model='resnet50', attention=attention, loss='ce', seed=seed)
                for i, (seed, attention) in enumerate(((43, 'none'), (43, 'se'), (44, 'se'), (44, 'none')))]
    blocks = {}
    for report in sorted(REPORTS.iterdir()):
        if not report.is_dir() or report.name < REPEAT_SINCE or not report.name[:8].isdigit():
            continue
        configs = list(report.glob('variant_*.yaml'))
        if not configs:
            continue
        cfg = yaml.safe_load(configs[0].read_text(encoding='utf-8'))
        seed = cfg.get('seed')
        if seed not in (43, 44) or seed in blocks:
            continue
        plan = json.loads((report / 'plan.json').read_text(encoding='utf-8'))
        if plan.get('variants') != ([0, 3] if seed == 43 else [3, 0]):
            continue
        blocks[seed] = read_formal_status(report)
    if not blocks:
        raise ValueError('No matching repeat blocks found')
    seed = max(blocks)
    status = blocks[seed]
    slot = lambda s, index: next(v['index'] for v in variants
                                if v['seed'] == s and v['attention'] == ('none' if index == 0 else 'se'))
    completed = [dict(row, variant=slot(s, row['variant']), seed=s)
                 for s, block in blocks.items() for row in block['completed']]
    return dict(status, variants=variants, completed=completed,
                current=slot(seed, status['current']) if status['current'] is not None else None,
                phase=f"seed {seed} · {status['phase']}" if len(completed) < 4 else '4次复验全部完成',
                title='基线与SE · 多种子复验',
                note='seed43：基线→SE；seed44：SE→基线。四次串行训练，每轮结束更新曲线；640、batch4、FP32。未计算测试性能。')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            payload = json.dumps(read_status(), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif self.path in ("/", "/index.html"):
            payload = (ROOT / "scripts/training_dashboard.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--formal-report", help="Workspace-relative formal report directory")
    parser.add_argument("--repeat-since", help="First report folder name for the seed43/44 repeat queue")
    args = parser.parse_args()
    global FORMAL_REPORT, REPEAT_SINCE
    if args.repeat_since:
        if not (REPORTS / args.repeat_since).is_dir() or Path(args.repeat_since).name != args.repeat_since:
            parser.error('Repeat start report must be an existing report folder name')
        REPEAT_SINCE = args.repeat_since
    if args.formal_report:
        target = (ROOT / args.formal_report).resolve()
        if not target.is_relative_to(REPORTS) or not target.is_dir():
            parser.error("Formal report must be an existing folder within reports/ablation")
        FORMAL_REPORT = target.relative_to(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Training dashboard: {url}", flush=True)
    output = REPORTS / REPEAT_SINCE if REPEAT_SINCE else ROOT / FORMAL_REPORT if FORMAL_REPORT else RESTART
    output.mkdir(parents=True, exist_ok=True)
    (output / "dashboard_url.json").write_text(json.dumps({"url": url}), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
