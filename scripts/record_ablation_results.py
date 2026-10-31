"""Archive the two completed pilot batches with provenance and SHA-256 checks."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record():
    from training_dashboard import VARIANTS

    sources = [ROOT / "reports/ablation" / stamp / "validation_summary.json"
               for stamp in ("20261028_134000", "20261028_162427")]
    rows = [row for source in sources for row in json.loads(source.read_text(encoding="utf-8"))]
    if sorted(r["variant"] for r in rows) != list(range(8)):
        raise ValueError("Expected exactly eight completed variants")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    destination = ROOT / "reports/ablation/records" / now.strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    dataset_hashes = set()
    for row in sorted(rows, key=lambda r: r["variant"]):
        index = row["variant"]
        if (tuple(row[k] for k in ("model", "attention", "loss")) != VARIANTS[index]
                or not row["pilot"] or row["imgsz"] != 320 or row["seed"] != 42):
            raise ValueError("Pilot identity mismatch")
        run = (ROOT / row["run"]).resolve()
        if not run.is_relative_to(ROOT):
            raise ValueError("Run path escapes workspace")
        metadata = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        selected = next(h for h in history if h["epoch"] == row["best_epoch"])
        if selected["val_metrics"] != row["validation"]:
            raise ValueError("Summary differs from recorded training history")
        dataset_hashes.add(metadata["dataset"]["dataset_json_sha256"])
        artifacts = {}
        for name in ("run_config.json", "history.json", "training_config.yaml", "best.pt", "last.pt"):
            path = run / name
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts[name] = {"path": path.relative_to(ROOT).as_posix(),
                               "bytes": path.stat().st_size, "sha256": digest(path)}
            if path.suffix != ".pt":
                target = destination / "evidence" / f"variant_{index}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        results.append({**row, "train_loss": selected["train_loss"],
                        "train_images": selected["train_images"], "configuration": metadata["config"],
                        "loss_configuration": metadata["loss"], "augmentation": metadata["augmentation"],
                        "dataset": metadata["dataset"], "training_source_hashes": metadata["source_hashes"],
                        "artifacts": artifacts})
    if len(dataset_hashes) != 1:
        raise ValueError("Pilot dataset fingerprints differ")
    log = ROOT / "reports/ablation/restart_20261028_5_to_7/training.stdout.log"
    points = []
    current = None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("Starting variant "):
            current = int(line.split()[2].rstrip(":"))
        elif line.startswith("{"):
            item = json.loads(line)
            if item.get("phase") == "train":
                points.append({"variant": current, **item})
    payload = {"recorded_at": now.isoformat(), "status": "all_eight_pilot_variants_completed",
               "scope": "image-level train/validation pilot only; no test performance computed",
               "dataset_sha256": next(iter(dataset_hashes)), "results": results,
               "batch_progress": points,
               "limitations": ["One epoch at 320 pixels; not a formal architecture comparison.",
                               "Source groups are not confirmed independent patients.",
                               "Batch logs exist only for restarted variants 5-7.",
                               "Training losses differ by objective and are not directly comparable."]}
    (destination / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(log, destination / "evidence/training.stdout.log")
    for source in sources:
        shutil.copy2(source, destination / "evidence" / f"summary_{source.parent.name}.json")
    lines = ["# 架构与损失消融试跑记录", "", f"记录时间：{now.isoformat()}", "",
             "8/8组完成；输入320、每组1轮、batch 4、seed 42、AdamW、学习率0.001。",
             "训练集893张，验证集112张。选模为无权重、无平滑验证交叉熵；阈值0.5。",
             "来源组未核验为独立患者；本记录没有测试集性能。", "",
             "| 编号 | 架构 | 新增注意力 | 损失 | 训练损失 | 验证CE | AUROC | 敏感度 | 特异度 | 平衡准确率 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        v = row["validation"]
        lines.append(f"| {row['variant']} | {row['model']} | {row['attention']} | {row['loss']} | "
                     + " | ".join(f"{x:.4f}" for x in [row["train_loss"], v["loss"], v["auroc"],
                                                            v["sensitivity"], v["specificity"], v["balanced_accuracy"]]) + " |")
    lines += ["", "## 记录边界", "",
              "- 这是流程试跑，不能按单轮结果宣称某架构或注意力有效。",
              "- 第0、4、6组在0.5阈值下将验证图像全部判为阳性候选，应保留该结果。",
              "- 第5组首次未完成且无检查点；重启后完成第5至7组，前五组保持原结果。",
              "- EfficientNet自身含SE；表中的none仅指未额外添加注意力。",
              "- 不同训练损失函数的数值不直接比较，验证CE统一计算。",
              "- results.json保存完整指标、混淆矩阵、配置、历史及权重SHA-256；权重留在原目录。",
              "- evidence/保存配置、训练历史和日志的副本；没有复制逐图记录或原始数据。", ""]
    (destination / "记录报告.md").write_text("\n".join(lines), encoding="utf-8")
    inventory = [{"path": p.relative_to(destination).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(destination.rglob("*")) if p.is_file()]
    (destination / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    record()
