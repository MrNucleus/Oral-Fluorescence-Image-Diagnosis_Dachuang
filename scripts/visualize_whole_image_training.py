"""Render a local, self-refreshing dashboard for whole-image classifier training."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
from datetime import datetime
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def safe_metric(record, name):
    value = record.get("val_metrics", {}).get(name)
    return None if value is None else float(value)


def best_record(history):
    if not history:
        return None
    return min(history, key=lambda item: float(item["val_metrics"]["loss"]))


def render_figure(history, output_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#f7f7f5",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d5d8dc",
        "axes.labelcolor": "#253238",
        "axes.titlecolor": "#17252b",
        "xtick.color": "#52636b",
        "ytick.color": "#52636b",
        "grid.color": "#e5e8ea",
        "font.size": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Whole-image training monitor", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    if history:
        epochs = [int(item["epoch"]) for item in history]
        train_loss = [float(item["train_loss"]) for item in history]
        val_loss = [float(item["val_metrics"]["loss"]) for item in history]
        ax.plot(epochs, train_loss, color="#19647e", linewidth=2, marker="o", markersize=3, label="Train")
        ax.plot(epochs, val_loss, color="#c84c3a", linewidth=2, marker="o", markersize=3, label="Validation")
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "Waiting for the first completed epoch", ha="center", va="center",
                transform=ax.transAxes, color="#6b7c84")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.grid(True, alpha=.7)

    ax = axes[0, 1]
    if history:
        ax.plot(epochs, [safe_metric(item, "auroc") for item in history],
                color="#2f6b4f", linewidth=2, marker="o", markersize=3, label="AUROC")
        ax.plot(epochs, [safe_metric(item, "balanced_accuracy") for item in history],
                color="#b07123", linewidth=2, marker="o", markersize=3, label="Balanced accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No validation metrics yet", ha="center", va="center",
                transform=ax.transAxes, color="#6b7c84")
    ax.set_title("Validation discrimination")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=.7)

    ax = axes[1, 0]
    if history:
        ax.plot(epochs, [safe_metric(item, "sensitivity") for item in history],
                color="#a83e52", linewidth=2, marker="o", markersize=3, label="Sensitivity")
        ax.plot(epochs, [safe_metric(item, "specificity") for item in history],
                color="#3d6ea8", linewidth=2, marker="o", markersize=3, label="Specificity")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No sensitivity or specificity yet", ha="center", va="center",
                transform=ax.transAxes, color="#6b7c84")
    ax.set_title("Validation operating point at 0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=.7)

    ax = axes[1, 1]
    ax.axis("off")
    if history:
        latest = history[-1]
        best = best_record(history)
        lines = [
            ("Latest epoch", str(latest["epoch"])),
            ("Latest validation loss", f'{float(latest["val_metrics"]["loss"]):.4f}'),
            ("Latest AUROC", f'{safe_metric(latest, "auroc"):.4f}'),
            ("Latest balanced accuracy", f'{safe_metric(latest, "balanced_accuracy"):.4f}'),
            ("Best epoch by validation loss", str(best["epoch"])),
            ("Best validation loss", f'{float(best["val_metrics"]["loss"]):.4f}'),
        ]
        for row, (label, value) in enumerate(lines):
            y = .84 - row * .13
            ax.text(.04, y, label, transform=ax.transAxes, color="#5f7078", fontsize=10)
            ax.text(.96, y, value, transform=ax.transAxes, color="#17252b",
                    fontsize=13, fontweight="bold", ha="right")
    else:
        ax.text(.5, .5, "The dashboard will populate after epoch 1",
                ha="center", va="center", transform=ax.transAxes, color="#6b7c84")
    ax.set_title("Run summary")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def table_rows(history):
    rows = []
    for item in history[-12:]:
        metrics = item["val_metrics"]
        values = [
            item["epoch"],
            f'{float(item["train_loss"]):.4f}',
            f'{float(metrics["loss"]):.4f}',
            f'{safe_metric(item, "auroc"):.4f}',
            f'{safe_metric(item, "balanced_accuracy"):.4f}',
            f'{safe_metric(item, "sensitivity"):.4f}',
            f'{safe_metric(item, "specificity"):.4f}',
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    return "\n".join(rows)


def render_html(run_dir: Path, history, config, image_b64: str, interval: int):
    latest = history[-1] if history else None
    best = best_record(history)
    metrics = latest["val_metrics"] if latest else {}
    cards = [
        ("Epoch", latest["epoch"] if latest else "waiting"),
        ("Train loss", f'{float(latest["train_loss"]):.4f}' if latest else "-"),
        ("Validation loss", f'{float(metrics.get("loss")):.4f}' if latest else "-"),
        ("Validation AUROC", f'{float(metrics.get("auroc")):.4f}'
         if metrics.get("auroc") is not None else "-"),
        ("Balanced accuracy", f'{float(metrics.get("balanced_accuracy")):.4f}'
         if metrics.get("balanced_accuracy") is not None else "-"),
        ("Best epoch", best["epoch"] if best else "-"),
    ]
    card_html = "\n".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in cards
    )
    model = config.get("config", {}).get("model", "unknown") if config else "unknown"
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{interval}">
<title>Whole-image training monitor</title>
<style>
  :root {{ color-scheme: light; --ink:#17252b; --muted:#64757d; --line:#dce1e3;
          --paper:#f7f7f5; --panel:#ffffff; --accent:#19647e; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
          font-family:"Segoe UI","Microsoft YaHei",sans-serif; }}
  main {{ max-width:1440px; margin:0 auto; padding:28px; }}
  header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end;
            border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:20px; }}
  h1 {{ margin:0; font-size:28px; letter-spacing:0; }}
  .meta {{ color:var(--muted); font-size:13px; text-align:right; line-height:1.7; }}
  .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin-bottom:20px; }}
  .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
             padding:14px 16px; min-width:0; }}
  .metric span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:8px; }}
  .metric strong {{ display:block; font-size:23px; white-space:nowrap; }}
  .figure {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
  .figure img {{ display:block; width:100%; height:auto; }}
  .table-wrap {{ margin-top:20px; background:var(--panel); border:1px solid var(--line);
                 border-radius:8px; overflow:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:10px 14px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; background:#f2f4f4; }}
  tr:last-child td {{ border-bottom:0; }}
  @media (max-width:900px) {{
    main {{ padding:16px; }}
    header {{ align-items:flex-start; flex-direction:column; }}
    .meta {{ text-align:left; }}
    .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    h1 {{ font-size:22px; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Training monitor</h1>
      <div class="meta">{html.escape(run_dir.name)}</div>
    </div>
    <div class="meta">Model: {html.escape(str(model))}<br>
      Updated: {html.escape(updated)}<br>Auto refresh: {interval}s</div>
  </header>
  <section class="metrics">{card_html}</section>
  <section class="figure"><img alt="Training curves" src="data:image/png;base64,{image_b64}"></section>
  <section class="table-wrap">
    <table>
      <thead><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th><th>AUROC</th>
      <th>Balanced acc.</th><th>Sensitivity</th><th>Specificity</th></tr></thead>
      <tbody>{table_rows(history)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    history_path = run_dir / "history.json"
    config_path = run_dir / "run_config.json"
    output = (args.output or run_dir / "dashboard").resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "training_curves.png"
    html_path = output / "index.html"

    while True:
        history = read_json(history_path)
        config = read_json(config_path)
        history = history if isinstance(history, list) else []
        image_b64 = render_figure(history, figure_path)
        html_path.write_text(
            render_html(run_dir, history, config, image_b64, max(1, int(args.interval))),
            encoding="utf-8",
        )
        print(json.dumps({
            "updated": datetime.now().isoformat(timespec="seconds"),
            "epoch": history[-1]["epoch"] if history else None,
            "dashboard": str(html_path),
        }, ensure_ascii=False), flush=True)
        if args.once:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
