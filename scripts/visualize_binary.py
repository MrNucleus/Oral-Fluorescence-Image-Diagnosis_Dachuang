"""Generate training-process and test-result visualizations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_binary import (  # noqa: E402
    build_model,
    load_config,
    predict_loader,
    save_json,
)
from src.classification.data import (  # noqa: E402
    BinaryImageDataset,
    build_image_transforms,
)
from src.classification.metrics import (  # noqa: E402
    evaluate_predictions,
    select_operating_threshold,
)
from src.classification.visualize import (  # noqa: E402
    animate_training_process,
    load_history,
    plot_test_overview,
    plot_training_curves,
    write_dashboard,
)
from src.utils.paths import PROJECT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize binary experiment")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "binary_annotated_seed43.yaml"),
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest_path = PROJECT_ROOT / config["manifest"]
    experiment_dir = PROJECT_ROOT / config["experiment_dir"]
    history = load_history(experiment_dir / "history.csv")
    summary = __import__("json").loads(
        (experiment_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    best_epoch = int(summary["best_epoch"])

    training_path = plot_training_curves(
        history, experiment_dir / "training_curves.png", best_epoch=best_epoch
    )
    animation_path = animate_training_process(
        history, experiment_dir / "training_process.gif", best_epoch=best_epoch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    val_dataset = BinaryImageDataset(
        manifest_path, "val", build_image_transforms(image_size, train=False)
    )
    test_dataset = BinaryImageDataset(
        manifest_path, "test", build_image_transforms(image_size, train=False)
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    model = build_model(pretrained=False, dropout=float(config["model"]["dropout"]))
    checkpoint = torch.load(
        experiment_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_labels, val_probabilities, val_patient_indices = predict_loader(
        model, val_loader, device, use_amp
    )
    val_result = evaluate_predictions(
        val_labels,
        val_probabilities,
        val_patient_indices,
        val_dataset.patient_ids,
        threshold=0.5,
        bootstrap=0,
    )
    threshold_result = select_operating_threshold(
        [row["label"] for row in val_result["patients"]],
        [row["mean_probability"] for row in val_result["patients"]],
    )
    selected_threshold = threshold_result["selected"]["threshold"]

    test_labels, test_probabilities, test_patient_indices = predict_loader(
        model, test_loader, device, use_amp
    )
    test_metrics = evaluate_predictions(
        test_labels,
        test_probabilities,
        test_patient_indices,
        test_dataset.patient_ids,
        threshold=selected_threshold,
        bootstrap=int(config["evaluation"]["bootstrap"]),
    )
    overview_path = plot_test_overview(
        test_labels,
        test_probabilities,
        test_metrics,
        selected_threshold,
        experiment_dir / "test_overview.png",
    )

    final = {
        "experiment": config["experiment_dir"],
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "threshold_selection": threshold_result,
        "test": test_metrics,
    }
    save_json(experiment_dir / "visualized_metrics.json", final)
    dashboard_path = write_dashboard(
        experiment_dir / "dashboard.html",
        Path(config["experiment_dir"]).name,
        final,
        {
            "training": training_path,
            "overview": overview_path,
            "animation": animation_path,
        },
    )
    print(f"training_curves={training_path}")
    print(f"training_animation={animation_path}")
    print(f"test_overview={overview_path}")
    print(f"dashboard={dashboard_path}")


if __name__ == "__main__":
    main()

