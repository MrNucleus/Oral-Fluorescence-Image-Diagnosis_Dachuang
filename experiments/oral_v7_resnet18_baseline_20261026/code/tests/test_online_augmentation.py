"""Whole-image geometry, scalar labels, split isolation and classification smoke test."""

import hashlib
import inspect
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")

from oral_fluorescence_detection.online_augmentation import (  # noqa: E402
    AugmentationPolicy,
    ImageClassificationDataset,
    augment_image,
    letterbox_full_image,
    validate_handoff,
    verify_checksums,
)


@pytest.fixture
def synthetic_handoff(tmp_path):
    """Only complete images and image-level JSON; deliberately no XML or YOLO TXT."""
    root = tmp_path / "input"
    records = []
    for split in ("train", "val", "test"):
        for modality in ("standard", "fluorescence"):
            for label in (0, 1):
                stem = f"synthetic_{split}_{modality}_{label}"
                relative = f"{split}/{modality}/images/{label}_{'normal' if label == 0 else 'lesion'}/{stem}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((48, 80, 3), [20, 60, 100], dtype=np.uint8)
                image[5:35, 8:38] = [50 + label * 40, 80, 130]
                Image.fromarray(image).save(path)
                records.append(dict(
                    image_id=stem, group_id=f"group_{split}", split=split,
                    modality=modality, label=label, image_relpath=relative,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    label_basis="synthetic software fixture; not a clinical label"))
    path = root / "dataset.json"
    path.write_text(json.dumps(dict(
        schema_version=1, task="image_classification",
        class_names=["normal_candidate", "lesion_candidate"], records=records)),
        encoding="utf-8")
    return path


def test_flip_operates_on_complete_rgb_image():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[10:30, 5:25] = [200, 30, 10]
    policy = AugmentationPolicy(horizontal_flip_probability=1, rotation_probability=0)
    result, replay = augment_image(image, "standard", policy)
    np.testing.assert_array_equal(result, image[:, ::-1])
    assert replay["flip"]
    assert "boxes" not in inspect.signature(augment_image).parameters


def test_rotation_and_letterbox_keep_all_four_image_edges():
    image = np.full((70, 110, 3), 20, dtype=np.uint8)
    # Distinct corners make cropping visible independently of any annotation.
    image[:12, :15] = [255, 0, 0]
    image[:12, -15:] = [0, 255, 0]
    image[-12:, :15] = [0, 0, 255]
    image[-12:, -15:] = [255, 255, 0]
    policy = AugmentationPolicy(horizontal_flip_probability=0, rotation_probability=1)
    rotated, replay = augment_image(image, "fluorescence", policy, rng=random.Random(4))
    assert rotated.shape[0] > image.shape[0] and rotated.shape[1] > image.shape[1]
    assert replay["rotation"] != 0
    squared = letterbox_full_image(rotated, 128)
    assert squared.shape == (128, 128, 3) and squared.dtype == np.uint8
    r, g, b = np.moveaxis(squared, -1, 0)
    for marker in ((r > 220) & (g < 30) & (b < 30),
                   (g > 220) & (r < 30) & (b < 30),
                   (b > 220) & (r < 30) & (g < 30),
                   (r > 220) & (g > 220) & (b < 30)):
        assert np.count_nonzero(marker) > 50


@pytest.mark.parametrize("mode", ["standard", "fluorescence"])
def test_eval_and_baseline_are_exact_and_seed_replays(mode):
    image = np.arange(18 * 32 * 3, dtype=np.uint8).reshape(18, 32, 3)
    for policy, training in ((AugmentationPolicy(), False),
                             (AugmentationPolicy(profile="baseline"), True)):
        result, _ = augment_image(image, mode, policy, training=training)
        np.testing.assert_array_equal(result, image)
    one, replay_one = augment_image(image, mode, AugmentationPolicy(), rng=random.Random(2))
    two, replay_two = augment_image(image, mode, AugmentationPolicy(), rng=random.Random(2))
    np.testing.assert_array_equal(one, two)
    assert replay_one == replay_two


def test_global_gain_preserves_channel_ratio_and_skips_clipping():
    image = np.full((20, 40, 3), [50, 100, 150], dtype=np.uint8)
    policy = AugmentationPolicy(profile="modality", horizontal_flip_probability=0,
                                rotation_probability=0, standard_gain_probability=1,
                                standard_gain_range=(1.1, 1.1))
    af, replay = augment_image(image, "fluorescence", policy)
    np.testing.assert_array_equal(af, image)
    assert replay["gain"] == 1
    white, replay = augment_image(image, "standard", policy)
    np.testing.assert_array_equal(white, np.full_like(image, [55, 110, 165]))
    assert replay["gain"] == 1.1
    saturated = image.copy()
    saturated[0, 0] = 255
    result, replay = augment_image(saturated, "standard", policy)
    np.testing.assert_array_equal(result, saturated)
    assert replay["gain_skipped_clipping"]


@pytest.mark.parametrize("bad", ["rotation_degrees", "standard_gain_probability"])
def test_unsafe_policy_rejected(bad):
    with pytest.raises(ValueError):
        replace(AugmentationPolicy(), **{bad: 100})


def test_unknown_modality_and_invalid_pixels_rejected():
    with pytest.raises(ValueError):
        augment_image(np.zeros((10, 10, 3), np.uint8), "unknown", AugmentationPolicy())
    with pytest.raises(ValueError):
        augment_image(np.zeros((10, 10, 3), np.float32), "standard", AugmentationPolicy())


@pytest.mark.parametrize("change", ["group_leak", "directory_label", "hash", "duplicate_id"])
def test_handoff_rejects_corrupt_mapping_or_integrity(synthetic_handoff, change):
    path = synthetic_handoff
    config = json.loads(path.read_text())
    assert validate_handoff(path)["splits"]["train"]["images"] == 4
    if change == "group_leak":
        next(r for r in config["records"] if r["split"] == "val")["group_id"] = "group_train"
    elif change == "directory_label":
        config["records"][0]["label"] = 1
    elif change == "duplicate_id":
        config["records"][1]["image_id"] = config["records"][0]["image_id"]
    else:
        image = path.parent / config["records"][0]["image_relpath"]
        image.write_bytes(image.read_bytes() + b"changed")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_handoff(path)


def test_legacy_detection_yaml_is_rejected(tmp_path):
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(dict(names={0: "normal_region", 1: "lesion_region"})))
    with pytest.raises(ValueError):
        validate_handoff(path)


def test_dataset_scalar_labels_batch_and_deterministic_eval(synthetic_handoff):
    root = synthetic_handoff.parent
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = json.loads(synthetic_handoff.read_text())
    expected = {row["image_id"]: row["label"] for row in manifest["records"]}
    policy = AugmentationPolicy(horizontal_flip_probability=1, rotation_probability=1,
                                profile="modality", standard_gain_probability=1)
    for split in ("train", "val", "test"):
        dataset = ImageClassificationDataset(synthetic_handoff, split, policy=policy, imgsz=64)
        assert len(dataset) == 4
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item["label"], int) and item["label"] == expected[item["image_id"]]
            assert item["image"].dtype == torch.float32 and item["image"].shape == (3, 64, 64)
            assert torch.isfinite(item["image"]).all()
            assert not any(key in item for key in ("boxes", "bboxes", "cls"))
            if split != "train":
                assert torch.equal(item["image"], dataset[i]["image"])
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        assert batch["label"].shape == (2,) and batch["label"].dtype == torch.int64
        logits = torch.nn.Linear(3, 2)(batch["image"].mean(dim=(2, 3)))
        loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        assert torch.isfinite(loss)
        loss.backward()
    one_mode = ImageClassificationDataset(synthetic_handoff, "train", imgsz=64,
                                         modalities=["fluorescence"])
    assert len(one_mode) == 2 and all(one_mode[i]["modality"] == "fluorescence" for i in range(2))
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not list(root.rglob("*.xml")) and not list(root.rglob("*.txt"))


def test_checksum_manifest_detects_input_change(synthetic_handoff):
    root = synthetic_handoff.parent
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}"
             for p in sorted(root.rglob("*")) if p.is_file()]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    assert verify_checksums(root) == len(lines)
    synthetic_handoff.write_text(synthetic_handoff.read_text() + " ")
    with pytest.raises(ValueError):
        verify_checksums(root)


def test_one_epoch_synthetic_classification_never_loads_test(synthetic_handoff, tmp_path, monkeypatch):
    from oral_fluorescence_detection import image_classification
    torch.set_num_threads(1)
    monkeypatch.setattr(image_classification, "EXPERIMENTS_DIR", tmp_path / "runs")
    root = synthetic_handoff.parent
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    observed = []
    original = ImageClassificationDataset.__init__

    def tracked(self, data_json, split, *args, **kwargs):
        observed.append(split)
        assert split != "test", "Training must not create a test-set loader"
        return original(self, data_json, split, *args, **kwargs)

    monkeypatch.setattr(ImageClassificationDataset, "__init__", tracked)
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(dict(model="resnet18", pretrained=False, epochs=1,
                                         imgsz=64, batch=2, workers=0, device="cpu", seed=17,
                                         lr0=.001, optimizer="SGD", patience=1)))
    best = image_classification.train(synthetic_handoff, config_yaml=config,
                                      name="synthetic_classification_smoke")
    assert best.is_file() and best.name == "best.pt"
    assert set(observed) == {"train", "val"}
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    assert checkpoint["task"] == "image_classification"
    assert checkpoint["model_state_dict"]
    assert not list(synthetic_handoff.parent.rglob("*.cache"))
    # Final test evaluation is a separate explicit call after selecting the checkpoint.
    observed.clear()

    def final_test_only(self, data_json, split, *args, **kwargs):
        observed.append(split)
        assert split == "test", "Final evaluation must use only the held-out split"
        return original(self, data_json, split, *args, **kwargs)

    monkeypatch.setattr(ImageClassificationDataset, "__init__", final_test_only)
    metrics = image_classification.evaluate_checkpoint(
        best, synthetic_handoff, tmp_path / "final_test", device="cpu", batch=2, workers=0)
    assert observed == ["test"] and metrics["images"] == 4
    assert np.asarray(metrics["confusion_matrix"]).sum() == 4
    assert metrics["auroc"] is not None and np.isfinite(metrics["loss"])
    predictions = json.loads((tmp_path / "final_test" / "test_predictions.json").read_text())
    assert len(predictions) == 4
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_check_only_never_builds_or_downloads_model(synthetic_handoff, monkeypatch):
    from oral_fluorescence_detection import image_classification

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not construct or download a model")

    monkeypatch.setattr(image_classification, "build_model", forbidden)
    result = image_classification.train(synthetic_handoff, check_only=True)
    assert result["dataset"]["splits"]["train"]["images"] == 4
    assert result["task"] == "image_classification" and result["training_executed"] is False


def test_binary_metrics_auc_ties_and_fixed_threshold():
    from oral_fluorescence_detection.image_classification import binary_metrics
    metrics = binary_metrics([0, 1, 0, 1], [.2, .5, .5, .8])
    assert metrics["auroc"] == pytest.approx(.875)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["sensitivity"] == 1 and metrics["specificity"] == .5
    assert metrics["balanced_accuracy"] == .75


@pytest.mark.parametrize("labels,missing", [([0, 0], "sensitivity"), ([1, 1], "specificity")])
def test_binary_metrics_missing_class_is_null(labels, missing):
    from oral_fluorescence_detection.image_classification import binary_metrics
    metrics = binary_metrics(labels, [.1, .6])
    assert metrics[missing] is None and metrics["auroc"] is None
    assert metrics["balanced_accuracy"] is None
