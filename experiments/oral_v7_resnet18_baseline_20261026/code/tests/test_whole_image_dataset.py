"""Migration invariants that prevent clinical label leakage or split leakage."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    "whole_image_builder", Path(__file__).resolve().parents[1] / "scripts/build_whole_image_classification.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class WholeImageMigrationTests(unittest.TestCase):
    def test_png_metadata_check_and_verify_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            Image.new("RGB", (19, 13), color=(12, 90, 45)).save(image_path)
            original_bytes = image_path.read_bytes()
            self.assertEqual(builder.decode_image(image_path), (19, 13))
            self.assertEqual(image_path.read_bytes(), original_bytes)

    def test_full_original_xml_prevents_filtered_mixed_label_leakage(self):
        label, reason = builder.candidate_label("1", {"l"}, {"n", "l"})
        self.assertIsNone(label)
        self.assertEqual(reason, "original_xml_not_uniform_n_or_l")

    def test_uniform_xml_must_match_frozen_label(self):
        self.assertEqual(builder.candidate_label("0", {"n"}, {"n"})[0], 0)
        self.assertEqual(builder.candidate_label("1", {"l"}, {"l"})[0], 1)
        self.assertIsNone(builder.candidate_label("0", {"l"}, {"l"})[0])
        self.assertIsNone(builder.candidate_label("0", set(), set())[0])
        self.assertIsNone(builder.candidate_label("0", {"n"}, {"other"})[0])

    def test_source_folder_cannot_cross_splits(self):
        rows = [
            {"image_id": "P001_I001", "split_group_id": "P001", "split": "train", "modality": "standard"},
            {"image_id": "P001_I002", "split_group_id": "P001", "split": "test", "modality": "fluorescence"},
        ]
        with self.assertRaisesRegex(ValueError, "cross split"):
            builder.validate_rows(rows)

    def test_package_paths_cannot_escape_via_traversal_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "package"
            root.mkdir()
            outside = Path(tmp) / "outside.jpg"
            outside.write_bytes(b"x")
            (root / "linked.jpg").symlink_to(outside)
            for relative in ("../outside.jpg", str(outside), "linked.jpg"):
                with self.assertRaises(ValueError):
                    builder.contained_path(root, relative)


if __name__ == "__main__":
    unittest.main()
