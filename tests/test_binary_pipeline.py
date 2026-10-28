from __future__ import annotations

import unittest

import numpy as np

from src.classification.data import LABEL_TO_INDEX, _stratified_patient_split
from src.classification.metrics import (
    binary_metrics,
    select_operating_threshold,
)


class BinaryPipelineTests(unittest.TestCase):
    def test_label_semantics(self):
        self.assertEqual(LABEL_TO_INDEX, {"n": 0, "l": 1})

    def test_stratified_patient_split_has_no_overlap(self):
        labels = {f"p{index}": index % 2 for index in range(80)}
        split = _stratified_patient_split(labels, (0.70, 0.15, 0.15), seed=42)
        self.assertFalse(split["train"] & split["val"])
        self.assertFalse(split["train"] & split["test"])
        self.assertFalse(split["val"] & split["test"])
        self.assertEqual(sum(len(value) for value in split.values()), 80)

    def test_binary_metrics_known_values(self):
        metrics = binary_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.4, 0.6, 0.9]),
            threshold=0.5,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)

    def test_threshold_selection_returns_valid_threshold(self):
        result = select_operating_threshold(
            [0, 0, 1, 1, 0, 1],
            [0.10, 0.20, 0.65, 0.80, 0.30, 0.70],
        )
        self.assertGreaterEqual(result["selected"]["threshold"], 0.01)
        self.assertLessEqual(result["selected"]["threshold"], 0.99)


if __name__ == "__main__":
    unittest.main()

