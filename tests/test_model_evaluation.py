import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(
    __import__("importlib.util").util.find_spec("torch"),
    "PyTorch is not installed.",
)
class PredictionExportTests(unittest.TestCase):
    def test_combined_prediction_rows_keep_ground_truth_columns_and_split_label(
        self,
    ) -> None:
        from model_evaluation import _prediction_rows
        import torch

        ground_truth = pd.DataFrame(
            {
                "image": ["sample_a.vti", "sample_b.vti"],
                "porosity": [0.2, 0.3],
                "sample_id": ["A", "B"],
                "LBM_Kx": [1e-8, 2e-8],
                "LBM_Ky": [3e-8, 4e-8],
                "LBM_Kz": [5e-8, 6e-8],
            }
        )

        class DatasetStub:
            image_paths = [Path("sample_a.vti"), Path("sample_b.vti")]
            target_columns = ["LBM_Kx", "LBM_Ky", "LBM_Kz"]

        config = {
            "labels_file": "unused.xlsx",
            "image_column": "image",
            "target_transform": "log10",
        }

        with tempfile.TemporaryDirectory(), patch(
            "model_evaluation.pd.read_excel",
            return_value=ground_truth,
        ):
            rows = _prediction_rows(
                config,
                DatasetStub(),
                np.array([0]),
                torch.tensor([[ -8.0, -7.5228787, -7.30103 ]]),
                "train",
            )

        self.assertEqual(rows.loc[0, "dataset_split"], "train")
        self.assertEqual(rows.loc[0, "vti_file_name"], "sample_a.vti")
        self.assertEqual(rows.loc[0, "sample_id"], "A")
        self.assertIn("porosity", rows.columns)
        self.assertIn("predicted_LBM_Kx", rows.columns)
        self.assertAlmostEqual(rows.loc[0, "predicted_LBM_Kx"], 1e-8, places=12)


if __name__ == "__main__":
    unittest.main()
