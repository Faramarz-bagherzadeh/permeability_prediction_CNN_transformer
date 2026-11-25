import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class PermeabilityDatasetTests(unittest.TestCase):
    def test_drops_rows_with_nan_negative_or_threshold_targets(self) -> None:
        from dataloader import PermeabilityDataset

        target_columns = ["kx", "ky", "kz"]
        rows = [
            ("valid.vti", 0.1, 2.1e-9, 3e-9, 4e-9, "Converged"),
            ("at_threshold.vti", 0.2, 2e-9, 3e-9, 4e-9, "Converged"),
            ("negative.vti", 0.3, 3e-9, -1.0, 4e-9, "Converged"),
            ("nan.vti", 0.4, 3e-9, 4e-9, np.nan, "Converged"),
            ("not_converged.vti", 0.5, 3e-9, 4e-9, 5e-9, "Diverged"),
        ]
        labels = pd.DataFrame(
            rows,
            columns=["image", "porosity", *target_columns, "overall_status"],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "images"
            data_dir.mkdir()
            for name, *_ in rows:
                (data_dir / name).touch()
            labels_file = root / "labels.xlsx"
            labels_file.touch()

            with patch("dataloader.pd.read_excel", return_value=labels):
                dataset = PermeabilityDataset(
                    data_dir=data_dir,
                    labels_file=labels_file,
                    image_column="image",
                    porosity_column="porosity",
                    target_columns=target_columns,
                    target_transform="identity",
                    target_threshold=2e-9,
                    target_floor=1e-14,
                )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.image_paths[0].name, "valid.vti")
        self.assertEqual(dataset.dropped_target_row_count, 3)
        self.assertEqual(dataset.dropped_status_row_count, 1)
        np.testing.assert_allclose(dataset.targets.numpy(), [[2.1e-9, 3e-9, 4e-9]])


if __name__ == "__main__":
    unittest.main()
