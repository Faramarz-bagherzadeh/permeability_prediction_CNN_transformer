"""Dataset utilities for pairing VTI images with directional permeability labels."""

from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _permeability_columns(columns: Sequence[object]) -> list[str]:
    normalized = {str(column).casefold().replace(" ", ""): str(column) for column in columns}
    selected = []
    for direction in ("x", "y", "z"):
        prefix = f"lbm_k{direction}"
        matches = [original for name, original in normalized.items() if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one spreadsheet column starting with {prefix!r}; "
                f"found {matches}."
            )
        selected.append(matches[0])
    return selected


class PermeabilityDataset(Dataset):
    """Load VTI pore images and their three permeability values."""

    def __init__(
        self,
        data_dir,
        labels_file,
        image_column,
        porosity_column,
        target_columns,
        target_transform,
        target_floor,
    ) -> None:
        self.data_dir = Path(data_dir)
        labels_path = Path(labels_file)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.data_dir}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels spreadsheet does not exist: {labels_path}")
        if target_transform not in ("log10", "identity"):
            raise ValueError("target_transform must be either 'log10' or 'identity'.")
        if (
            isinstance(target_floor, bool)
            or not isinstance(target_floor, (int, float, np.number))
            or not np.isfinite(target_floor)
            or target_floor <= 0
        ):
            raise ValueError("target_floor must be a finite positive value.")

        table = pd.read_excel(labels_path)
        if image_column not in table.columns:
            raise ValueError(
                f"Image column {image_column!r} was not found in {labels_path}. "
                f"Available columns: {list(table.columns)}"
            )
        if porosity_column not in table.columns:
            raise ValueError(
                f"Porosity column {porosity_column!r} was not found in {labels_path}. "
                f"Available columns: {list(table.columns)}"
            )
        self.target_columns = list(target_columns or _permeability_columns(table.columns))
        if len(self.target_columns) != 3 or any(
            column not in table.columns for column in self.target_columns
        ):
            raise ValueError("target_columns must name three columns present in the spreadsheet.")
        self.target_transform = target_transform
        self.target_floor = float(target_floor)

        labels_by_stem: dict[str, tuple[np.ndarray, object]] = {}
        for _, row in table.iterrows():
            name = row[image_column]
            if pd.isna(name):
                continue
            stem = Path(str(name)).stem.casefold()
            if stem in labels_by_stem:
                raise ValueError(f"Duplicate label rows found for image {name!r}.")
            try:
                values = row[self.target_columns].to_numpy(dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid permeability values for image {name!r}.") from error
            labels_by_stem[stem] = values, row[porosity_column]

        all_image_paths = sorted(
            path for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".vti"
        )
        if not all_image_paths:
            raise FileNotFoundError(f"No .vti image files found in {self.data_dir}.")

        missing_labels = [
            path.name for path in all_image_paths if path.stem.casefold() not in labels_by_stem
        ]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            warnings.warn(
                f"Ignoring {len(missing_labels)} VTI file(s) without ground-truth rows: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )
        image_paths = [
            path for path in all_image_paths if path.stem.casefold() in labels_by_stem
        ]
        if not image_paths:
            raise ValueError("No VTI images have matching ground-truth rows.")

        self.image_paths = image_paths
        matched_rows = [labels_by_stem[path.stem.casefold()] for path in image_paths]
        targets = np.stack([row[0] for row in matched_rows])
        self.nan_target_count = int(np.count_nonzero(np.isnan(targets)))
        targets = np.where(np.isnan(targets), self.target_floor, targets)
        if np.isinf(targets).any():
            raise ValueError("Infinite permeability values are not supported for matched images.")
        try:
            porosities = np.asarray([float(row[1]) for row in matched_rows], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Porosity values must be numeric for every matched image.") from error
        if not np.isfinite(porosities).all():
            raise ValueError("Porosity values must be finite for every matched image.")
        self.porosities = porosities
        self.negative_target_count = int(np.count_nonzero(targets < 0))
        targets = np.maximum(targets, 0)
        if self.target_transform == "log10":
            targets = np.maximum(targets, self.target_floor)
            targets = np.log10(targets)
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image(self, path: Path) -> torch.Tensor:
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            from vtkmodules.vtkIOXML import vtkXMLImageDataReader
        except ImportError as error:
            raise ImportError(
                "Reading .vti files requires the 'vtk' package. Install it with: pip install vtk"
            ) from error

        reader = vtkXMLImageDataReader()
        reader.SetFileName(str(path))
        reader.UpdateInformation()
        selection = reader.GetCellDataArraySelection()
        selection.DisableAllArrays()
        selection.EnableArray("image")
        reader.Update()

        image_data = reader.GetOutput()
        scalars = image_data.GetCellData().GetArray("image")
        if scalars is None:
            raise ValueError(f"No cell-data array named 'image' in {path}.")

        extent = image_data.GetExtent()
        dimensions = (extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
        if any(size <= 0 for size in dimensions):
            raise ValueError(f"Invalid image extent in {path}: {extent}.")
        image_array = vtk_to_numpy(scalars)
        expected_values = int(np.prod(dimensions))
        if image_array.size != expected_values:
            raise ValueError(
                f"Expected {expected_values} image voxels in {path}, got {image_array.size}."
            )

        depth, height, width = dimensions[2], dimensions[1], dimensions[0]
        image_array = np.ascontiguousarray(image_array.reshape(depth, height, width))
        image = torch.from_numpy(image_array).to(dtype=torch.float32)
        if image.max().item() > 1:
            image = image / 255.0
        return image.unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._read_image(self.image_paths[index])
        return image, self.targets[index]