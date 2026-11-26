"""Plot train/validation/test porosity distributions for each saved scenario."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("scenario-I", "scenario-II", "scenario-III")
SPLITS = ("train", "validation", "test")
SPLIT_LABELS = ("Train", "Validation", "Test")
SPLIT_COLORS = ("#4C78A8", "#F2CF5B", "#E45756")
POROSITY_BIN_COUNT = 8
PNM_PHASE = "ice"

SCATTER_FIGSIZE = (16, 8)
SCATTER_SIZE = 24
SCATTER_COLOR = "#4C78A8"
SCATTER_EDGE_COLOR = "black"
SCATTER_ALPHA = 0.65
SCATTER_MIN_POROSITY = 0.12
SCATTER_OUTPUT_FILENAME = "porosity_feature_scatter_scenario-I.png"
SUMMARY_TABLE_FILENAME = "scenario-I_feature_summary.tex"
MEAN_TORTUOSITY_COLUMNS = ("tortuosity_x", "tortuosity_y", "tortuosity_z")
FEM_PERMEABILITY_COLUMNS = (
    "fem_permeability_x_m2",
    "fem_permeability_y_m2",
    "fem_permeability_z_m2",
)

SCATTER_PANELS = (
    {
        "key": "euler_density",
        "title": "Euler density",
        "ylabel": "Euler density (cm$^{-3}$)",
        "columns": ("euler_density_per_cm3",),
    },
    {
        "key": "ssa",
        "title": "Specific surface area",
        "ylabel": "SSA (m$^2$/m$^3$)",
        "columns": ("specific_surface_area_m2_per_m3",),
    },
    {
        "key": "coordination_number",
        "title": "Coordination number",
        "ylabel": "Coordination number",
        "columns": (f"{PNM_PHASE}_PNM_coordination_number",),
    },
    {
        "key": "throat_diameter",
        "title": "Mean throat diameter",
        "ylabel": "Mean throat diameter (mm)",
        "columns": (f"{PNM_PHASE}_PNM_avg_throat_diameter",),
    },
    {
        "key": "number_of_pores",
        "title": "Number of pores",
        "ylabel": "Number of pores",
        "columns": (f"{PNM_PHASE}_PNM_num_pores",),
    },
    {
        "key": "mean_tortuosity",
        "title": "Mean tortuosity",
        "ylabel": "Mean tortuosity",
        "columns": MEAN_TORTUOSITY_COLUMNS,
    },
    {
        "key": "mean_mil",
        "title": "Mean MIL",
        "ylabel": "Mean MIL (mm)",
        "columns": (f"mil_mean_mm_{PNM_PHASE}",),
    },
    {
        "key": "log10mean_fem_permeability",
        "title": "Log10 mean FEM permeability",
        "ylabel": "log10(mean FEM permeability)",
        "columns": FEM_PERMEABILITY_COLUMNS,
    },
)

# Set any limit to None to use the axis' automatic limits.
SCATTER_AXIS_LIMITS = {
    "euler_density": {"xlim": None, "ylim": None},
    "ssa": {"xlim": None, "ylim": None},
    "coordination_number": {"xlim": None, "ylim": None},
    "throat_diameter": {"xlim": None, "ylim": None},
    "number_of_pores": {"xlim": None, "ylim": None},
    "mean_tortuosity": {"xlim": None, "ylim": (0, 200)},
    "mean_mil": {"xlim": None, "ylim": None},
    "log10mean_fem_permeability": {"xlim": None, "ylim": None},
}

TABLE_ROW_LABELS = {
    "euler_density": r"Euler density ($\mathrm{cm}^{-3}$)",
    "ssa": r"Specific surface area ($\mathrm{m}^2/\mathrm{m}^3$)",
    "coordination_number": "Coordination number",
    "throat_diameter": "Mean throat diameter (mm)",
    "number_of_pores": "Number of pores",
    "mean_tortuosity": "Mean tortuosity",
    "mean_mil": "Mean MIL (mm)",
    "log10mean_fem_permeability": r"$\log_{10}$ mean FEM permeability",
}


def _prediction_file(scenario: str) -> Path:
    scenario_dir = ROOT / "checkpoints" / scenario
    files = sorted(scenario_dir.rglob("*_predictions.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No *_predictions.xlsx workbook found under {scenario_dir}."
        )
    if len(files) > 1:
        raise ValueError(
            f"Found multiple prediction workbooks for {scenario}: "
            f"{', '.join(str(path) for path in files)}"
        )
    return files[0]


def _read_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_excel(
        path,
        sheet_name="Predictions",
        usecols=["dataset_split", "porosity"],
    )
    split_names = predictions["dataset_split"]
    if split_names.isna().any():
        raise ValueError(f"Missing dataset_split values in {path}.")

    predictions["dataset_split"] = split_names.astype(str).str.strip().str.casefold()
    present_splits = set(predictions["dataset_split"])
    unexpected = sorted(present_splits - set(SPLITS))
    missing = [split for split in SPLITS if split not in present_splits]
    if unexpected or missing:
        raise ValueError(
            f"Invalid split labels in {path}; "
            f"missing={missing}, unexpected={unexpected}."
        )

    porosity = pd.to_numeric(predictions["porosity"], errors="coerce")
    if not np.isfinite(porosity.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite porosity values in {path}.")
    predictions["porosity"] = porosity
    return predictions


def _read_scenario_feature_data() -> pd.DataFrame:
    prediction_path = _prediction_file("scenario-I")
    columns = ["porosity"]
    columns.extend(
        column
        for panel in SCATTER_PANELS
        for column in panel["columns"]
    )
    columns = list(dict.fromkeys(columns))
    data = pd.read_excel(
        prediction_path, sheet_name="Predictions", usecols=columns
    )
    data["porosity"] = pd.to_numeric(data["porosity"], errors="coerce")
    if not np.isfinite(data["porosity"].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite porosity values in {prediction_path}.")
    data = data[data["porosity"] > SCATTER_MIN_POROSITY].copy()
    if data.empty:
        raise ValueError(
            f"No Scenario-I rows have porosity greater than "
            f"{SCATTER_MIN_POROSITY}."
        )
    return data


def _feature_values(
    data: pd.DataFrame, panel: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    values = data.loc[:, panel["columns"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    if panel["key"] == "mean_tortuosity":
        valid_rows = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        return values[valid_rows].mean(axis=1), valid_rows
    if panel["key"] == "log10mean_fem_permeability":
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("FEM permeability values must be finite and positive.")
        return np.log10(values.mean(axis=1)), np.ones(len(data), dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing or non-finite values in feature columns {panel['columns']}."
        )
    if panel["key"] == "throat_diameter":
        return values[:, 0] * 1000, np.ones(len(data), dtype=bool)
    return values[:, 0], np.ones(len(data), dtype=bool)


def _plot_feature_scatter(data: pd.DataFrame) -> None:
    porosity = data["porosity"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 4, figsize=SCATTER_FIGSIZE)
    for panel_index, (axis, panel) in enumerate(zip(axes.flat, SCATTER_PANELS)):
        y_values, valid_rows = _feature_values(data, panel)
        x_values = porosity[valid_rows]
        if panel["key"] == "mean_tortuosity":
            print(
                "Mean tortuosity: "
                f"using {int(valid_rows.sum())} valid rows; "
                f"excluded {int((~valid_rows).sum())} invalid rows."
            )

        axis.scatter(
            x_values,
            y_values,
            s=SCATTER_SIZE,
            c=SCATTER_COLOR,
            edgecolors=SCATTER_EDGE_COLOR,
            alpha=SCATTER_ALPHA,
            linewidths=0.3,
        )
        axis.set_title(f"({chr(ord('a') + panel_index)})")
        axis.set_xlabel("Porosity")
        axis.set_ylabel(panel["ylabel"])
        axis.grid(linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

        limits = SCATTER_AXIS_LIMITS[panel["key"]]
        if limits["xlim"] is not None:
            axis.set_xlim(*limits["xlim"])
        if limits["ylim"] is not None:
            axis.set_ylim(*limits["ylim"])

    fig.tight_layout()
    output_path = ROOT / "figs" / SCATTER_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def _latex_summary_table(data: pd.DataFrame) -> str:
    rows = []
    for panel in SCATTER_PANELS:
        values, _ = _feature_values(data, panel)
        statistics = (
            float(np.min(values)),
            float(np.mean(values)),
            float(np.max(values)),
            float(np.std(values, ddof=1)),
        )
        formatted_statistics = " & ".join(
            f"{value:.3e}" for value in statistics
        )
        rows.append(
            f"{TABLE_ROW_LABELS[panel['key']]} & {formatted_statistics} \\\\"
        )

    return "\n".join(
        [
            "% Requires \\usepackage{booktabs,siunitx} in the preamble.",
            r"\begin{table}[htbp]",
            r"\centering",
            (
                r"\caption{Descriptive statistics for Scenario-I samples with "
                f"porosity greater than {SCATTER_MIN_POROSITY}; mean tortuosity "
                r"omits rows with invalid or non-positive directional values.}"
            ),
            r"\label{tab:scenario-i-feature-statistics}",
            r"\begin{tabular}{lSSSS}",
            r"\toprule",
            r"Feature & {Min} & {Mean} & {Max} & {Std. dev.} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _print_latex_summary_table(data: pd.DataFrame) -> None:
    latex_table = _latex_summary_table(data)
    output_path = ROOT / "figs" / SUMMARY_TABLE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table + "\n", encoding="utf-8")
    print("\nLaTeX summary table:\n")
    print(latex_table)
    print(f"\nSaved LaTeX table to {output_path}")


def main() -> None:
    scenario_data = [
        (scenario, _read_predictions(_prediction_file(scenario)))
        for scenario in SCENARIOS
    ]
    all_porosity = np.concatenate(
        [data["porosity"].to_numpy(dtype=float) for _, data in scenario_data]
    )
    min_porosity = float(all_porosity.min())
    max_porosity = float(all_porosity.max())
    if min_porosity == max_porosity:
        raise ValueError("Cannot create porosity bins when all values are identical.")

    bin_edges = np.linspace(
        min_porosity, max_porosity, POROSITY_BIN_COUNT + 1
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    bar_width = bin_width * 0.3
    offsets = (-bar_width, 0, bar_width)
    bin_labels = [
        f"{left:.2f}-{right:.2f}"
        for left, right in zip(bin_edges[:-1], bin_edges[1:])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, (scenario, data) in zip(axes, scenario_data):
        axis.grid(axis="y", linestyle="--", alpha=0.8, color="black")
        for split, label, color, offset in zip(
            SPLITS, SPLIT_LABELS, SPLIT_COLORS, offsets
        ):
            split_porosity = data.loc[
                data["dataset_split"] == split, "porosity"
            ].to_numpy(dtype=float)
            bin_counts = np.histogram(split_porosity, bins=bin_edges)[0]
            axis.bar(
                bin_centers + offset,
                bin_counts,
                width=bar_width,
                color=color,
                label=label,
                edgecolor="black"
            )
        axis.set_title(scenario[0].upper() + scenario[1:])
        axis.set_xticks(bin_centers, bin_labels, rotation=45, ha="right")
        axis.set_xlabel("Porosity range")
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Number of samples per bin")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=len(SPLITS),
        frameon=False,
    )
    
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir = ROOT / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_val_test_split_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    scenario_i_data = _read_scenario_feature_data()
    _plot_feature_scatter(scenario_i_data)
    _print_latex_summary_table(scenario_i_data)


if __name__ == "__main__":
    main()