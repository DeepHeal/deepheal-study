#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.util
import json
import math
import string
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from openpyxl import load_workbook


def find_project_root(start: Path) -> Path:
    return ARTICLE_ROOT


ROOT = ARTICLE_ROOT
MAY18_HELPER = ROOT / 'src/deepheal/experimental/fit_dose_response.py'
JP1302_XLSX = ROOT / 'data/raw/HCT116_cck_data.xlsx'
TABLE_DIR = ROOT / "results" / "experimental" / "tables"
FIG_DIR = ROOT / "results" / "experimental" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_ROOT = ROOT / "results" / "experimental"

JP1302_RAW_PATH = TABLE_DIR / "deepheal_cck8_jp1302_raw_long.csv"
JP1302_FIT_PATH = TABLE_DIR / "deepheal_cck8_jp1302_fit_summary.csv"
JP1302_SCREEN_PATH = TABLE_DIR / "deepheal_cck8_jp1302_50uM_screen.csv"
COMBINED_RAW_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_raw_long.csv"
COMBINED_DOSE_SUMMARY_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_dose_summary.csv"
COMBINED_FIT_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_fit_summary.csv"
COMBINED_SOURCE_BLOCKS_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_source_blocks.csv"
WORKBOOK_PAYLOAD_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_workbook_payload.json"
MANIFEST_PATH = TABLE_DIR / "deepheal_cck8_jp1302_combined_manifest.json"
MAY18_MANIFEST_PATH = TABLE_DIR / 'deepheal_cck8_ic50_manifest.json'
UNDER10_SUMMARY_PATH = TABLE_DIR / 'deepheal_cck8_ic50_ic50_lt10uM_summary.csv'

UNDER10_FIGURE_STEM = FIG_DIR / 'deepheal_cck8_ic50_ic50_lt10uM'
COMBINED_XLSX_PATH = PROCESSED_DIR / 'deepheal_cck8_clean_combined.xlsx'
CANONICAL_INPUT_CSV_PATH = PROCESSED_DIR / 'deepheal_cck8_clean_combined_input.csv'

DPI = 300
RNG_SEED = 20260526
UNDER10_COMPOUND_LABELS = [
    "PF-431396",
    "VER-82576",
    "SB225002",
    "BIX-01294",
    "OTSSP167",
    "BAY 61-3606",
    "JP1302",
]


def configure_output_root(output_root: Path) -> None:
    global OUTPUT_ROOT, TABLE_DIR, FIG_DIR, PROCESSED_DIR
    global JP1302_RAW_PATH, JP1302_FIT_PATH, JP1302_SCREEN_PATH
    global COMBINED_RAW_PATH, COMBINED_DOSE_SUMMARY_PATH, COMBINED_FIT_PATH, COMBINED_SOURCE_BLOCKS_PATH
    global WORKBOOK_PAYLOAD_PATH, MANIFEST_PATH, MAY18_MANIFEST_PATH, UNDER10_SUMMARY_PATH
    global UNDER10_FIGURE_STEM, COMBINED_XLSX_PATH, CANONICAL_INPUT_CSV_PATH

    OUTPUT_ROOT = output_root.resolve()
    TABLE_DIR = OUTPUT_ROOT / "tables"
    FIG_DIR = OUTPUT_ROOT / "figures"
    PROCESSED_DIR = OUTPUT_ROOT / "data" / "processed"
    JP1302_RAW_PATH = TABLE_DIR / "deepheal_cck8_jp1302_raw_long.csv"
    JP1302_FIT_PATH = TABLE_DIR / "deepheal_cck8_jp1302_fit_summary.csv"
    JP1302_SCREEN_PATH = TABLE_DIR / "deepheal_cck8_jp1302_50uM_screen.csv"
    COMBINED_RAW_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_raw_long.csv"
    COMBINED_DOSE_SUMMARY_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_dose_summary.csv"
    COMBINED_FIT_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_fit_summary.csv"
    COMBINED_SOURCE_BLOCKS_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_source_blocks.csv"
    WORKBOOK_PAYLOAD_PATH = TABLE_DIR / "deepheal_cck8_clean_combined_workbook_payload.json"
    MANIFEST_PATH = TABLE_DIR / "deepheal_cck8_jp1302_combined_manifest.json"
    MAY18_MANIFEST_PATH = TABLE_DIR / 'deepheal_cck8_ic50_manifest.json'
    UNDER10_SUMMARY_PATH = TABLE_DIR / 'deepheal_cck8_ic50_ic50_lt10uM_summary.csv'
    UNDER10_FIGURE_STEM = FIG_DIR / 'deepheal_cck8_ic50_ic50_lt10uM'
    COMBINED_XLSX_PATH = PROCESSED_DIR / 'deepheal_cck8_clean_combined.xlsx'
    CANONICAL_INPUT_CSV_PATH = PROCESSED_DIR / 'deepheal_cck8_clean_combined_input.csv'


def relative_to_base(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def project_relative(path: Path) -> str:
    return relative_to_base(path, ROOT)


def output_relative(path: Path) -> str:
    return relative_to_base(path, OUTPUT_ROOT)


def resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else OUTPUT_ROOT / candidate


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MAY18 = load_module(MAY18_HELPER, "deepheal_cck8_may18_helper")

build_fit_summary = MAY18.build_fit_summary
four_parameter_logistic = MAY18.four_parameter_logistic
compact_tick = MAY18.compact_tick


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").strip().split())


def is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_compound_target(label: str) -> tuple[str, str]:
    text = normalize_text(label)
    if "_" not in text:
        return text, ""
    compound_name, target = text.split("_", 1)
    return compound_name.strip(), target.strip()


def find_jp1302_dose_block(ws) -> tuple[int, int, int, int]:
    for merged in ws.merged_cells.ranges:
        value = normalize_text(ws.cell(merged.min_row, merged.min_col).value)
        if value == "JP1302_ADRA2C" and merged.min_row >= 10:
            concentration_col = merged.min_col
            response_start_col = concentration_col + 1
            response_end_col = merged.max_col
            return merged.min_row, concentration_col, response_start_col, response_end_col
    raise ValueError("Could not locate JP1302_ADRA2C dose-response block.")


def parse_jp1302_raw(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wb = load_workbook(xlsx_path, data_only=True, read_only=False)
    ws = wb["Sheet1"]
    header_row, concentration_col, response_start_col, response_end_col = find_jp1302_dose_block(ws)
    compound_full = normalize_text(ws.cell(header_row, concentration_col).value)
    compound_label, target = parse_compound_target(compound_full)
    source_block_key = (
        f"{xlsx_path.name}|Sheet1|r{header_row}|c{concentration_col}|"
        f"resp{response_start_col}-{response_end_col}"
    )

    rows: list[dict[str, object]] = []
    for row_idx in range(header_row + 2, ws.max_row + 1):
        concentration = ws.cell(row_idx, concentration_col).value
        responses = [ws.cell(row_idx, col).value for col in range(response_start_col, response_end_col + 1)]
        numeric_responses = [response for response in responses if is_numeric(response)]
        if is_numeric(concentration) and numeric_responses:
            for replicate, response in enumerate(responses, start=1):
                if not is_numeric(response):
                    continue
                rows.append(
                    {
                        "compound": "19. JP1302",
                        "compound_label": compound_label,
                        "compound_name": compound_label,
                        "target": target,
                        "compound_order": 19,
                        "dataset": 'JP1302',
                        "source_workbook": project_relative(xlsx_path),
                        "source_sheet": "Sheet1",
                        "header_row": header_row,
                        "concentration_col": concentration_col,
                        "response_col": response_start_col + replicate - 1,
                        "response_start_col": response_start_col,
                        "response_end_col": response_end_col,
                        "source_block_key": source_block_key,
                        "unit": "uM",
                        "concentration_native": float(concentration),
                        "concentration_uM": float(concentration),
                        "replicate": replicate,
                        "response": float(response),
                        "response_percent": float(response),
                        "response_metric": "Inhibition (%)",
                        "existing_ic50_note": "",
                    }
                )
        elif rows:
            break

    raw = pd.DataFrame(rows).sort_values(["concentration_native", "replicate"]).reset_index(drop=True)
    if raw.empty:
        raise ValueError("JP1302 dose-response block had no numeric rows.")

    source_blocks = pd.DataFrame(
        [
            {
                "source_block_key": source_block_key,
                "compound": "19. JP1302",
                "compound_label": compound_label,
                "compound_name": compound_label,
                "target": target,
                "compound_order": 19,
                "dataset": 'JP1302',
                "source_workbook": project_relative(xlsx_path),
                "source_sheet": "Sheet1",
                "header_row": header_row,
                "concentration_col": concentration_col,
                "response_start_col": response_start_col,
                "response_end_col": response_end_col,
                "run_index": 1,
                "selected": True,
                "selection_rule": "selected JP1302 dose-response block",
                "n_concentrations": int(raw["concentration_native"].nunique()),
                "n_observations": int(len(raw)),
                "concentration_min_native": float(raw["concentration_native"].min()),
                "concentration_max_native": float(raw["concentration_native"].max()),
            }
        ]
    )

    screen_rows: list[dict[str, object]] = []
    screen_col = None
    for cell in ws[4]:
        if normalize_text(cell.value) == compound_full:
            screen_col = cell.column
            break
    if screen_col is not None:
        for replicate, row_idx in enumerate([5, 6, 7], start=1):
            value = ws.cell(row_idx, screen_col).value
            if is_numeric(value):
                screen_rows.append(
                    {
                        "dataset": 'JP1302_screen',
                        "source_workbook": project_relative(xlsx_path),
                        "source_sheet": "Sheet1",
                        "compound_label": compound_label,
                        "compound_name": compound_label,
                        "target": target,
                        "concentration_uM": 50.0,
                        "replicate": replicate,
                        "response_percent": float(value),
                        "response_metric": "Inhibition (%)",
                        "source_cell": ws.cell(row_idx, screen_col).coordinate,
                    }
                )
    screen = pd.DataFrame(screen_rows)
    return raw, source_blocks, screen


def prepare_may18_clean_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_all = MAY18.parse_excel_raw(MAY18.RAW_XLSX)
    raw, selection = MAY18.build_source_block_selection(raw_all)
    raw = raw.copy()
    raw["compound_name"] = raw["compound_label"]
    raw["target"] = ""
    raw["dataset"] = 'cck_final'
    raw["source_workbook"] = project_relative(MAY18.RAW_XLSX)
    raw["response_percent"] = raw["response"]
    raw["response_metric"] = "Response (%)"

    selection = selection.copy()
    selection["compound_name"] = selection["compound_label"]
    selection["target"] = ""
    selection["dataset"] = 'cck_final'
    selection["source_workbook"] = project_relative(MAY18.RAW_XLSX)
    return raw, selection


def harmonize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "source_workbook",
        "source_sheet",
        "source_block_key",
        "compound",
        "compound_order",
        "compound_label",
        "compound_name",
        "target",
        "unit",
        "concentration_native",
        "concentration_uM",
        "replicate",
        "response",
        "response_percent",
        "response_metric",
        "header_row",
        "concentration_col",
        "response_col",
        "response_start_col",
        "response_end_col",
        "existing_ic50_note",
    ]
    for column in columns:
        if column not in raw.columns:
            raw[column] = ""
    return raw[columns].sort_values(["compound_order", "concentration_uM", "replicate"]).reset_index(drop=True)


def fit_for_workbook(raw: pd.DataFrame) -> pd.DataFrame:
    fit = build_fit_summary(raw)
    metadata = raw[["compound", "dataset", "source_workbook", "compound_name", "target", "source_block_key"]].drop_duplicates(
        "compound"
    )
    fit = fit.merge(metadata, on="compound", how="left")
    ordered_cols = [
        "dataset",
        "source_workbook",
        "source_block_key",
        "compound",
        "compound_order",
        "compound_label",
        "compound_name",
        "target",
        "unit",
        "n_concentrations",
        "n_observations",
        "replicate_count_min",
        "replicate_count_max",
        "concentration_min_native",
        "concentration_max_native",
        "fit_status",
        "nd_reason",
        "ic50_native",
        "ic50_uM",
        "ic50_ci_low_native",
        "ic50_ci_high_native",
        "ic50_ci_low_uM",
        "ic50_ci_high_uM",
        "log_ic50",
        "r_squared",
        "rmse",
        "fitted_bottom",
        "fitted_top",
        "fitted_span",
        "hill_slope",
        "bootstrap_successes",
        "bootstrap_ci_fold_width",
    ]
    return fit[ordered_cols].sort_values("compound_order").reset_index(drop=True)


def dose_summary(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(
            [
                "dataset",
                "source_workbook",
                "source_block_key",
                "compound_order",
                "compound_label",
                "compound_name",
                "target",
                "unit",
                "concentration_native",
                "concentration_uM",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            n=("response_percent", "size"),
            mean_response_percent=("response_percent", "mean"),
            sd_response_percent=("response_percent", "std"),
            sem_response_percent=("response_percent", lambda s: float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0),
        )
        .reset_index()
    )
    return summary.sort_values(["compound_order", "concentration_uM"]).reset_index(drop=True)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )
    sns.set_theme(style="ticks", context="paper")


def format_ic50_text(fit_row: pd.Series) -> str:
    if str(fit_row["fit_status"]) != "reliable":
        return "IC50 n.d."
    ic50 = float(fit_row["ic50_native"])
    low = float(fit_row["ic50_ci_low_native"])
    high = float(fit_row["ic50_ci_high_native"])
    unit = str(fit_row["unit"]).replace("uM", "\N{MICRO SIGN}M")
    if math.isfinite(low) and math.isfinite(high):
        return f"IC50 {compact_tick(ic50)} {unit}\n95% CI {compact_tick(low)}-{compact_tick(high)}"
    return f"IC50 {compact_tick(ic50)} {unit}"


def select_under10_with_jp1302(fit_summary: pd.DataFrame) -> pd.DataFrame:
    under10 = fit_summary.loc[
        (fit_summary["fit_status"] == "reliable")
        & (fit_summary["ic50_uM"] < 10)
        & fit_summary["compound_label"].isin(UNDER10_COMPOUND_LABELS)
    ].copy()
    under10["compound_label"] = pd.Categorical(
        under10["compound_label"],
        categories=UNDER10_COMPOUND_LABELS,
        ordered=True,
    )
    under10 = under10.sort_values("compound_label").reset_index(drop=True)
    under10["compound_label"] = under10["compound_label"].astype(str)
    observed = under10["compound_label"].tolist()
    if observed != UNDER10_COMPOUND_LABELS:
        raise AssertionError(f"Expected under-10 Fig. 6 labels {UNDER10_COMPOUND_LABELS}, observed {observed}")
    return under10


def draw_under10_combined_figure(raw: pd.DataFrame, fit_summary: pd.DataFrame) -> list[Path]:
    set_publication_style()
    line_color = "#0072B2"
    mean_color = "#D55E00"
    point_color = "#4D4D4D"
    band_color = "#56B4E9"
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.2), sharey=False, squeeze=False)
    axes_flat = axes.ravel()

    for idx, (_, fit_row) in enumerate(fit_summary.iterrows()):
        ax = axes_flat[idx]
        compound = str(fit_row["compound"])
        compound_label = str(fit_row["compound_label"])
        sub = raw.loc[raw["compound"] == compound].copy()
        if sub.empty:
            raise AssertionError(f"No raw data found for {compound_label}")
        unit = str(fit_row["unit"])
        unit_label = unit.replace("uM", "\N{MICRO SIGN}M")
        response_col = "response_percent" if "response_percent" in sub.columns else "response"
        grouped = sub.groupby("concentration_native", sort=True)[response_col]
        means = grouped.mean()
        sems = grouped.sem().fillna(0.0)
        x_values = means.index.to_numpy(dtype=float)

        for _, point in sub.iterrows():
            jitter = float(np.exp(rng.normal(0.0, 0.025)))
            ax.scatter(
                float(point["concentration_native"]) * jitter,
                float(point[response_col]),
                s=9,
                c=point_color,
                alpha=0.52,
                linewidths=0,
                zorder=2,
            )

        ax.errorbar(
            x_values,
            means.to_numpy(dtype=float),
            yerr=sems.to_numpy(dtype=float),
            fmt="o",
            ms=3.2,
            mfc=mean_color,
            mec="white",
            mew=0.35,
            ecolor=mean_color,
            elinewidth=0.78,
            capsize=1.8,
            zorder=4,
        )

        if fit_row["fit_status"] == "reliable":
            x_curve = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), 240)
            params = [
                float(fit_row["fitted_bottom"]),
                float(fit_row["fitted_span"]),
                float(fit_row["log_ic50"]),
                float(fit_row["hill_slope"]),
            ]
            y_curve = four_parameter_logistic(x_curve, *params)
            ax.plot(x_curve, y_curve, color=line_color, lw=1.28, zorder=3)
            low = float(fit_row["ic50_ci_low_native"])
            high = float(fit_row["ic50_ci_high_native"])
            if math.isfinite(low) and math.isfinite(high):
                low_params = params.copy()
                high_params = params.copy()
                low_params[2] = math.log10(low)
                high_params[2] = math.log10(high)
                y_low = four_parameter_logistic(x_curve, *low_params)
                y_high = four_parameter_logistic(x_curve, *high_params)
                ax.fill_between(
                    x_curve,
                    np.minimum(y_low, y_high),
                    np.maximum(y_low, y_high),
                    color=band_color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=1,
                )

        ax.set_xscale("log")
        ax.set_xlim(x_values.min() / 1.25, x_values.max() * 1.25)
        y_min = min(float(sub[response_col].min()), float(means.min())) - 8
        y_max = max(float(sub[response_col].max()), float(means.max())) + 12
        ax.set_ylim(max(-25, y_min), min(125, max(60, y_max)))
        ax.xaxis.set_major_locator(FixedLocator([x_values[0], x_values[len(x_values) // 2], x_values[-1]]))
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.2, pad=1.5)
        ax.set_title(textwrap.fill(compound_label, width=18), loc="left", pad=2.0, fontweight="bold")
        ax.text(
            -0.16,
            1.08,
            string.ascii_lowercase[idx],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.03,
            0.95,
            format_ic50_text(fit_row),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=5.9,
            color="#1A1A1A",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.set_xlabel(f"Concentration ({unit_label})", labelpad=1.8)
        if idx % 4 == 0:
            ax.set_ylabel("Response (%)", labelpad=1.8)
        else:
            ax.set_ylabel("")

    for empty_ax in axes_flat[len(fit_summary) :]:
        empty_ax.axis("off")

    fig.subplots_adjust(left=0.062, right=0.99, top=0.94, bottom=0.12, wspace=0.35, hspace=0.62)
    output_paths: list[Path] = []
    for suffix in [".png", ".pdf"]:
        path = UNDER10_FIGURE_STEM.with_suffix(suffix)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def update_may18_manifest_for_combined_fig6(under10: pd.DataFrame) -> None:
    if MAY18_MANIFEST_PATH.exists():
        manifest = json.loads(MAY18_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}
    may18_only = [label for label in UNDER10_COMPOUND_LABELS if label != "JP1302"]
    manifest.update(
        {
            "ic50_lt10uM_compounds": UNDER10_COMPOUND_LABELS,
            "may18_only_ic50_lt10uM_compounds": may18_only,
            "ic50_lt10uM_summary_csv": output_relative(UNDER10_SUMMARY_PATH),
            "ic50_lt10uM_figure_paths": [
                output_relative(UNDER10_FIGURE_STEM.with_suffix(".png")),
                output_relative(UNDER10_FIGURE_STEM.with_suffix(".pdf")),
            ],
            "ic50_lt10uM_current_fig6_note": (
                "Active Fig. 6 under-10 panel contains six reliable May 18 compounds plus JP1302. "
                "The May 18 all-18 outputs remain May 18-only provenance."
            ),
            "jp1302_source_workbook": project_relative(JP1302_XLSX),
            "jp1302_source_block": "Sheet1!BT14:BW24",
            "jp1302_ic50_uM": float(under10.loc[under10["compound_label"] == "JP1302", "ic50_uM"].iloc[0]),
        }
    )
    MAY18_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def records_for_json(df: pd.DataFrame) -> list[dict[str, object]]:
    cleaned = df.replace({np.nan: None})
    return cleaned.to_dict("records")


def write_workbook_payload(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    fit: pd.DataFrame,
    source_blocks: pd.DataFrame,
    screen: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    payload = {
        "output_xlsx": output_relative(COMBINED_XLSX_PATH),
        "sheets": {
            "README": [
                ["Field", "Value"],
                ["Description", "Clean combined CCK8 dose-response data from corrected May 18 selected blocks plus JP1302."],
                ["Generated", __import__("datetime").date.today().isoformat()],
                ["Main raw rows", int(len(raw))],
                ["Compounds", int(raw["compound_label"].nunique())],
                ["JP1302 source block", "Sheet1 row 14, concentration column BT, response columns BU:BW"],
                ["May 18 policy", "One selected block per compound; duplicate May 18 compounds use run #2."],
            ],
            "Raw_Long": records_for_json(raw),
            "Dose_Summary": records_for_json(summary),
            "Fit_Summary": records_for_json(fit),
            "Source_Blocks": records_for_json(source_blocks),
            "JP1302_50uM_Screen": records_for_json(screen),
            "Manifest": [{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value} for key, value in manifest.items()],
        },
    }
    WORKBOOK_PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_outputs(
    raw: pd.DataFrame,
    fit: pd.DataFrame,
    jp_fit: pd.DataFrame,
    under10: pd.DataFrame,
    source_blocks: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    if raw["compound_label"].nunique() != 19:
        raise AssertionError(f"Expected 19 compounds in combined raw data, found {raw['compound_label'].nunique()}")
    if len(raw) != 393:
        raise AssertionError(f"Expected 393 combined raw observations, found {len(raw)}")
    may18_raw = raw.loc[raw["compound_label"] != "JP1302"]
    if len(may18_raw) != 366:
        raise AssertionError(f"Expected 366 May 18 selected observations, found {len(may18_raw)}")
    jp_raw = raw.loc[raw["compound_label"] == "JP1302"]
    if len(jp_raw) != 27 or jp_raw["concentration_uM"].nunique() != 9:
        raise AssertionError("JP1302 should have 9 concentrations with triplicates (27 rows).")
    jp_block = jp_raw[
        ["source_sheet", "header_row", "concentration_col", "response_start_col", "response_end_col"]
    ].drop_duplicates()
    expected_jp_block = ("Sheet1", 14, 72, 73, 75)
    observed_jp_blocks = [tuple(row) for row in jp_block.to_numpy()]
    if observed_jp_blocks != [expected_jp_block]:
        raise AssertionError(f"JP1302 must come from Sheet1!BT14:BW24, observed {observed_jp_blocks}")
    if int(source_blocks["selected"].sum()) != 19:
        raise AssertionError("Expected 19 selected source blocks in combined source block table.")
    if len(jp_fit) != 1:
        raise AssertionError("Expected one JP1302 fit row.")
    if str(jp_fit["fit_status"].iloc[0]) != "reliable":
        raise AssertionError("JP1302 fit must be reliable.")
    jp_ic50 = float(jp_fit["ic50_uM"].iloc[0])
    if not math.isclose(jp_ic50, 1.3757696372, rel_tol=1e-6, abs_tol=1e-9):
        raise AssertionError(f"JP1302 IC50 expected near 1.3757696372 uM, found {jp_ic50}")
    reliable_under10 = fit.loc[(fit["fit_status"] == "reliable") & (fit["ic50_uM"] < 10), "compound_label"].tolist()
    if reliable_under10 != UNDER10_COMPOUND_LABELS:
        raise AssertionError(f"Reliable IC50 < 10 uM set mismatch: {reliable_under10}")
    if under10["compound_label"].tolist() != UNDER10_COMPOUND_LABELS:
        raise AssertionError(f"Under-10 figure subset mismatch: {under10['compound_label'].tolist()}")
    if len(under10) != 7:
        raise AssertionError(f"Expected 7 under-10 Fig. 6 compounds, found {len(under10)}")
    sb225002 = raw.loc[raw["compound_label"] == "SB225002"]
    sb_block = sb225002[
        ["source_sheet", "header_row", "concentration_col", "response_start_col", "response_end_col"]
    ].drop_duplicates()
    expected_sb_block = ("Sheet1", 47, 11, 12, 14)
    observed_sb_blocks = [tuple(row) for row in sb_block.to_numpy()]
    if observed_sb_blocks != [expected_sb_block]:
        raise AssertionError(f"SB225002 must use run #2 {expected_sb_block}, observed {observed_sb_blocks}")
    for path in [
        JP1302_RAW_PATH,
        JP1302_FIT_PATH,
        JP1302_SCREEN_PATH,
        COMBINED_RAW_PATH,
        COMBINED_DOSE_SUMMARY_PATH,
        COMBINED_FIT_PATH,
        COMBINED_SOURCE_BLOCKS_PATH,
        UNDER10_SUMMARY_PATH,
        CANONICAL_INPUT_CSV_PATH,
        WORKBOOK_PAYLOAD_PATH,
        MANIFEST_PATH,
        MAY18_MANIFEST_PATH,
        *output_paths,
    ]:
        if not path.exists() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing or empty output: {path}")


def run_analysis() -> dict[str, object]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    may18_raw, may18_source_blocks = prepare_may18_clean_raw()
    jp_raw, jp_source_blocks, jp_screen = parse_jp1302_raw(JP1302_XLSX)
    combined_raw = harmonize_raw(pd.concat([may18_raw, jp_raw], ignore_index=True))
    summary = dose_summary(combined_raw)
    fit = fit_for_workbook(combined_raw)
    jp_fit = fit.loc[fit["compound_label"] == "JP1302"].copy()
    under10 = select_under10_with_jp1302(fit)

    source_blocks = pd.concat([may18_source_blocks, jp_source_blocks], ignore_index=True, sort=False)
    source_blocks = source_blocks.sort_values(["compound_order", "run_index"]).reset_index(drop=True)

    jp_raw.to_csv(JP1302_RAW_PATH, index=False, encoding="utf-8-sig")
    jp_fit.to_csv(JP1302_FIT_PATH, index=False, encoding="utf-8-sig")
    jp_screen.to_csv(JP1302_SCREEN_PATH, index=False, encoding="utf-8-sig")
    combined_raw.to_csv(COMBINED_RAW_PATH, index=False, encoding="utf-8-sig")
    combined_raw.to_csv(CANONICAL_INPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(COMBINED_DOSE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    fit.to_csv(COMBINED_FIT_PATH, index=False, encoding="utf-8-sig")
    under10.to_csv(UNDER10_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    source_blocks.to_csv(COMBINED_SOURCE_BLOCKS_PATH, index=False, encoding="utf-8-sig")

    figure_paths = draw_under10_combined_figure(combined_raw, under10)
    manifest = {
        "jp1302_source_workbook": project_relative(JP1302_XLSX),
        "jp1302_source_block": "cck_jp1302.xlsx::Sheet1!BT14:BW24",
        "may18_source_workbook": project_relative(MAY18.RAW_XLSX),
        "combined_xlsx": output_relative(COMBINED_XLSX_PATH),
        "canonical_input_csv": output_relative(CANONICAL_INPUT_CSV_PATH),
        "combined_raw_csv": output_relative(COMBINED_RAW_PATH),
        "combined_fit_summary_csv": output_relative(COMBINED_FIT_PATH),
        "combined_source_blocks_csv": output_relative(COMBINED_SOURCE_BLOCKS_PATH),
        "ic50_lt10uM_summary_csv": output_relative(UNDER10_SUMMARY_PATH),
        "ic50_lt10uM_figure_paths": [output_relative(path) for path in figure_paths],
        "ic50_lt10uM_compounds": under10["compound_label"].tolist(),
        "ic50_lt10uM_note": (
            "Active Fig. 6 under-10 output contains six reliable May 18 compounds plus JP1302; "
            "May 18 all-18 outputs remain May 18-only provenance."
        ),
        "compound_count": int(combined_raw["compound_label"].nunique()),
        "raw_observation_count": int(len(combined_raw)),
        "may18_selected_observation_count": int(len(may18_raw)),
        "selected_source_block_count": int(source_blocks["selected"].sum()),
        "jp1302_observation_count": int(len(jp_raw)),
        "jp1302_fit_status": str(jp_fit["fit_status"].iloc[0]),
        "jp1302_ic50_uM": float(jp_fit["ic50_uM"].iloc[0]) if math.isfinite(float(jp_fit["ic50_uM"].iloc[0])) else None,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_may18_manifest_for_combined_fig6(under10)
    write_workbook_payload(combined_raw, summary, fit, source_blocks, jp_screen, manifest)
    validate_outputs(combined_raw, fit, jp_fit, under10, source_blocks, figure_paths)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JP1302-inclusive combined CCK-8 inputs and Fig. 6 under-10 panels.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "experimental",
        help="Directory for generated tables/figures/data. Defaults to results/experimental.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_output_root(args.output_root)
    manifest = run_analysis()
    print("JP1302 and combined CCK8 data analysis completed")
    print(f"Compounds: {manifest['compound_count']}")
    print(f"Raw observations: {manifest['raw_observation_count']}")
    print(f"May 18 selected raw observations: {manifest['may18_selected_observation_count']}")
    print(f"Selected source blocks: {manifest['selected_source_block_count']}")
    print(f"IC50 < 10 uM Fig. 6 compounds: {', '.join(manifest['ic50_lt10uM_compounds'])}")
    print(f"JP1302 fit status: {manifest['jp1302_fit_status']}")
    print(f"JP1302 IC50 (uM): {manifest['jp1302_ic50_uM']}")
    print("Outputs:")
    for path in [
        JP1302_RAW_PATH,
        JP1302_FIT_PATH,
        JP1302_SCREEN_PATH,
        COMBINED_RAW_PATH,
        CANONICAL_INPUT_CSV_PATH,
        COMBINED_DOSE_SUMMARY_PATH,
        COMBINED_FIT_PATH,
        UNDER10_SUMMARY_PATH,
        COMBINED_SOURCE_BLOCKS_PATH,
        WORKBOOK_PAYLOAD_PATH,
        MANIFEST_PATH,
        MAY18_MANIFEST_PATH,
        *map(resolve_output_path, manifest["ic50_lt10uM_figure_paths"]),
    ]:
        print(f"  {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
