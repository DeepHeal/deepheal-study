#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SOURCE_XLSX_NAME = "41589_2020_572_MOESM7_ESM.xlsx"
PANEL_NAME = "Ruprecht Nat Chem Biol 2020 Supplementary Data 1 lung cancer cell lines"
PANEL_SLUG = "ruprecht_2020_suppdata1_lung_cell_lines"
THRESHOLD_LOG2FC = 0.15
CELL_LINE_SHEETS = {
    "A549": "A549_dataset",
    "Calu1": "Calu1_dataset",
    "Calu6": "Calu6_dataset",
    "2030": "2030_dataset",
    "2122": "2122_dataset",
}
EXPECTED_PROFILE_COUNTS = {
    "A549": 53,
    "Calu1": 50,
    "Calu6": 53,
    "2030": 53,
    "2122": 53,
}


def repo_root_from_script() -> Path:
    return ARTICLE_ROOT


def package_dir_from_script() -> Path:
    return ARTICLE_ROOT


def add_pipeline_path(root: Path) -> Path:
    pipeline_dir = root / 'src/deepheal/external'
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    return pipeline_dir


def load_module(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\xa0", " ")


def norm_accession(value: object) -> str:
    return norm_text(value).upper()


def norm_label_key(value: object) -> str:
    text = norm_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def protein_id_candidates(value: object) -> list[str]:
    text = norm_text(value)
    candidates: list[str] = []
    for part in re.split(r"[;|]", text):
        raw = norm_accession(part)
        if not raw:
            continue
        swissprot = re.match(r"^(?:SP|TR)_([^_]+)_", raw)
        if swissprot:
            candidates.append(swissprot.group(1))
        candidates.append(raw)
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def select_feature_id(value: object, feature_norm_to_exact: dict[str, str]) -> tuple[str, bool, str]:
    candidates = protein_id_candidates(value)
    matches = [candidate for candidate in candidates if candidate in feature_norm_to_exact]
    if matches:
        exact_matches = ";".join(feature_norm_to_exact[item] for item in matches)
        return feature_norm_to_exact[matches[0]], True, exact_matches
    return (candidates[0] if candidates else ""), False, ""


def inhibitor_key(value: object) -> str:
    text = norm_text(value)
    return re.sub(r"^i", "", text, count=1)


def load_feature_order(root: Path) -> list[str]:
    path = root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv"
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [norm_text(col) for col in columns if col != "Sample_ID"]


def load_inhibitor_map(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="Inhibitors_used", engine="openpyxl")
    required = {"Compound Name", "Inhibitor"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Inhibitors_used is missing required columns: {missing}")
    out = raw[["Compound Name", "Inhibitor"]].copy()
    out["Compound"] = out["Compound Name"].map(norm_text)
    out["inhibitor"] = out["Inhibitor"].map(norm_text)
    out["inhibitor_key"] = out["inhibitor"].map(inhibitor_key)
    if out["inhibitor_key"].duplicated().any():
        dupes = out.loc[out["inhibitor_key"].duplicated(), "inhibitor_key"].tolist()
        raise ValueError(f"Duplicate inhibitor keys in Inhibitors_used: {dupes[:5]}")
    return out[["inhibitor", "inhibitor_key", "Compound"]]


def log2fc_columns(columns: list[object]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for column in columns:
        text = norm_text(column)
        if not text.startswith("Log2FC "):
            continue
        key = text.removeprefix("Log2FC ").strip()
        if re.fullmatch(r"EMPTY(?:\.\d+)?", key.upper()):
            continue
        found.append((text, key))
    return found


def read_cell_line_sheet(
    xlsx_path: Path,
    cell_line: str,
    sheet_name: str,
    inhibitor_lookup: pd.DataFrame,
    feature_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    if "Majority protein IDs" not in source.columns:
        raise ValueError(f"{sheet_name} is missing Majority protein IDs")
    if "Gene names" not in source.columns:
        raise ValueError(f"{sheet_name} is missing Gene names")

    log_cols = log2fc_columns(source.columns.tolist())
    if len(log_cols) != EXPECTED_PROFILE_COUNTS[cell_line]:
        raise AssertionError(
            f"{cell_line} expected {EXPECTED_PROFILE_COUNTS[cell_line]} inhibitor profiles, observed {len(log_cols)}"
        )

    lookup = inhibitor_lookup.set_index("inhibitor_key")
    missing = sorted(set(key for _, key in log_cols) - set(lookup.index))
    if missing:
        raise ValueError(f"{cell_line} Log2FC inhibitor keys missing from Inhibitors_used: {missing}")

    feature_norm_to_exact = {norm_accession(col): col for col in feature_order}
    selected = [select_feature_id(value, feature_norm_to_exact) for value in source["Majority protein IDs"]]
    selected_ids = [item[0] for item in selected]
    matched_flags = [item[1] for item in selected]
    matched_lists = [item[2] for item in selected]

    mapping = pd.DataFrame(
        {
            "cell_line": cell_line,
            "source_row": np.arange(1, len(source) + 1, dtype=int),
            "Majority protein IDs": source["Majority protein IDs"],
            "Gene names": source["Gene names"],
            "selected_uniprot_id": selected_ids,
            "selected_id_matches_deepheal_feature": matched_flags,
            "matching_deepheal_feature_ids": matched_lists,
        }
    )

    profile_values = pd.DataFrame(index=source.index)
    profile_rows: list[dict[str, object]] = []
    for source_column, key in log_cols:
        compound = norm_text(lookup.loc[key, "Compound"])
        profile_id = f"{cell_line}__{key}"
        profile_values[profile_id] = pd.to_numeric(source[source_column], errors="coerce")
        profile_rows.append(
            {
                "profile_id": profile_id,
                "cell_line": cell_line,
                "inhibitor_key": key,
                "inhibitor": f"i{key}",
                "Compound": compound,
                "source_sheet": sheet_name,
                "source_log2fc_column": source_column,
            }
        )

    matched_values = profile_values.copy()
    matched_values["selected_uniprot_id"] = selected_ids
    matched_values = matched_values.loc[[bool(flag) for flag in matched_flags]].copy()
    collapsed = matched_values.groupby("selected_uniprot_id", sort=False)[profile_values.columns.tolist()].mean()
    aligned = collapsed.reindex(feature_order).apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    aligned.index.name = "profile_id"
    aligned = aligned.mask(aligned.abs() <= THRESHOLD_LOG2FC, 0.0)

    profile_meta = pd.DataFrame(profile_rows)
    aligned = profile_meta.merge(aligned.reset_index(), on="profile_id", how="inner")
    aligned["compound_norm"] = aligned["Compound"].map(norm_accession)
    ordered_meta = [
        "profile_id",
        "cell_line",
        "inhibitor_key",
        "inhibitor",
        "Compound",
        "compound_norm",
        "source_sheet",
        "source_log2fc_column",
    ]
    aligned = aligned[ordered_meta + feature_order]

    coverage = pd.DataFrame(
        [
            {
                "panel_name": PANEL_NAME,
                "panel_slug": PANEL_SLUG,
                "cell_line": cell_line,
                "source_sheet": sheet_name,
                "n_profiles": int(len(profile_meta)),
                "n_input_proteins": int(len(source)),
                "n_hct116_features": int(len(feature_order)),
                "n_rows_with_selected_id_matching_deepheal_feature": int(sum(matched_flags)),
                "n_shared_features": int(collapsed.index.nunique()),
                "selected_id_match_pct": float(sum(matched_flags) / max(1, len(source)) * 100.0),
                "hct116_feature_overlap_pct": float(collapsed.index.nunique() / max(1, len(feature_order)) * 100.0),
                "threshold_log2fc": float(THRESHOLD_LOG2FC),
            }
        ]
    )
    return aligned, mapping, coverage


def build_label_match(root: Path, profiles: pd.DataFrame) -> pd.DataFrame:
    labels_path = root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv"
    labels = pd.read_csv(labels_path)
    labels["label_key"] = labels["Drug_Name"].map(norm_label_key)
    by_key = labels.drop_duplicates("label_key").set_index("label_key")

    rows: list[dict[str, object]] = []
    for _, row in profiles.iterrows():
        compound = norm_text(row["Compound"])
        key = norm_label_key(compound)
        if key in by_key.index:
            match = by_key.loc[key]
            rows.append(
                {
                    "profile_id": row["profile_id"],
                    "cell_line": row["cell_line"],
                    "inhibitor_key": row["inhibitor_key"],
                    "Compound": compound,
                    "compound_norm": row["compound_norm"],
                    "deepheal_label_match": True,
                    "matched_deepheal_drug_name": match["Drug_Name"],
                    "matched_deepheal_drug_class": match["Drug_Class"],
                    "matched_deepheal_sample_id": match["Sample_ID"],
                    "matched_deepheal_cluster": int(match["Cluster"]),
                }
            )
        else:
            rows.append(
                {
                    "profile_id": row["profile_id"],
                    "cell_line": row["cell_line"],
                    "inhibitor_key": row["inhibitor_key"],
                    "Compound": compound,
                    "compound_norm": row["compound_norm"],
                    "deepheal_label_match": False,
                    "matched_deepheal_drug_name": "",
                    "matched_deepheal_drug_class": "",
                    "matched_deepheal_sample_id": "",
                    "matched_deepheal_cluster": np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_projection_inputs(
    aligned_df: pd.DataFrame,
    root: Path,
    package_dir: Path,
    feature_order: list[str],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_dir = add_pipeline_path(root)
    external_mod = load_module(
        "deepheal_external_panel_projection_lung_cell_lines",
        pipeline_dir / 'panel_projection.py',
    )
    helper_mod = load_module(
        "deepheal_projection_helper_lung_cell_lines",
        pipeline_dir / 'projection_utils.py',
    )
    supcon_base = load_module("deepheal_supcon_base_lung_cell_lines", external_mod.SUPCON_BASE_SCRIPT)
    average_mod = load_module("deepheal_average_mod_lung_cell_lines", external_mod.AVERAGE_ENSEMBLE_SCRIPT)

    supcon_pca = helper_mod.load_pickle(external_mod.SUPCON_DIR / "supcon_pca.pkl")
    average_pca = helper_mod.load_pickle(external_mod.AVG_DIR / "average_pca.pkl")
    raw_values = aligned_df[feature_order].to_numpy(dtype=np.float32)

    pca_values = supcon_pca.transform(raw_values).astype(np.float32)[:, :200]
    pca_df = pd.concat(
        [
            aligned_df[
                [
                    "profile_id",
                    "cell_line",
                    "inhibitor_key",
                    "inhibitor",
                    "Compound",
                    "compound_norm",
                    "source_sheet",
                    "source_log2fc_column",
                ]
            ].reset_index(drop=True),
            pd.DataFrame(pca_values, columns=[f"pca_{idx:03d}" for idx in range(pca_values.shape[1])]),
        ],
        axis=1,
    )

    projection_input = pd.concat(
        [
            pd.DataFrame(
                {
                    "Compound": aligned_df["profile_id"].to_numpy(),
                    "compound_norm": aligned_df["profile_id"].map(norm_accession).to_numpy(),
                }
            ),
            aligned_df[feature_order].reset_index(drop=True),
        ],
        axis=1,
    )
    supcon_query_df, average_query_df = external_mod.build_external_branch_frames(
        projection_input,
        supcon_pca,
        average_pca,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] projection device={device}")
    supcon_embed_df = external_mod.aggregate_from_checkpoints(
        "supcon",
        supcon_query_df,
        external_mod.SOURCE_MODEL_CKPT_DIR / "supcon",
        external_mod.SUPCON_DIR / "supcon_ensemble_node_labels.csv",
        helper_mod,
        supcon_base,
        average_mod,
        device,
        batch_size,
    )
    average_embed_df = external_mod.aggregate_from_checkpoints(
        "average",
        average_query_df,
        external_mod.SOURCE_MODEL_CKPT_DIR / "average",
        external_mod.AVG_DIR / "average_ensemble_node_labels.csv",
        helper_mod,
        supcon_base,
        average_mod,
        device,
        batch_size,
    )

    meta_cols = [
        "profile_id",
        "cell_line",
        "inhibitor_key",
        "inhibitor",
        "Compound",
        "compound_norm",
        "source_sheet",
        "source_log2fc_column",
    ]
    projection_meta = aligned_df[meta_cols].reset_index(drop=True)
    latent_cols = [col for col in supcon_embed_df.columns if col.startswith("latent_")]
    supcon_embed_df = pd.concat([projection_meta, supcon_embed_df[latent_cols].reset_index(drop=True)], axis=1)
    latent_cols = [col for col in average_embed_df.columns if col.startswith("latent_")]
    average_embed_df = pd.concat([projection_meta, average_embed_df[latent_cols].reset_index(drop=True)], axis=1)

    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pca_df.to_csv(table_dir / f"{PANEL_SLUG}_pca_200_features.csv", index=False, encoding="utf-8")
    supcon_embed_df.to_csv(
        table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv",
        index=False,
        encoding="utf-8",
    )
    average_embed_df.to_csv(
        table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv",
        index=False,
        encoding="utf-8",
    )
    return pca_df, supcon_embed_df, average_embed_df


def verify_prepared_outputs(
    aligned_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    supcon_df: pd.DataFrame,
    average_df: pd.DataFrame,
    feature_order: list[str],
) -> dict[str, object]:
    profile_counts = aligned_df["cell_line"].value_counts().reindex(EXPECTED_PROFILE_COUNTS).to_dict()
    profile_counts = {str(key): int(value) for key, value in profile_counts.items()}
    if profile_counts != EXPECTED_PROFILE_COUNTS:
        raise AssertionError(f"Unexpected profile counts: {profile_counts}")
    if len(aligned_df) != 262:
        raise AssertionError(f"Expected 262 profiles, observed {len(aligned_df)}")

    feature_counts = {
        "full_protein": len(feature_order),
        "pca_200": len([col for col in pca_df.columns if col.startswith("pca_")]),
        "supcon_latent": len([col for col in supcon_df.columns if col.startswith("latent_")]),
        "average_latent": len([col for col in average_df.columns if col.startswith("latent_")]),
    }
    expected_feature_counts = {
        "full_protein": 9960,
        "pca_200": 200,
        "supcon_latent": 257,
        "average_latent": 194,
    }
    if feature_counts != expected_feature_counts:
        raise AssertionError(f"Unexpected feature counts: {feature_counts}")
    if aligned_df["profile_id"].duplicated().any():
        raise AssertionError("Duplicate profile_id values detected")
    return {
        "profile_counts_by_cell_line": profile_counts,
        "total_profiles": int(len(aligned_df)),
        "feature_counts": {key: int(value) for key, value in feature_counts.items()},
        "profile_ids_unique": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Ruprecht 2020 Supplementary Data 1 lung cell-line profiles for DeepHeal prediction."
    )
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--package-dir", type=Path, default=package_dir_from_script())
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=package_dir_from_script() / "data" / SOURCE_XLSX_NAME,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = args.package_dir.resolve()
    xlsx_path = args.input_xlsx.resolve()
    processed_dir = package_dir / "data" / "processed"
    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    feature_order = load_feature_order(root)
    inhibitor_lookup = load_inhibitor_map(xlsx_path)

    aligned_frames: list[pd.DataFrame] = []
    mapping_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for cell_line, sheet_name in CELL_LINE_SHEETS.items():
        aligned, mapping, coverage = read_cell_line_sheet(
            xlsx_path=xlsx_path,
            cell_line=cell_line,
            sheet_name=sheet_name,
            inhibitor_lookup=inhibitor_lookup,
            feature_order=feature_order,
        )
        aligned_frames.append(aligned)
        mapping_frames.append(mapping)
        coverage_frames.append(coverage)

    aligned_df = pd.concat(aligned_frames, ignore_index=True)
    mapping_df = pd.concat(mapping_frames, ignore_index=True)
    coverage_df = pd.concat(coverage_frames, ignore_index=True)
    profile_manifest = aligned_df[
        [
            "profile_id",
            "cell_line",
            "inhibitor_key",
            "inhibitor",
            "Compound",
            "compound_norm",
            "source_sheet",
            "source_log2fc_column",
        ]
    ].copy()
    label_match = build_label_match(root, profile_manifest)

    aligned_df.to_csv(table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv", index=False, encoding="utf-8")
    mapping_df.to_csv(processed_dir / "suppdata1_lung_cell_line_uniprot_selection_map.csv", index=False, encoding="utf-8")
    profile_manifest.to_csv(processed_dir / "suppdata1_lung_cell_line_profile_manifest.csv", index=False, encoding="utf-8")
    label_match.to_csv(processed_dir / "suppdata1_lung_cell_line_deepheal_label_match.csv", index=False, encoding="utf-8")
    coverage_df.to_csv(table_dir / f"{PANEL_SLUG}_coverage_summary.csv", index=False, encoding="utf-8")

    pca_df, supcon_df, average_df = write_projection_inputs(
        aligned_df=aligned_df,
        root=root,
        package_dir=package_dir,
        feature_order=feature_order,
        batch_size=args.batch_size,
    )
    verification = verify_prepared_outputs(aligned_df, pca_df, supcon_df, average_df, feature_order)

    missing_by_cell_line = {}
    inhibitor_keys = set(inhibitor_lookup["inhibitor_key"])
    for cell_line, sheet_name in CELL_LINE_SHEETS.items():
        source = pd.read_excel(xlsx_path, sheet_name=sheet_name, nrows=0, engine="openpyxl")
        observed = {key for _, key in log2fc_columns(source.columns.tolist())}
        missing_by_cell_line[cell_line] = sorted(inhibitor_keys - observed)

    summary = {
        "analysis": "Prepare Ruprecht 2020 Supplementary Data 1 lung cancer cell-line Log2FC profiles for DeepHeal four-space prediction",
        "source_article": "https://www.nature.com/articles/s41589-020-0572-3",
        "source_xlsx": str(xlsx_path),
        "panel_name": PANEL_NAME,
        "panel_slug": PANEL_SLUG,
        "cell_line_sheets": CELL_LINE_SHEETS,
        "expected_profile_counts": EXPECTED_PROFILE_COUNTS,
        "missing_inhibitor_profiles_by_cell_line": missing_by_cell_line,
        "threshold_log2fc": float(THRESHOLD_LOG2FC),
        "verification": verification,
        "deepheal_direct_label_match_counts": label_match["matched_deepheal_drug_class"]
        .replace("", "Unmatched")
        .value_counts()
        .to_dict(),
        "outputs": {
            "aligned_threshold_csv": str(table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv"),
            "coverage_summary_csv": str(table_dir / f"{PANEL_SLUG}_coverage_summary.csv"),
            "pca_200_csv": str(table_dir / f"{PANEL_SLUG}_pca_200_features.csv"),
            "supcon_projected_embeddings_csv": str(table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv"),
            "average_projected_embeddings_csv": str(table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv"),
            "profile_manifest_csv": str(processed_dir / "suppdata1_lung_cell_line_profile_manifest.csv"),
            "uniprot_selection_map_csv": str(processed_dir / "suppdata1_lung_cell_line_uniprot_selection_map.csv"),
            "label_match_csv": str(processed_dir / "suppdata1_lung_cell_line_deepheal_label_match.csv"),
        },
    }
    summary_path = processed_dir / "suppdata1_lung_cell_line_feature_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
