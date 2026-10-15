#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PANEL_NAME = "MCF7 78 bioactive compounds Table S2"
PANEL_SLUG = "mcf7_78_bioactive_table_s2"
THRESHOLD_LOG2FC = 0.15


def repo_root_from_script() -> Path:
    return ARTICLE_ROOT


def package_dir_from_script() -> Path:
    return ARTICLE_ROOT


def norm_accession(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\xa0", " ").upper()


def norm_compound(value: object) -> str:
    return str(value).strip().replace("\xa0", " ")


def norm_label_key(value: object) -> str:
    text = str(value).strip().replace("\xa0", " ").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def compound_from_sample_column(column: object) -> str:
    text = str(column).strip()
    match = re.match(r"^(.*)_R(\d+)$", text)
    if not match:
        raise ValueError(f"Sample column does not end in _R<replicate>: {column}")
    return match.group(1)


def load_module(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_pipeline_path(root: Path) -> Path:
    pipeline_dir = root / 'src/deepheal/external'
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    return pipeline_dir


def protein_id_candidates(value: object) -> list[str]:
    text = str(value)
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
        exact = feature_norm_to_exact[matches[0]]
        exact_matches = ";".join(feature_norm_to_exact[item] for item in matches)
        return exact, True, exact_matches
    return (candidates[0] if candidates else ""), False, ""


def read_table_s2(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Table S2", header=1, engine="openpyxl")
    required = {"Protein", "Gene"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Table S2 is missing required columns: {missing}")
    return df


def build_average_log2ratio_table(source: pd.DataFrame, sample_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    compounds = [compound_from_sample_column(col) for col in sample_cols]
    replicate_counts = Counter(compounds)
    if set(replicate_counts.values()) != {2}:
        raise ValueError(f"Expected exactly two replicates per compound, observed {dict(replicate_counts)}")

    values = source[sample_cols].apply(pd.to_numeric, errors="coerce")
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Table S2 contains no finite numeric sample values")
    if np.any(finite <= 0):
        raise ValueError("Table S2 contains non-positive ratios; log2 conversion would be invalid")

    avg = pd.DataFrame(index=source.index)
    valid_counts = pd.DataFrame(index=source.index)
    for compound in replicate_counts:
        cols = [col for col in sample_cols if compound_from_sample_column(col) == compound]
        log_values = np.log2(values[cols].where(values[cols] > 0))
        avg[compound] = log_values.mean(axis=1, skipna=True)
        valid_counts[compound] = log_values.notna().sum(axis=1)
    return avg, valid_counts


def load_feature_order(root: Path) -> list[str]:
    path = root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv"
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(col).strip().replace("\xa0", " ") for col in cols if col != "Sample_ID"]


def build_label_match(root: Path, compounds: list[str]) -> pd.DataFrame:
    labels_path = root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv"
    labels = pd.read_csv(labels_path)
    labels["label_key"] = labels["Drug_Name"].map(norm_label_key)
    by_key = labels.drop_duplicates("label_key").set_index("label_key")
    rows: list[dict[str, object]] = []
    for compound in compounds:
        key = norm_label_key(compound)
        if key in by_key.index:
            row = by_key.loc[key]
            rows.append(
                {
                    "Compound": compound,
                    "compound_norm": norm_accession(compound),
                    "deepheal_label_match": True,
                    "matched_deepheal_drug_name": row["Drug_Name"],
                    "matched_deepheal_drug_class": row["Drug_Class"],
                    "matched_deepheal_sample_id": row["Sample_ID"],
                    "matched_deepheal_cluster": int(row["Cluster"]),
                }
            )
        else:
            rows.append(
                {
                    "Compound": compound,
                    "compound_norm": norm_accession(compound),
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
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_dir = add_pipeline_path(root)
    external_mod = load_module(
        "deepheal_external_panel_projection_mcf7",
        pipeline_dir / 'panel_projection.py',
    )
    helper_mod = load_module("deepheal_projection_helper_mcf7", pipeline_dir / 'projection_utils.py')
    supcon_base = load_module("deepheal_supcon_base_mcf7", external_mod.SUPCON_BASE_SCRIPT)
    average_mod = load_module("deepheal_average_mod_mcf7", external_mod.AVERAGE_ENSEMBLE_SCRIPT)

    supcon_pca = helper_mod.load_pickle(external_mod.SUPCON_DIR / "supcon_pca.pkl")
    average_pca = helper_mod.load_pickle(external_mod.AVG_DIR / "average_pca.pkl")
    feature_cols = [col for col in aligned_df.columns if col not in {"Compound", "compound_norm"}]
    raw_values = aligned_df[feature_cols].to_numpy(dtype=np.float32)

    pca_values = supcon_pca.transform(raw_values).astype(np.float32)[:, :200]
    pca_df = pd.concat(
        [
            aligned_df[["Compound", "compound_norm"]].reset_index(drop=True),
            pd.DataFrame(pca_values, columns=[f"pca_{idx:03d}" for idx in range(pca_values.shape[1])]),
        ],
        axis=1,
    )

    supcon_query_df, average_query_df = external_mod.build_external_branch_frames(aligned_df, supcon_pca, average_pca)
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

    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pca_df.to_csv(table_dir / f"{PANEL_SLUG}_pca_200_features.csv", index=False, encoding="utf-8")
    supcon_embed_df.to_csv(table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv", index=False, encoding="utf-8")
    average_embed_df.to_csv(table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv", index=False, encoding="utf-8")
    return pca_df, supcon_embed_df, average_embed_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Table S2 MCF7 features for DeepHeal four-space prediction.")
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--package-dir", type=Path, default=package_dir_from_script())
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=package_dir_from_script() / "data" / "1-s2.0-S2095177923002101-mmc3.xlsx",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = args.package_dir.resolve()
    xlsx_path = args.input_xlsx.resolve()
    processed_dir = package_dir / "data" / "processed"
    panel_table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    panel_table_dir.mkdir(parents=True, exist_ok=True)

    source = read_table_s2(xlsx_path)
    sample_cols = [col for col in source.columns if col not in {"Protein", "Gene"}]
    compounds = list(dict.fromkeys(compound_from_sample_column(col) for col in sample_cols))
    avg_values, valid_counts = build_average_log2ratio_table(source, sample_cols)

    feature_order = load_feature_order(root)
    feature_norm_to_exact = {norm_accession(col): col for col in feature_order}
    selected = [select_feature_id(value, feature_norm_to_exact) for value in source["Protein"]]
    selected_ids = [item[0] for item in selected]
    matched_flags = [item[1] for item in selected]
    matched_lists = [item[2] for item in selected]

    mapping = pd.DataFrame(
        {
            "source_row": np.arange(1, len(source) + 1, dtype=int),
            "Protein": source["Protein"],
            "Gene": source["Gene"],
            "selected_uniprot_id": selected_ids,
            "selected_id_matches_deepheal_feature": matched_flags,
            "matching_deepheal_feature_ids": matched_lists,
        }
    )
    mapping.to_csv(processed_dir / "mcf7_table_s2_uniprot_selection_map.csv", index=False, encoding="utf-8")

    avg_out = pd.concat([pd.DataFrame({"protein_id": selected_ids, "Gene": source["Gene"]}), avg_values], axis=1)
    avg_out.to_csv(processed_dir / "mcf7_table_s2_78_compounds_avg_log2ratio_for_deepheal.csv", index=False, encoding="utf-8")
    counts_out = pd.concat([pd.DataFrame({"protein_id": selected_ids, "Gene": source["Gene"]}), valid_counts], axis=1)
    counts_out.to_csv(processed_dir / "mcf7_table_s2_valid_replicate_counts.csv", index=False, encoding="utf-8")

    value_with_ids = avg_values.copy()
    value_with_ids["selected_uniprot_id"] = selected_ids
    value_with_ids = value_with_ids.loc[[bool(flag) for flag in matched_flags]].copy()
    collapsed = value_with_ids.groupby("selected_uniprot_id", sort=False)[compounds].mean()
    aligned = collapsed.reindex(feature_order).apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    aligned.index.name = "Compound"
    aligned = aligned.mask(aligned.abs() <= THRESHOLD_LOG2FC, 0.0)
    aligned["compound_norm"] = [norm_accession(compound) for compound in aligned.index]
    aligned_df = aligned.reset_index()
    aligned_df.to_csv(panel_table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv", index=False, encoding="utf-8")

    shared_features = int(collapsed.index.nunique())
    coverage = pd.DataFrame(
        [
            {
                "panel_name": PANEL_NAME,
                "panel_slug": PANEL_SLUG,
                "n_compounds": int(len(compounds)),
                "n_sample_columns": int(len(sample_cols)),
                "n_input_proteins": int(len(source)),
                "n_hct116_features": int(len(feature_order)),
                "n_shared_features": shared_features,
                "panel_feature_coverage_pct": float(shared_features / max(1, len(source)) * 100.0),
                "hct116_feature_overlap_pct": float(shared_features / max(1, len(feature_order)) * 100.0),
                "threshold_log2fc": float(THRESHOLD_LOG2FC),
                "source_kind": "abundance_ratio_converted_to_log2_ratio",
            }
        ]
    )
    coverage.to_csv(panel_table_dir / f"{PANEL_SLUG}_coverage_summary.csv", index=False, encoding="utf-8")

    label_match = build_label_match(root, compounds)
    label_match.to_csv(processed_dir / "mcf7_table_s2_deepheal_label_match.csv", index=False, encoding="utf-8")

    pca_df, supcon_embed_df, average_embed_df = write_projection_inputs(
        aligned_df=aligned_df,
        root=root,
        package_dir=package_dir,
        batch_size=args.batch_size,
    )

    finite_values = source[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    summary = {
        "analysis": "Prepare MCF7 Table S2 abundance ratios for DeepHeal four-space prediction",
        "source_xlsx": str(xlsx_path),
        "panel_name": PANEL_NAME,
        "panel_slug": PANEL_SLUG,
        "table_s2_shape": [int(source.shape[0]), int(source.shape[1])],
        "n_compounds": int(len(compounds)),
        "replicate_count_histogram": {str(k): int(v) for k, v in sorted(Counter(Counter(compound_from_sample_column(c) for c in sample_cols).values()).items())},
        "ratio_value_min": float(np.min(finite_values)),
        "ratio_value_median": float(np.median(finite_values)),
        "ratio_value_max": float(np.max(finite_values)),
        "feature_counts": {
            "full_protein": int(len(feature_order)),
            "pca_200": int(len([col for col in pca_df.columns if col.startswith("pca_")])),
            "supcon_latent": int(len([col for col in supcon_embed_df.columns if col.startswith("latent_")])),
            "average_latent": int(len([col for col in average_embed_df.columns if col.startswith("latent_")])),
        },
        "coverage": coverage.iloc[0].to_dict(),
        "deepheal_direct_label_match_counts": label_match["matched_deepheal_drug_class"].replace("", "Unmatched").value_counts().to_dict(),
        "outputs": {
            "avg_log2ratio_csv": str(processed_dir / "mcf7_table_s2_78_compounds_avg_log2ratio_for_deepheal.csv"),
            "uniprot_selection_map_csv": str(processed_dir / "mcf7_table_s2_uniprot_selection_map.csv"),
            "valid_replicate_counts_csv": str(processed_dir / "mcf7_table_s2_valid_replicate_counts.csv"),
            "label_match_csv": str(processed_dir / "mcf7_table_s2_deepheal_label_match.csv"),
            "aligned_threshold_csv": str(panel_table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv"),
            "coverage_summary_csv": str(panel_table_dir / f"{PANEL_SLUG}_coverage_summary.csv"),
            "pca_200_csv": str(panel_table_dir / f"{PANEL_SLUG}_pca_200_features.csv"),
            "supcon_projected_embeddings_csv": str(panel_table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv"),
            "average_projected_embeddings_csv": str(panel_table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv"),
        },
    }
    summary_path = processed_dir / "mcf7_table_s2_feature_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
