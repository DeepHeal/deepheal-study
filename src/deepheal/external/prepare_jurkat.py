#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import html
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch


PANEL_NAME = "Jurkat decryptE 144-drug proteome 10000 nM"
PANEL_SLUG = "jurkat_decrypte_144_10000nm"
MASSIVE_ACCESSION = "MSV000093659"
PXD_ACCESSION = "PXD047799"
MASSIVE_DOWNLOAD_ENDPOINT = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile"
THRESHOLD_LOG2FC = 0.15
REQUIRED_TXT_COLUMNS = [
    ".pred_class",
    "identifier",
    "drug",
    "gene",
    "IC50",
    "stdIC50",
    "r2",
    "AUC",
    "Normalized_Intensity_0",
    "Normalized_Intensity_10000",
    "Description",
    "Gene names",
]


def repo_root_from_script() -> Path:
    return ARTICLE_ROOT


def package_dir_from_script() -> Path:
    return ARTICLE_ROOT


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\xa0", " ")


def norm_accession(value: object) -> str:
    return norm_text(value).upper()


def norm_compound(value: object) -> str:
    return norm_text(value)


def norm_label_key(value: object) -> str:
    text = norm_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def first_gene_symbol(value: object) -> str:
    text = norm_text(value)
    if not text:
        return ""
    return re.split(r"[;,\s]+", text)[0].upper()


def load_module(module_name: str, path: Path):
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


def extract_massive_index_from_html(html_path: Path) -> pd.DataFrame:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"var dataset_files = (\{.*?\});\s*function init", text, flags=re.S)
    if not match:
        match = re.search(r"var dataset_files = (\{.*?\});", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find embedded MassIVE dataset_files JSON in {html_path}")
    payload = json.loads(match.group(1))
    return pd.DataFrame(payload["row_data"])


def build_massive_txt_indexes(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    html_path = data_dir / "massive_dataset_files_page.html"
    if not html_path.exists():
        raise FileNotFoundError(
            f"Missing {html_path}. Download it first from "
            "https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=70cd42693c7f4d4da2cf466a0832e87d"
        )
    file_index = extract_massive_index_from_html(html_path)
    file_index.to_csv(data_dir / "massive_file_index.csv", index=False, encoding="utf-8")

    for col in ["file_descriptor", "name", "relative_path", "collection", "type"]:
        file_index[col] = file_index[col].astype(str)

    jurkat_txt = file_index[
        file_index["file_descriptor"].str.contains(
            r"Dose response data - Jurkat proteome/.*/TXTs_Classified/.*\.txt$",
            case=False,
            regex=True,
        )
    ].copy()
    jurkat_txt["drug"] = jurkat_txt["file_descriptor"].str.extract(
        r"Dose response data - Jurkat proteome/([^/]+)/TXTs_Classified/"
    )
    jurkat_txt["txt_class"] = jurkat_txt["name"].str.replace(r"^.*?_", "", regex=True).str.replace(
        ".txt", "", regex=False
    )
    jurkat_txt.to_csv(data_dir / "massive_jurkat_proteome_txt_files.csv", index=False, encoding="utf-8")

    primary_all = jurkat_txt[
        jurkat_txt["relative_path"].str.startswith("other/Dose response data - Jurkat proteome/")
        & jurkat_txt["name"].str.endswith("_ALL.txt")
    ].copy()
    primary_all = primary_all.sort_values(["drug", "file_descriptor"]).reset_index(drop=True)
    primary_all.to_csv(data_dir / "massive_jurkat_proteome_all_txt_files.csv", index=False, encoding="utf-8")
    if len(primary_all) != 144:
        raise AssertionError(f"Expected 144 primary Jurkat _ALL.txt files, observed {len(primary_all)}")
    return jurkat_txt, primary_all


def is_valid_txt(path: Path, min_bytes: int = 1024) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with open(path, "rb") as handle:
        head = handle.read(512)
    lower = head.lower()
    if b"<html" in lower or b"<!doctype" in lower or b"too many requests" in lower:
        return False
    return b".pred_class\tidentifier\tdrug\tgene" in head


def download_one(
    file_descriptor: str,
    output_path: Path,
    max_time: int,
    attempts: int,
    initial_backoff_seconds: float,
) -> tuple[bool, str]:
    url = f"{MASSIVE_DOWNLOAD_ENDPOINT}?file={quote(file_descriptor, safe='')}&forceDownload=true"
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_message = ""
    for attempt in range(1, attempts + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        command = [
            "curl.exe",
            "-L",
            "-k",
            "--ssl-no-revoke",
            "--connect-timeout",
            "30",
            "--max-time",
            str(max_time),
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            url,
            "-o",
            str(tmp_path),
        ]
        proc = subprocess.run(command, text=True, capture_output=True, errors="replace")
        if proc.returncode != 0:
            last_message = proc.stderr[-1000:]
            if tmp_path.exists():
                tmp_path.unlink()
        elif is_valid_txt(tmp_path):
            tmp_path.replace(output_path)
            return True, "ok"
        else:
            body = tmp_path.read_text(encoding="utf-8", errors="replace")[:300]
            tmp_path.unlink(missing_ok=True)
            last_message = f"Downloaded file failed TXT validation: {body!r}"

        lower_message = last_message.lower()
        is_transient = "429" in lower_message or "too many requests" in lower_message or proc.returncode != 0
        if attempt < attempts and is_transient:
            sleep_seconds = min(initial_backoff_seconds * (2 ** (attempt - 1)), 300.0)
            print(f"[BACKOFF] attempt {attempt}/{attempts} failed; sleeping {sleep_seconds:.0f}s")
            time.sleep(sleep_seconds)
            continue
        break
    return False, last_message


def download_primary_txts(
    primary_all: pd.DataFrame,
    raw_dir: Path,
    force: bool,
    max_time: int,
    attempts: int,
    initial_backoff_seconds: float,
    request_delay_seconds: float,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for idx, row in primary_all.reset_index(drop=True).iterrows():
        safe_name = str(row["name"])
        output_path = raw_dir / safe_name
        expected_size = int(row["size"])
        status = "skipped_existing"
        message = "existing valid TXT"
        if force or not is_valid_txt(output_path):
            print(f"[DOWNLOAD] {idx + 1:03d}/{len(primary_all)} {safe_name}")
            ok, message = download_one(
                str(row["file_descriptor"]),
                output_path,
                max_time=max_time,
                attempts=attempts,
                initial_backoff_seconds=initial_backoff_seconds,
            )
            status = "downloaded" if ok else "failed"
            if not ok:
                print(f"[WARN] {safe_name}: {message}")
            time.sleep(request_delay_seconds)
        actual_size = output_path.stat().st_size if output_path.exists() else 0
        rows.append(
            {
                "drug": row["drug"],
                "name": safe_name,
                "file_descriptor": row["file_descriptor"],
                "expected_size_bytes": expected_size,
                "actual_size_bytes": int(actual_size),
                "size_matches_index": bool(abs(actual_size - expected_size) <= 2),
                "local_path": str(output_path),
                "status": status,
                "message": message,
            }
        )
    manifest = pd.DataFrame(rows)
    failed = manifest[manifest["status"].eq("failed")]
    if len(failed) > 0:
        raise RuntimeError(f"{len(failed)} TXT downloads failed; see download manifest.")
    return manifest


def load_feature_order(root: Path) -> list[str]:
    path = root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv"
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [norm_text(col) for col in cols if col != "Sample_ID"]


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


def parse_txt_files(download_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, object]] = []
    for _, file_row in download_manifest.iterrows():
        path = Path(str(file_row["local_path"]))
        df = pd.read_csv(path, sep="\t", encoding="utf-8", low_memory=False)
        missing = [col for col in REQUIRED_TXT_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")

        drug = norm_compound(file_row["drug"])
        out = pd.DataFrame(
            {
                "Compound": df["drug"].map(norm_compound),
                "compound_norm": df["drug"].map(norm_accession),
                "source_drug_folder": drug,
                "source_file": path.name,
                "identifier": df["identifier"].map(norm_text),
                "protein_feature_id": df["gene"].map(norm_accession),
                "uniprot_id": df["gene"].map(norm_accession),
                "gene_symbol": df["Gene names"].map(first_gene_symbol),
                "gene_names": df["Gene names"].map(norm_text),
                "description": df["Description"].map(norm_text),
                "pred_class": df[".pred_class"].map(norm_text),
                "down": df.get("down", pd.Series(index=df.index, dtype=object)).map(norm_text),
                "r2": pd.to_numeric(df["r2"], errors="coerce"),
                "rse": pd.to_numeric(df.get("rse", np.nan), errors="coerce"),
                "bic": pd.to_numeric(df.get("bic", np.nan), errors="coerce"),
                "aic": pd.to_numeric(df.get("aic", np.nan), errors="coerce"),
                "AUC": pd.to_numeric(df["AUC"], errors="coerce"),
                "slope": pd.to_numeric(df.get("slope", np.nan), errors="coerce"),
                "bottom": pd.to_numeric(df.get("bottom", np.nan), errors="coerce"),
                "top": pd.to_numeric(df.get("top", np.nan), errors="coerce"),
                "inflection": pd.to_numeric(df.get("inflection", np.nan), errors="coerce"),
                "IC50_nM": pd.to_numeric(df["IC50"], errors="coerce"),
                "stdIC50_nM": pd.to_numeric(df["stdIC50"], errors="coerce"),
                "DMSO_percentile": pd.to_numeric(df.get("DMSO_percentile", np.nan), errors="coerce"),
                "MAD": pd.to_numeric(df.get("MAD", np.nan), errors="coerce"),
                "beginningOfCurve": pd.to_numeric(df.get("beginningOfCurve", np.nan), errors="coerce"),
                "endOfCurve": pd.to_numeric(df.get("endOfCurve", np.nan), errors="coerce"),
                "linearSlope": pd.to_numeric(df.get("linearSlope", np.nan), errors="coerce"),
                "R2_Bins": df.get("R2_Bins", pd.Series(index=df.index, dtype=object)).map(norm_text),
            }
        )
        for dose in ["0", "1", "10", "100", "1000", "10000"]:
            col = f"Normalized_Intensity_{dose}"
            out[f"normalized_intensity_{dose}nM"] = pd.to_numeric(df[col], errors="coerce")
            pep_col = f"Unique_Peptides_{dose}"
            if pep_col in df.columns:
                out[f"unique_peptides_{dose}nM"] = pd.to_numeric(df[pep_col], errors="coerce")
        intensity = out["normalized_intensity_10000nM"]
        out["log2FC_10000nM"] = np.where(intensity > 0, np.log2(intensity), np.nan)
        out["pEC50"] = np.where(out["IC50_nM"] > 0, 9.0 - np.log10(out["IC50_nM"]), np.nan)
        frames.append(out)
        file_rows.append(
            {
                "source_file": path.name,
                "drug": drug,
                "rows": int(len(df)),
                "required_columns_present": True,
                "finite_log2FC_10000nM": int(np.isfinite(out["log2FC_10000nM"]).sum()),
                "normalized_intensity_0_equals_1_fraction": float(np.isclose(out["normalized_intensity_0nM"], 1.0).mean()),
                "pred_class_counts": json.dumps(out["pred_class"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    long_df = pd.concat(frames, ignore_index=True)
    file_summary = pd.DataFrame(file_rows)
    long_df = long_df[long_df["protein_feature_id"] != ""].copy()
    return long_df, file_summary


def build_wide_and_aligned(
    long_df: pd.DataFrame,
    root: Path,
    package_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = package_dir / "data"
    processed_dir = data_dir / "processed"
    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    compounds = sorted(long_df["Compound"].dropna().unique().tolist(), key=lambda x: x.upper())
    collapsed = (
        long_df.groupby(["Compound", "compound_norm", "protein_feature_id"], sort=False)["log2FC_10000nM"]
        .median()
        .reset_index()
    )
    wide = collapsed.pivot(index=["Compound", "compound_norm"], columns="protein_feature_id", values="log2FC_10000nM")
    wide = wide.reset_index().sort_values("Compound").reset_index(drop=True)
    wide.to_csv(data_dir / "jurkat_decrypte_wide_log2fc_10000nM.csv", index=False, encoding="utf-8")

    gene_symbol_collapsed = (
        long_df[long_df["gene_symbol"] != ""]
        .groupby(["Compound", "compound_norm", "gene_symbol"], sort=False)["log2FC_10000nM"]
        .median()
        .reset_index()
    )
    gene_symbol_wide = gene_symbol_collapsed.pivot(
        index=["Compound", "compound_norm"], columns="gene_symbol", values="log2FC_10000nM"
    )
    gene_symbol_wide = gene_symbol_wide.reset_index().sort_values("Compound").reset_index(drop=True)
    gene_symbol_wide.to_csv(data_dir / "jurkat_decrypte_wide_gene_symbol_log2fc_10000nM.csv", index=False, encoding="utf-8")

    feature_order = load_feature_order(root)
    matrix = collapsed.pivot(index="protein_feature_id", columns="Compound", values="log2FC_10000nM")
    shared = sorted(set(matrix.index) & set(feature_order))
    aligned = matrix.reindex(feature_order).apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    aligned.index.name = "Compound"
    aligned = aligned.loc[compounds]
    aligned = aligned.mask(aligned.abs() <= THRESHOLD_LOG2FC, 0.0)
    aligned["compound_norm"] = [norm_accession(compound) for compound in aligned.index]
    aligned_df = aligned.reset_index()
    aligned_df.to_csv(table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv", index=False, encoding="utf-8")

    coverage = pd.DataFrame(
        [
            {
                "panel_name": PANEL_NAME,
                "panel_slug": PANEL_SLUG,
                "massive_accession": MASSIVE_ACCESSION,
                "pxd_accession": PXD_ACCESSION,
                "n_compounds": int(len(compounds)),
                "n_input_proteins": int(long_df["protein_feature_id"].nunique()),
                "n_gene_symbols": int(long_df["gene_symbol"].replace("", np.nan).nunique()),
                "n_hct116_features": int(len(feature_order)),
                "n_shared_features": int(len(shared)),
                "panel_feature_coverage_pct": float(len(shared) / max(1, long_df["protein_feature_id"].nunique()) * 100.0),
                "hct116_feature_overlap_pct": float(len(shared) / max(1, len(feature_order)) * 100.0),
                "threshold_log2fc": float(THRESHOLD_LOG2FC),
                "source_kind": "decryptE Jurkat normalized intensity at 10000 nM converted by log2",
                "feature_identifier_for_prediction": "UniProt accession from TXT column gene",
                "gene_symbol_annotation_column": "Gene names",
            }
        ]
    )
    coverage.to_csv(table_dir / f"{PANEL_SLUG}_coverage_summary.csv", index=False, encoding="utf-8")
    return wide, gene_symbol_wide, aligned_df, coverage


def write_projection_inputs(
    aligned_df: pd.DataFrame,
    root: Path,
    package_dir: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_dir = add_pipeline_path(root)
    external_mod = load_module(
        "deepheal_external_panel_projection_jurkat",
        pipeline_dir / 'panel_projection.py',
    )
    helper_mod = load_module("deepheal_projection_helper_jurkat", pipeline_dir / 'projection_utils.py')
    supcon_base = load_module("deepheal_supcon_base_jurkat", external_mod.SUPCON_BASE_SCRIPT)
    average_mod = load_module("deepheal_average_mod_jurkat", external_mod.AVERAGE_ENSEMBLE_SCRIPT)

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


def write_column_dictionary(data_dir: Path) -> None:
    rows = [
        {
            "file_or_table": "*_ALL.txt",
            "column": ".pred_class",
            "role": "dose-response classification/regulation status from MassIVE TXT",
            "use_for_prediction": False,
        },
        {"file_or_table": "*_ALL.txt", "column": "gene", "role": "UniProt-like protein accession; DeepHeal feature alignment key", "use_for_prediction": True},
        {"file_or_table": "*_ALL.txt", "column": "Gene names", "role": "gene-symbol annotation retained in long CSV", "use_for_prediction": False},
        {"file_or_table": "*_ALL.txt", "column": "Normalized_Intensity_10000", "role": "10,000 nM abundance ratio versus DMSO", "use_for_prediction": True},
        {"file_or_table": "jurkat_decrypte_long_log2fc_10000nM.csv", "column": "log2FC_10000nM", "role": "log2(Normalized_Intensity_10000), NaN when intensity <= 0 or missing", "use_for_prediction": True},
        {"file_or_table": "*_ALL.txt", "column": "IC50", "role": "curve IC50 in nM; also converted to pEC50 = 9 - log10(IC50_nM)", "use_for_prediction": False},
        {"file_or_table": "*_ALL.txt", "column": "AUC", "role": "curve area summary", "use_for_prediction": False},
        {"file_or_table": "*_ALL.txt", "column": "r2", "role": "curve fit R2", "use_for_prediction": False},
        {"file_or_table": "aligned_threshold015", "column": "protein feature columns", "role": "UniProt-aligned, missing filled with 0, abs(log2FC) <= 0.15 thresholded to 0", "use_for_prediction": True},
    ]
    pd.DataFrame(rows).to_csv(data_dir / "jurkat_decrypte_column_dictionary.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download decryptE Jurkat proteome TXT files and prepare DeepHeal features.")
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--package-dir", type=Path, default=package_dir_from_script())
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-max-time", type=int, default=240)
    parser.add_argument("--download-attempts", type=int, default=6)
    parser.add_argument("--initial-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--request-delay-seconds", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = args.package_dir.resolve()
    data_dir = package_dir / "data"
    raw_dir = data_dir / "massive_jurkat_proteome_all_txt"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    jurkat_txt, primary_all = build_massive_txt_indexes(data_dir)
    if args.skip_download:
        rows = []
        for _, row in primary_all.iterrows():
            local_path = raw_dir / str(row["name"])
            rows.append(
                {
                    "drug": row["drug"],
                    "name": row["name"],
                    "file_descriptor": row["file_descriptor"],
                    "expected_size_bytes": int(row["size"]),
                    "actual_size_bytes": int(local_path.stat().st_size if local_path.exists() else 0),
                    "size_matches_index": bool(local_path.exists() and abs(local_path.stat().st_size - int(row["size"])) <= 2),
                    "local_path": str(local_path),
                    "status": "existing" if is_valid_txt(local_path) else "missing_or_invalid",
                    "message": "skip-download mode",
                }
            )
        download_manifest = pd.DataFrame(rows)
        if not download_manifest["status"].isin(["existing"]).all():
            bad = download_manifest[~download_manifest["status"].isin(["existing"])]
            raise RuntimeError(f"skip-download requested but {len(bad)} local TXT files are missing or invalid")
    else:
        download_manifest = download_primary_txts(
            primary_all=primary_all,
            raw_dir=raw_dir,
            force=args.force_download,
            max_time=args.download_max_time,
            attempts=args.download_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            request_delay_seconds=args.request_delay_seconds,
        )
    download_manifest.to_csv(data_dir / "massive_jurkat_txt_download_manifest.csv", index=False, encoding="utf-8")

    long_df, file_summary = parse_txt_files(download_manifest)
    long_path = data_dir / "jurkat_decrypte_long_log2fc_10000nM.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    file_summary.to_csv(data_dir / "jurkat_decrypte_txt_file_parse_summary.csv", index=False, encoding="utf-8")

    wide_df, gene_symbol_wide, aligned_df, coverage = build_wide_and_aligned(long_df, root, package_dir)
    label_match = build_label_match(root, wide_df["Compound"].tolist())
    label_match.to_csv(processed_dir / "jurkat_decrypte_deepheal_label_match.csv", index=False, encoding="utf-8")
    write_column_dictionary(data_dir)

    pca_df, supcon_embed_df, average_embed_df = write_projection_inputs(
        aligned_df=aligned_df,
        root=root,
        package_dir=package_dir,
        batch_size=args.batch_size,
    )

    verification = {
        "n_primary_all_txt_expected": 144,
        "n_primary_all_txt_indexed": int(len(primary_all)),
        "n_downloaded_or_existing_valid_txt": int(download_manifest["status"].isin(["downloaded", "skipped_existing", "existing"]).sum()),
        "download_total_size_MB": float(download_manifest["actual_size_bytes"].sum() / 1024 / 1024),
        "n_jurkat_txt_index_rows": int(len(jurkat_txt)),
        "n_long_rows": int(len(long_df)),
        "n_compounds": int(wide_df.shape[0]),
        "n_uniprot_features_in_wide": int(wide_df.shape[1] - 2),
        "n_gene_symbol_features_in_extra_wide": int(gene_symbol_wide.shape[1] - 2),
        "finite_log2FC_10000nM_rows": int(np.isfinite(long_df["log2FC_10000nM"]).sum()),
        "normalized_intensity_0_equals_1_median_fraction_by_file": float(file_summary["normalized_intensity_0_equals_1_fraction"].median()),
        "feature_counts": {
            "full_protein": int(len([col for col in aligned_df.columns if col not in {"Compound", "compound_norm"}])),
            "pca_200": int(len([col for col in pca_df.columns if col.startswith("pca_")])),
            "supcon_latent": int(len([col for col in supcon_embed_df.columns if col.startswith("latent_")])),
            "average_latent": int(len([col for col in average_embed_df.columns if col.startswith("latent_")])),
        },
        "coverage": coverage.iloc[0].to_dict(),
        "required_columns": REQUIRED_TXT_COLUMNS,
        "outputs": {
            "long_csv": str(long_path),
            "wide_csv": str(data_dir / "jurkat_decrypte_wide_log2fc_10000nM.csv"),
            "gene_symbol_wide_csv": str(data_dir / "jurkat_decrypte_wide_gene_symbol_log2fc_10000nM.csv"),
            "column_dictionary_csv": str(data_dir / "jurkat_decrypte_column_dictionary.csv"),
            "aligned_threshold_csv": str(package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables" / f"{PANEL_SLUG}_aligned_threshold015.csv"),
        },
    }
    expected_counts = {"full_protein": 9960, "pca_200": 200, "supcon_latent": 257, "average_latent": 194}
    if verification["feature_counts"] != expected_counts:
        raise AssertionError(f"Unexpected feature counts: {verification['feature_counts']}")
    if verification["n_compounds"] != 144:
        raise AssertionError(f"Expected 144 compounds, observed {verification['n_compounds']}")
    summary_path = processed_dir / "jurkat_decrypte_feature_preparation_summary.json"
    summary_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
