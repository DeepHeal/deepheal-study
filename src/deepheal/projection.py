from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT

from pathlib import Path
from collections import Counter
import pickle
import re
import numpy as np
import pandas as pd
from cptac_annotations import norm

def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)

def load_source_feature_data(repo_root: Path) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_file = repo_root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv"
    feature_order = [norm(c) for c in pd.read_csv(feature_file, nrows=0).columns if norm(c) != "SAMPLE_ID"]
    avg = pd.read_csv(repo_root / "data" / "raw" / "875_Compounds_Average_Data.csv", encoding="latin-1")
    avg.columns = [norm(c) for c in avg.columns]
    avg["GENE NAME"] = avg["GENE NAME"].map(norm)
    avg["UNIPROTID"] = avg["UNIPROTID"].map(norm)
    avg = avg.drop_duplicates("UNIPROTID", keep="first").set_index("UNIPROTID")
    avg = avg.reindex(feature_order)
    if avg.index.isna().any() or avg.iloc[:, 2:].isna().all(axis=1).any():
        raise RuntimeError("Could not align all 9,960 DeepHeal features to average drug data")
    meta = pd.read_csv(repo_root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv")
    meta["sample_norm"] = meta["Sample_ID"].map(norm)
    meta["drug_norm"] = meta["Drug_Name"].map(norm)
    sample_columns = [c for c in avg.columns[1:] if c in set(meta["sample_norm"])]
    drug_matrix = avg[sample_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    drug_matrix = drug_matrix.mask(drug_matrix.abs() <= 0.15, 0.0)
    drug_meta = meta.set_index("sample_norm").loc[sample_columns].copy()
    drug_meta.index = pd.Index(drug_meta["drug_norm"], name="drug_norm")
    drug_matrix.index = drug_meta.index
    local_mapping = pd.DataFrame(
        {
            "feature": feature_order,
            "gene_symbol": avg["GENE NAME"].to_numpy(),
        }
    )
    return feature_order, drug_matrix, drug_meta.reset_index(drop=True), local_mapping

def load_supcon_drug_matrix(repo_root: Path, feature_order: list[str]) -> pd.DataFrame:
    """Recreate the exact replicate-aware SupCon deployment target for fidelity."""
    data = pd.read_csv(repo_root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv")
    features = [c for c in data.columns if c != "Sample_ID"]
    if [norm(c) for c in features] != feature_order:
        raise RuntimeError("SupCon replicate feature order differs from the archived 9,960-feature order")
    x_all = data[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    x_all[np.abs(x_all) <= 0.15] = 0.0

    def original_name(sample_id: object) -> str:
        value = str(sample_id).strip().replace("\xa0", " ")
        return value.rsplit("_", 1)[0] if value.endswith(("_1", "_2")) else value

    grouped: dict[str, list[np.ndarray]] = {}
    for sample_id, vector in zip(data["Sample_ID"], x_all, strict=True):
        grouped.setdefault(original_name(sample_id), []).append(vector)
    baseline_header = pd.read_csv(repo_root / "data" / "raw" / "1628_Experiments_Data.csv", nrows=0).columns.tolist()[2:]
    real_rep_count = Counter(original_name(sample_id) for sample_id in baseline_header)
    meta = pd.read_csv(repo_root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv")
    rows: list[np.ndarray] = []
    names: list[str] = []
    for drug in sorted(meta["Drug_Name"].astype(str).str.strip().unique()):
        replicates = grouped.get(drug)
        if not replicates:
            raise RuntimeError(f"Missing SupCon replicate-aware input for {drug}")
        target = (replicates[0] + replicates[1]) / 2.0 if real_rep_count.get(drug, 0) >= 2 and len(replicates) >= 2 else replicates[0]
        rows.append(target.astype(np.float32))
        names.append(norm(drug))
    matrix = pd.DataFrame(np.vstack(rows), index=pd.Index(names, name="drug_norm"), columns=feature_order)
    if len(matrix) != 875 or matrix.index.duplicated().any():
        raise RuntimeError("SupCon reference must contain 875 unique drugs")
    return matrix

def annotation_gene_to_swissprot(annotation: pd.DataFrame) -> dict[str, str]:
    usable = annotation.loc[annotation["gene_symbol"].ne("") & annotation["swissprot"].ne("")].copy()
    usable = usable.sort_values(["gene_symbol", "swissprot"]).drop_duplicates("gene_symbol")
    return usable.set_index("gene_symbol")["swissprot"].to_dict()

def build_feature_assignment(
    genes: list[str],
    feature_order: list[str],
    local_mapping: pd.DataFrame,
    gene_to_swissprot: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_order)}
    base_to_features: dict[str, list[int]] = {}
    gene_to_features: dict[str, list[int]] = {}
    for idx, row in local_mapping.iterrows():
        base = str(row["feature"]).split("-", 1)[0]
        base_to_features.setdefault(base, []).append(idx)
        gene_to_features.setdefault(str(row["gene_symbol"]), []).append(idx)

    assigned_features: set[int] = set()
    assigned_genes: set[int] = set()
    feature_indices: list[int] = []
    gene_indices: list[int] = []
    methods: list[str] = []
    for gene_idx, gene in enumerate(genes):
        swissprot = gene_to_swissprot.get(gene, "")
        feature_idx: int | None = None
        method = ""
        if swissprot in feature_to_idx:
            feature_idx = feature_to_idx[swissprot]
            method = "swissprot_exact"
        elif swissprot and len(base_to_features.get(swissprot.split("-", 1)[0], [])) == 1:
            feature_idx = base_to_features[swissprot.split("-", 1)[0]][0]
            method = "swissprot_unique_base"
        elif len(gene_to_features.get(gene, [])) == 1:
            feature_idx = gene_to_features[gene][0]
            method = "unique_gene_symbol"
        if feature_idx is None or feature_idx in assigned_features or gene_idx in assigned_genes:
            continue
        assigned_features.add(feature_idx)
        assigned_genes.add(gene_idx)
        feature_indices.append(feature_idx)
        gene_indices.append(gene_idx)
        methods.append(method)
    return np.asarray(feature_indices, dtype=int), np.asarray(gene_indices, dtype=int), methods

def ood_diagnostics(
    raw_all: np.ndarray,
    pca_all: np.ndarray,
    pca_model,
    n_drugs: int,
    patient_meta: pd.DataFrame,
    branch: str,
) -> pd.DataFrame:
    reconstructed = pca_model.inverse_transform(pca_all)
    rmse = np.sqrt(np.mean((raw_all - reconstructed) ** 2, axis=1))
    l2 = np.linalg.norm(raw_all, axis=1)
    center = pca_all[:n_drugs].mean(axis=0)
    scale = pca_all[:n_drugs].std(axis=0)
    scale[scale < 1e-8] = 1.0
    std_distance = np.sqrt(np.mean(((pca_all - center) / scale) ** 2, axis=1))
    thresholds = {
        "raw_l2": float(np.quantile(l2[:n_drugs], 0.99)),
        "pca_reconstruction_rmse": float(np.quantile(rmse[:n_drugs], 0.99)),
        "standardized_pca_distance": float(np.quantile(std_distance[:n_drugs], 0.99)),
    }
    out = patient_meta.copy()
    out["branch"] = branch
    out["raw_l2"] = l2[n_drugs:]
    out["pca_reconstruction_rmse"] = rmse[n_drugs:]
    out["standardized_pca_distance"] = std_distance[n_drugs:]
    out["ood_raw_l2"] = out["raw_l2"] > thresholds["raw_l2"]
    out["ood_reconstruction"] = out["pca_reconstruction_rmse"] > thresholds["pca_reconstruction_rmse"]
    out["ood_pca_distance"] = out["standardized_pca_distance"] > thresholds["standardized_pca_distance"]
    out["ood_any"] = out[["ood_raw_l2", "ood_reconstruction", "ood_pca_distance"]].any(axis=1)
    for key, value in thresholds.items():
        out[f"drug_reference_99pct_{key}"] = value
    return out
