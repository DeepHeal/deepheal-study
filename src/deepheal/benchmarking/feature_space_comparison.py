#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.metadata
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
POSITIVE_CLASS = "Antineoplastic"
NEGATIVE_CLASS = "Non-Antineoplastic"
KNOWN_CLASSES = [POSITIVE_CLASS, NEGATIVE_CLASS]
THRESHOLD_LOG2FC = 0.15
OUTPUT_DIR_NAME = 'autogluon_feature_space_comparison'
# Keep comparison artifacts isolated from existing release predictors and tables.

FEATURE_SPACE_ORDER = [
    "pca_200",
    "supcon_latent",
    "average_latent",
    "full_protein",
]

FEATURE_SPACE_LABELS = {
    "pca_200": "First 200 PCA dimensions",
    "supcon_latent": "DeepHeal SupCon latent",
    "average_latent": "DeepHeal Average latent",
    "full_protein": "Full-dimensional protein features",
}

FEATURE_SPACE_EXPECTED_COUNTS = {
    "pca_200": 200,
    "supcon_latent": 257,
    "average_latent": 194,
    "full_protein": 9960,
}

MODEL_COLORS = {
    "Traditional": "#4C78A8",
    "GBDT": "#59A14F",
    "Deep Learning": "#E15759",
    "Foundation": "#F28E2B",
    "Other": "#79706E",
}

FEATURE_COLORS = {
    "pca_200": "#4C78A8",
    "supcon_latent": "#E15759",
    "average_latent": "#59A14F",
    "full_protein": "#F28E2B",
}


def norm(value: object) -> str:
    return str(value).strip().replace("\xa0", " ")


def norm_sample_id(value: object) -> str:
    return norm(value).replace(" _", "_").replace("_ ", "_")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="whitegrid")


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def categorize_model(model_name: str) -> str:
    name = model_name.lower()
    if any(x in name for x in ["kneighbors", "randomforest", "extra_trees"]):
        return "Traditional"
    if any(x in name for x in ["lightgbm", "catboost", "xgboost"]):
        return "GBDT"
    if any(x in name for x in ["neuralnet", "fttransformer", "nn_torch", "realmlp"]):
        return "Deep Learning"
    if any(x in name for x in ["tabpfn", "tabm", "tabicl", "tabdpt"]):
        return "Foundation"
    return "Other"


def build_stratify_key(
    y: np.ndarray,
    clusters: np.ndarray,
    min_count: int = 2,
    context: str = "",
) -> np.ndarray:
    combined = np.array([f"{label}_C{cluster}" for label, cluster in zip(y, clusters)])
    counts = Counter(combined)
    if min(counts.values()) >= min_count:
        return combined

    merged = combined.copy()
    for cls in sorted(set(y.tolist())):
        cls_prefix = f"{cls}_C"
        cls_keys = {k: v for k, v in counts.items() if k.startswith(cls_prefix)}
        cls_tiny = {k: v for k, v in cls_keys.items() if v < min_count}
        cls_large = {k: v for k, v in cls_keys.items() if v >= min_count}
        if not cls_tiny:
            continue
        if cls_large:
            largest_key = max(cls_large, key=cls_large.get)
            for key in cls_tiny:
                merged[merged == key] = largest_key
        else:
            for key in cls_tiny:
                merged[merged == key] = str(cls)

    counts_merged = Counter(merged)
    if min(counts_merged.values()) < min_count:
        print(f"[{context}] sparse strata remain after merge; fallback to label-only stratification")
        return y.astype(str)
    return merged


def model_hyperparameters(full: bool = True) -> dict[str, dict]:
    if full:
        return {
            "KNN": {},
            "RF": {},
            "GBM": {},
            "CAT": {},
            "XGB": {},
            "REALMLP": {},
            "TABM": {},
            "TABICL": {},
            "REALTABPFN-V2": {},
            "REALTABPFN-V2.5": {},
            "NN_TORCH": {},
            "FT_TRANSFORMER": {},
            "TABDPT": {"ag_args_fit": {"num_gpus": 0}},
        }
    return {
        "KNN": {},
        "RF": {},
        "GBM": {},
        "CAT": {},
        "XGB": {},
        "NN_TORCH": {},
    }


def load_labels(root: Path) -> pd.DataFrame:
    path = root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv"
    meta = pd.read_csv(path)
    meta["Drug_Name"] = meta["Drug_Name"].map(norm)
    meta["Sample_ID"] = meta["Sample_ID"].map(norm_sample_id)
    meta["Drug_Class"] = meta["Drug_Class"].map(norm)
    meta["Drug_ID"] = meta["Drug_Name"].map(norm)
    if meta["Drug_ID"].duplicated().any():
        duplicates = meta.loc[meta["Drug_ID"].duplicated(), "Drug_ID"].tolist()
        raise ValueError(f"Duplicate Drug_ID values in labels: {duplicates[:5]}")
    return meta


def load_average_protein_matrix(root: Path, meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    raw_dir = root / "data" / "raw"
    ref_cols = pd.read_csv(raw_dir / "1752_Experiments_Data_With_Replicate_for_VAE.csv", nrows=0).columns.tolist()
    ref_feature_order = [norm(c) for c in ref_cols if c != "Sample_ID"]

    avg = pd.read_csv(raw_dir / "875_Compounds_Average_Data.csv", encoding="latin-1")
    avg.columns = [norm(c) if i < 2 else norm_sample_id(c) for i, c in enumerate(avg.columns)]
    avg["Gene Name"] = avg["Gene Name"].map(norm)
    avg["UniprotID"] = avg["UniprotID"].map(norm)
    if avg["UniprotID"].duplicated().any():
        n_dup = int(avg["UniprotID"].duplicated().sum())
        raise ValueError(f"Average protein matrix has {n_dup} duplicated UniprotID rows")

    missing_ref = sorted(set(ref_feature_order) - set(avg["UniprotID"]))
    if missing_ref:
        raise ValueError(f"Average protein matrix is missing {len(missing_ref)} reference proteins")

    sample_ids = [norm_sample_id(c) for c in avg.columns.tolist()[2:]]
    meta_by_sample = meta.set_index("Sample_ID")
    missing_samples = sorted(set(sample_ids) - set(meta_by_sample.index))
    if missing_samples:
        raise ValueError(f"Average protein matrix has {len(missing_samples)} samples missing from labels")

    avg_matrix = avg.set_index("UniprotID")[sample_ids]
    avg_matrix = avg_matrix.loc[ref_feature_order]
    avg_matrix = avg_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    avg_matrix = avg_matrix.where(avg_matrix.abs() > THRESHOLD_LOG2FC, 0.0)
    x_avg = avg_matrix.to_numpy(dtype=np.float32).T

    rows = []
    for sample_id, values in zip(sample_ids, x_avg):
        label_row = meta_by_sample.loc[sample_id]
        rows.append(
            {
                "Drug_ID": norm(label_row["Drug_Name"]),
                "Sample_ID": sample_id,
                "Drug_Name": norm(label_row["Drug_Name"]),
                "Drug_Class": norm(label_row["Drug_Class"]),
                "Cluster": int(label_row["Cluster"]),
                "values": values,
            }
        )

    protein_df = pd.DataFrame(rows)
    if protein_df["Drug_ID"].duplicated().any():
        duplicates = protein_df.loc[protein_df["Drug_ID"].duplicated(), "Drug_ID"].tolist()
        raise ValueError(f"Duplicate Drug_ID values in protein matrix: {duplicates[:5]}")

    provenance = {
        "protein_matrix_path": str(raw_dir / "875_Compounds_Average_Data.csv"),
        "protein_matrix_encoding": "latin-1",
        "reference_feature_order_path": str(raw_dir / "1752_Experiments_Data_With_Replicate_for_VAE.csv"),
        "threshold_log2fc": THRESHOLD_LOG2FC,
        "n_samples": int(len(protein_df)),
        "n_features": int(len(ref_feature_order)),
    }
    return protein_df, ref_feature_order, provenance


def values_to_feature_frame(
    source_df: pd.DataFrame,
    feature_names: list[str],
    key_col: str = "Drug_ID",
) -> pd.DataFrame:
    values = np.vstack(source_df["values"].to_list()).astype(np.float32)
    features = pd.DataFrame(values, columns=feature_names)
    features.insert(0, key_col, source_df[key_col].to_numpy())
    return features


def make_full_protein_features(protein_df: pd.DataFrame, ref_feature_order: list[str]) -> tuple[pd.DataFrame, list[str]]:
    feature_names = [f"protein_{i:04d}_{protein}" for i, protein in enumerate(ref_feature_order)]
    feature_df = values_to_feature_frame(protein_df, feature_names)
    return feature_df, feature_names


def make_pca_features(root: Path, protein_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    pca_path = root / "results" / "supcon" / "supcon_pca.pkl"
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    values = np.vstack(protein_df["values"].to_list()).astype(np.float32)
    transformed = pca.transform(values).astype(np.float32)
    if transformed.shape[1] < FEATURE_SPACE_EXPECTED_COUNTS["pca_200"]:
        raise ValueError(f"PCA transform produced only {transformed.shape[1]} dimensions")
    pca_200 = transformed[:, : FEATURE_SPACE_EXPECTED_COUNTS["pca_200"]]
    feature_names = [f"pca_{i:03d}" for i in range(pca_200.shape[1])]
    features = pd.DataFrame(pca_200, columns=feature_names)
    features.insert(0, "Drug_ID", protein_df["Drug_ID"].to_numpy())
    provenance = {
        "pca_path": str(pca_path),
        "pca_source": "results/supcon/supcon_pca.pkl",
        "pca_dimensions_available": int(transformed.shape[1]),
        "pca_dimensions_used": int(pca_200.shape[1]),
    }
    return features, feature_names, provenance


def make_latent_features(path: Path, expected_prefix: str = "latent_") -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    df = pd.read_csv(path)
    df["Drug_Name"] = df["Drug_Name"].map(norm)
    df["Drug_ID"] = df["Drug_Name"].map(norm)
    if df["Drug_ID"].duplicated().any():
        duplicates = df.loc[df["Drug_ID"].duplicated(), "Drug_ID"].tolist()
        raise ValueError(f"Duplicate Drug_ID values in {path}: {duplicates[:5]}")
    latent_cols = [c for c in df.columns if c.startswith(expected_prefix)]
    if not latent_cols:
        raise ValueError(f"No {expected_prefix} columns found in {path}")
    feature_df = df[["Drug_ID", *latent_cols]].copy()
    provenance = {
        "embedding_path": str(path),
        "n_rows": int(len(df)),
        "n_features": int(len(latent_cols)),
    }
    return feature_df, latent_cols, provenance


def build_feature_spaces(root: Path, meta: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, object]]:
    protein_df, ref_feature_order, protein_provenance = load_average_protein_matrix(root, meta)
    pca_features, pca_cols, pca_provenance = make_pca_features(root, protein_df)
    full_features, full_cols = make_full_protein_features(protein_df, ref_feature_order)
    supcon_features, supcon_cols, supcon_provenance = make_latent_features(
        root / "results" / "supcon" / "supcon_consensus_embedding_including_marginal.csv"
    )
    average_features, average_cols, average_provenance = make_latent_features(
        root / "results" / "average" / "average_consensus_embedding_including_marginal.csv"
    )

    features = {
        "pca_200": pca_features,
        "supcon_latent": supcon_features,
        "average_latent": average_features,
        "full_protein": full_features,
    }
    feature_cols = {
        "pca_200": pca_cols,
        "supcon_latent": supcon_cols,
        "average_latent": average_cols,
        "full_protein": full_cols,
    }
    provenance = {
        "protein": protein_provenance,
        "pca": pca_provenance,
        "supcon_latent": supcon_provenance,
        "average_latent": average_provenance,
    }
    return features, feature_cols, provenance


def make_shared_labeled_cohort(meta: pd.DataFrame, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labeled = meta[meta["Drug_Class"].isin(KNOWN_CLASSES)].copy()
    shared_ids = set(labeled["Drug_ID"])
    for feature_df in features.values():
        shared_ids &= set(feature_df["Drug_ID"])
    missing = {
        name: sorted(set(labeled["Drug_ID"]) - set(feature_df["Drug_ID"]))
        for name, feature_df in features.items()
        if set(labeled["Drug_ID"]) - set(feature_df["Drug_ID"])
    }
    if missing:
        raise ValueError(f"Feature spaces are missing labeled drugs: {missing}")

    cohort = labeled[labeled["Drug_ID"].isin(shared_ids)].copy()
    cohort = cohort.sort_values("Drug_ID").reset_index(drop=True)
    if len(cohort) != 448:
        raise ValueError(f"Expected 448 labeled drugs after feature intersection, observed {len(cohort)}")
    return cohort


def split_cohort(cohort: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    y_binary = (cohort["Drug_Class"] == POSITIVE_CLASS).astype(int).to_numpy()
    clusters = cohort["Cluster"].to_numpy(dtype=int)
    stratify_key = build_stratify_key(y_binary, clusters, min_count=2, context="train_test_split")
    idx = np.arange(len(cohort))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_key,
    )
    train_meta = cohort.iloc[train_idx].sort_values("Drug_ID").reset_index(drop=True)
    test_meta = cohort.iloc[test_idx].sort_values("Drug_ID").reset_index(drop=True)
    split_info = {
        "seed": int(seed),
        "test_size": 0.2,
        "n_train": int(len(train_meta)),
        "n_test": int(len(test_meta)),
        "train_drug_ids": train_meta["Drug_ID"].tolist(),
        "test_drug_ids": test_meta["Drug_ID"].tolist(),
        "train_class_counts": train_meta["Drug_Class"].value_counts().to_dict(),
        "test_class_counts": test_meta["Drug_Class"].value_counts().to_dict(),
        "stratification": "Drug_Class binary label plus Cluster with sparse-strata merge",
    }
    return train_meta, test_meta, split_info


def assemble_autogluon_frame(meta_part: pd.DataFrame, feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    indexed = feature_df.set_index("Drug_ID", verify_integrity=True)
    missing = sorted(set(meta_part["Drug_ID"]) - set(indexed.index))
    if missing:
        raise ValueError(f"Missing feature rows for {len(missing)} drugs")
    x = indexed.loc[meta_part["Drug_ID"], feature_cols].reset_index(drop=True)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x["target"] = meta_part["Drug_Class"].to_numpy()
    return x


def fit_predictor(
    train_df: pd.DataFrame,
    model_dir: Path,
    time_limit: int,
    verbosity: int,
) -> tuple[TabularPredictor, str]:
    if model_dir.exists():
        shutil.rmtree(model_dir)

    try:
        predictor = TabularPredictor(
            label="target",
            path=str(model_dir),
            problem_type="binary",
            eval_metric="roc_auc",
            positive_class=POSITIVE_CLASS,
            verbosity=verbosity,
        ).fit(
            train_data=train_df,
            hyperparameters=model_hyperparameters(full=True),
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=time_limit,
            presets="best_quality",
        )
        return predictor, "full"
    except Exception as exc:
        print(f"[WARN] Full AutoGluon model zoo failed for {model_dir.name}: {exc}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        predictor = TabularPredictor(
            label="target",
            path=str(model_dir),
            problem_type="binary",
            eval_metric="roc_auc",
            positive_class=POSITIVE_CLASS,
            verbosity=verbosity,
        ).fit(
            train_data=train_df,
            hyperparameters=model_hyperparameters(full=False),
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=max(180, time_limit // 2),
            presets="best_quality",
        )
        return predictor, "fallback"


def positive_probabilities(predictor: TabularPredictor, data: pd.DataFrame, model: str) -> np.ndarray:
    proba = predictor.predict_proba(data, model=model)
    if POSITIVE_CLASS not in proba.columns:
        raise ValueError(f"Positive class column '{POSITIVE_CLASS}' missing from predict_proba output")
    return proba[POSITIVE_CLASS].to_numpy(dtype=float)


def compute_binary_metrics(y_true_label: pd.Series, y_pred_label: pd.Series, prob_pos: np.ndarray) -> dict[str, float]:
    y_true = (y_true_label == POSITIVE_CLASS).astype(int).to_numpy()
    y_pred = (y_pred_label == POSITIVE_CLASS).astype(int).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, prob_pos)),
        "auprc": float(average_precision_score(y_true, prob_pos)),
    }


def plot_metric_comparison(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    metric_order = ["auroc", "auprc", "mcc", "f1_pos", "accuracy"]
    plot_df = metrics_df.melt(
        id_vars=["feature_space", "feature_space_label"],
        value_vars=metric_order,
        var_name="metric",
        value_name="score",
    )
    plot_df["feature_space_label"] = pd.Categorical(
        plot_df["feature_space_label"],
        categories=[FEATURE_SPACE_LABELS[k] for k in FEATURE_SPACE_ORDER],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="feature_space_label",
        palette=[FEATURE_COLORS[k] for k in FEATURE_SPACE_ORDER],
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Held-out score")
    ax.set_ylim(0, 1.05)
    ax.set_title("AutoGluon antineoplastic classification across feature spaces")
    ax.legend(frameon=False, title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=6, padding=2)
    fig.tight_layout()
    save_figure(fig, out_dir / "figure_feature_space_metric_comparison")


def plot_roc_overlay(predictions_df: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for feature_space in FEATURE_SPACE_ORDER:
        sub = predictions_df[predictions_df["feature_space"] == feature_space].copy()
        y_true = (sub["Drug_Class"] == POSITIVE_CLASS).astype(int).to_numpy()
        prob = sub["prob_antineoplastic"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=FEATURE_COLORS[feature_space],
            label=f"{FEATURE_SPACE_LABELS[feature_space]} (AUROC {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC curves")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_dir / "figure_feature_space_roc_overlay")


def plot_confusion_matrices(predictions_df: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    class_order = [POSITIVE_CLASS, NEGATIVE_CLASS]
    for feature_space in FEATURE_SPACE_ORDER:
        sub = predictions_df[predictions_df["feature_space"] == feature_space]
        cm = confusion_matrix(sub["Drug_Class"], sub["y_pred"], labels=class_order)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_order,
            yticklabels=class_order,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(FEATURE_SPACE_LABELS[feature_space])
        fig.tight_layout()
        save_figure(fig, out_dir / f"figure_confusion_matrix_{feature_space}")


def train_and_evaluate_feature_space(
    feature_space: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    out_dir: Path,
    time_limit: int,
    verbosity: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Training {feature_space}: {len(feature_cols)} features")
    model_dir = out_dir / "predictors" / feature_space
    train_df = assemble_autogluon_frame(train_meta, feature_df, feature_cols)
    test_df = assemble_autogluon_frame(test_meta, feature_df, feature_cols)

    predictor, fit_mode = fit_predictor(train_df, model_dir, time_limit=time_limit, verbosity=verbosity)
    lb_full = predictor.leaderboard(test_df, silent=True)
    lb_full.insert(0, "feature_space", feature_space)
    lb_full.insert(1, "feature_space_label", FEATURE_SPACE_LABELS[feature_space])

    lb_single = lb_full[~lb_full["model"].str.contains("Ensemble|WeightedEnsemble", case=False, na=False)].copy()
    lb_single["category"] = lb_single["model"].map(categorize_model)
    lb_single = lb_single.sort_values(["score_test", "score_val"], ascending=False).reset_index(drop=True)
    if len(lb_single) == 0:
        raise ValueError(f"No non-ensemble AutoGluon model found for {feature_space}")
    best_model = str(lb_single.iloc[0]["model"])

    x_test = test_df[feature_cols]
    y_pred_label = predictor.predict(x_test, model=best_model).reset_index(drop=True)
    prob_pos = positive_probabilities(predictor, x_test, model=best_model)
    y_true_label = test_meta["Drug_Class"].reset_index(drop=True)
    metrics = compute_binary_metrics(y_true_label, y_pred_label, prob_pos)

    report = pd.DataFrame(
        classification_report(y_true_label, y_pred_label, output_dict=True, zero_division=0)
    ).T.reset_index(names="class_label")
    report.insert(0, "feature_space", feature_space)
    report.insert(1, "feature_space_label", FEATURE_SPACE_LABELS[feature_space])

    pred_df = test_meta[["Drug_ID", "Sample_ID", "Drug_Name", "Drug_Class", "Cluster"]].copy().reset_index(drop=True)
    pred_df.insert(0, "feature_space", feature_space)
    pred_df.insert(1, "feature_space_label", FEATURE_SPACE_LABELS[feature_space])
    pred_df["y_pred"] = y_pred_label.to_numpy()
    pred_df["prob_antineoplastic"] = prob_pos
    pred_df["prob_non_antineoplastic"] = 1.0 - prob_pos
    pred_df["best_model"] = best_model

    metrics_row: dict[str, object] = {
        "feature_space": feature_space,
        "feature_space_label": FEATURE_SPACE_LABELS[feature_space],
        "feature_count": int(len(feature_cols)),
        "expected_feature_count": int(FEATURE_SPACE_EXPECTED_COUNTS[feature_space]),
        "fit_mode": fit_mode,
        "best_model": best_model,
        "model_dir": str(model_dir),
        "n_train": int(len(train_meta)),
        "n_test": int(len(test_meta)),
        **metrics,
    }
    print(
        f"[RESULT] {feature_space} best_model={best_model} "
        f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"MCC={metrics['mcc']:.4f}"
    )
    return metrics_row, pred_df, lb_full, lb_single, report


def verify_outputs(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    split_info: dict[str, object],
) -> dict[str, object]:
    observed_spaces = metrics_df["feature_space"].tolist()
    if observed_spaces != FEATURE_SPACE_ORDER:
        raise AssertionError(f"Unexpected feature spaces: {observed_spaces}")

    feature_counts = dict(zip(metrics_df["feature_space"], metrics_df["feature_count"]))
    if feature_counts != FEATURE_SPACE_EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected feature counts: {feature_counts}")

    expected_test_ids = list(split_info["test_drug_ids"])
    for feature_space in FEATURE_SPACE_ORDER:
        test_ids = sorted(predictions_df.loc[predictions_df["feature_space"] == feature_space, "Drug_ID"].tolist())
        if test_ids != expected_test_ids:
            raise AssertionError(f"Test Drug_ID mismatch for {feature_space}")

    required_metrics = ["auroc", "auprc", "mcc", "f1_pos", "accuracy"]
    missing = metrics_df[required_metrics].isna().sum().to_dict()
    if any(count > 0 for count in missing.values()):
        raise AssertionError(f"Missing metric values: {missing}")

    return {
        "feature_spaces_present": observed_spaces,
        "feature_counts": feature_counts,
        "shared_test_ids_verified": True,
        "non_missing_metrics": required_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ARTICLE_ROOT)
    parser.add_argument("--output-subdir", default=OUTPUT_DIR_NAME)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--verbosity", type=int, default=2)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictors").mkdir(parents=True, exist_ok=True)

    meta = load_labels(root)
    features, feature_cols, provenance = build_feature_spaces(root, meta)
    cohort = make_shared_labeled_cohort(meta, features)
    train_meta, test_meta, split_info = split_cohort(cohort, seed=args.seed)

    metrics_rows = []
    prediction_frames = []
    leaderboard_full_frames = []
    leaderboard_single_frames = []
    report_frames = []

    for feature_space in FEATURE_SPACE_ORDER:
        metrics_row, pred_df, lb_full, lb_single, report = train_and_evaluate_feature_space(
            feature_space=feature_space,
            feature_df=features[feature_space],
            feature_cols=feature_cols[feature_space],
            train_meta=train_meta,
            test_meta=test_meta,
            out_dir=out_dir,
            time_limit=args.time_limit,
            verbosity=args.verbosity,
        )
        metrics_rows.append(metrics_row)
        prediction_frames.append(pred_df)
        leaderboard_full_frames.append(lb_full)
        leaderboard_single_frames.append(lb_single)
        report_frames.append(report)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["feature_space"] = pd.Categorical(metrics_df["feature_space"], FEATURE_SPACE_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("feature_space").reset_index(drop=True)
    metrics_df["feature_space"] = metrics_df["feature_space"].astype(str)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    leaderboard_full_df = pd.concat(leaderboard_full_frames, ignore_index=True)
    leaderboard_single_df = pd.concat(leaderboard_single_frames, ignore_index=True)
    reports_df = pd.concat(report_frames, ignore_index=True)

    verification = verify_outputs(metrics_df, predictions_df, split_info)

    metrics_df.to_csv(out_dir / "feature_space_test_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "feature_space_test_predictions.csv", index=False)
    leaderboard_full_df.to_csv(out_dir / "feature_space_leaderboard_full.csv", index=False)
    leaderboard_single_df.to_csv(out_dir / "feature_space_single_model_leaderboard.csv", index=False)
    reports_df.to_csv(out_dir / "feature_space_classification_reports.csv", index=False)

    plot_metric_comparison(metrics_df, out_dir)
    plot_roc_overlay(predictions_df, out_dir)
    plot_confusion_matrices(predictions_df, out_dir)

    summary = {
        "analysis": "AutoGluon feature-space comparison for antineoplastic classification",
        "root": str(root),
        "output_dir": str(out_dir),
        "autogluon_version": importlib.metadata.version("autogluon.tabular"),
        "primary_metric": "AUROC",
        "time_limit_per_feature_space_seconds": int(args.time_limit),
        "seed": int(args.seed),
        "class_distribution_all_875": meta["Drug_Class"].value_counts().to_dict(),
        "class_distribution_labeled_shared": cohort["Drug_Class"].value_counts().to_dict(),
        "split_info": split_info,
        "feature_space_order": FEATURE_SPACE_ORDER,
        "feature_space_expected_counts": FEATURE_SPACE_EXPECTED_COUNTS,
        "feature_provenance": provenance,
        "transductive_notice": (
            "The cached DeepHeal SupCon latent representation was learned on the full 875-drug cohort, "
            "and all 448 labeled drugs contributed labels during SupCon representation learning. "
            "Within-cohort SupCon latent classification should therefore be treated as transductive/exploratory, "
            "not as an unbiased generalization estimate."
        ),
        "verification": verification,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(out_dir / "feature_space_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"output_dir": str(out_dir), "verification": verification, "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
