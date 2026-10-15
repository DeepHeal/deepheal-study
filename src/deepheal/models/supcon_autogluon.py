#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
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
MODEL_COLORS = {
    "Traditional": "#4575b4",
    "GBDT": "#74add1",
    "Deep Learning": "#f46d43",
    "Foundation": "#fdae61",
    "Other": "#808080",
}


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
    if any(x in name for x in ["neuralnet", "fttransformer", "nn_torch"]):
        return "Deep Learning"
    if any(x in name for x in ["tabpfn", "tabm", "tabicl"]):
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


def load_embedding(root: Path, results_subdir: str, embedding_file: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    emb = pd.read_csv(root / "results" / "supcon" / embedding_file)
    latent_cols = [c for c in emb.columns if c.startswith("latent_")]
    known = emb[emb["Drug_Class"].isin(["Antineoplastic", "Non-Antineoplastic"])].copy().reset_index(drop=True)
    unknown = emb[emb["Drug_Class"] == "Unknown"].copy().reset_index(drop=True)
    if len(latent_cols) == 0:
        raise ValueError(f"No latent columns found in {embedding_file}")
    return known, unknown, latent_cols


def model_hyperparameters(full: bool = True) -> dict:
    if full:
        return {
            "KNN": {},
            "RF": {},
            "GBM": {},
            "CAT": {},
            "XGB": {},
            "NN_TORCH": {},
            "TABPFNV2": {},
            "TABM": {},
            "TABICL": {},
        }
    return {
        "KNN": {},
        "RF": {},
        "GBM": {},
        "CAT": {},
        "XGB": {},
        "NN_TORCH": {},
    }


def fit_predictor(train_df: pd.DataFrame, model_dir: Path, time_limit: int) -> tuple[TabularPredictor, str]:
    if model_dir.exists():
        shutil.rmtree(model_dir)

    fit_mode = "full"
    try:
        predictor = TabularPredictor(
            label="target",
            path=str(model_dir),
            problem_type="binary",
            eval_metric="accuracy",
            positive_class=POSITIVE_CLASS,
            verbosity=2,
        ).fit(
            train_data=train_df,
            hyperparameters=model_hyperparameters(full=True),
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=time_limit,
            presets="best_quality",
        )
        return predictor, fit_mode
    except Exception as exc:
        print(f"[WARN] Full AutoGluon model zoo failed: {exc}")
        fit_mode = "fallback"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        predictor = TabularPredictor(
            label="target",
            path=str(model_dir),
            problem_type="binary",
            eval_metric="accuracy",
            positive_class=POSITIVE_CLASS,
            verbosity=2,
        ).fit(
            train_data=train_df,
            hyperparameters=model_hyperparameters(full=False),
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=max(180, time_limit // 2),
            presets="best_quality",
        )
        return predictor, fit_mode


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray) -> dict[str, float]:
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


def plot_model_overview(lb: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    lb_sorted = lb.sort_values("score_test", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(lb_sorted))
    colors = [MODEL_COLORS.get(cat, MODEL_COLORS["Other"]) for cat in lb_sorted["category"]]

    ax1 = axes[0, 0]
    bars = ax1.barh(y_pos, lb_sorted["score_test"], color=colors, edgecolor="black", linewidth=0.4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(lb_sorted["model"], fontsize=7)
    ax1.set_xlabel("Test accuracy")
    ax1.set_title("a  Test accuracy ranking", loc="left", fontweight="bold")
    ax1.set_xlim(0, 1.05)
    for bar, score in zip(bars, lb_sorted["score_test"]):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=6)

    ax2 = axes[0, 1]
    lb_time = lb.sort_values("fit_time", ascending=True).reset_index(drop=True)
    colors_time = [MODEL_COLORS.get(cat, MODEL_COLORS["Other"]) for cat in lb_time["category"]]
    bars2 = ax2.barh(np.arange(len(lb_time)), lb_time["fit_time"], color=colors_time, edgecolor="black", linewidth=0.4)
    ax2.set_yticks(np.arange(len(lb_time)))
    ax2.set_yticklabels(lb_time["model"], fontsize=7)
    ax2.set_xlabel("Training time (s)")
    ax2.set_title("b  Training time", loc="left", fontweight="bold")
    ax2.set_xscale("log")
    for bar, val in zip(bars2, lb_time["fit_time"]):
        ax2.text(max(val * 1.05, 0.02), bar.get_y() + bar.get_height() / 2, f"{val:.1f}s", va="center", fontsize=6)

    ax3 = axes[1, 0]
    for cat in lb["category"].unique():
        sub = lb[lb["category"] == cat]
        ax3.scatter(sub["score_val"], sub["score_test"], s=70, color=MODEL_COLORS.get(cat, MODEL_COLORS["Other"]), edgecolor="black", linewidth=0.4, label=cat)
    min_val = max(0.0, min(lb["score_val"].min(), lb["score_test"].min()) - 0.05)
    max_val = min(1.05, max(lb["score_val"].max(), lb["score_test"].max()) + 0.05)
    ax3.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    ax3.set_xlabel("Validation accuracy")
    ax3.set_ylabel("Test accuracy")
    ax3.set_title("c  Validation vs test", loc="left", fontweight="bold")
    ax3.legend(frameon=False, fontsize=7, loc="lower right")
    ax3.set_xlim(min_val, max_val)
    ax3.set_ylim(min_val, max_val)

    ax4 = axes[1, 1]
    for cat in lb["category"].unique():
        sub = lb[lb["category"] == cat]
        ax4.scatter(sub["fit_time"], sub["score_test"], s=70, color=MODEL_COLORS.get(cat, MODEL_COLORS["Other"]), edgecolor="black", linewidth=0.4, label=cat)
    best = lb.iloc[0]
    ax4.scatter(best["fit_time"], best["score_test"], s=180, facecolor="none", edgecolor="red", linewidth=2)
    ax4.annotate(best["model"], (best["fit_time"], best["score_test"]), xytext=(8, 8), textcoords="offset points", fontsize=7, color="red")
    ax4.set_xlabel("Training time (s, log)")
    ax4.set_ylabel("Test accuracy")
    ax4.set_title("d  Accuracy-time tradeoff", loc="left", fontweight="bold")
    ax4.set_xscale("log")

    fig.tight_layout()
    save_figure(fig, out_base)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("SupCon embedding + AutoGluon")
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_roc_curve(y_true: np.ndarray, prob_pos: np.ndarray, out_base: Path) -> None:
    set_style()
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = roc_auc_score(y_true, prob_pos)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#d73027", linewidth=2, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.18, color="#d73027")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("SupCon embedding + AutoGluon ROC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_feature_importance(importance_df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    feature_col = "feature" if "feature" in importance_df.columns else importance_df.columns[0]
    top = importance_df.nlargest(min(20, len(importance_df)), "importance").sort_values("importance", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))

    ax1 = axes[0]
    ax1.barh(top[feature_col].astype(str), top["importance"], color="#4575b4", edgecolor="black", linewidth=0.4)
    ax1.set_xlabel("Importance score")
    ax1.set_title("a  Top latent features", loc="left", fontweight="bold")

    ax2 = axes[1]
    pie_df = importance_df.nlargest(min(10, len(importance_df)), "importance").copy()
    pie_values = pie_df["importance"].fillna(0).clip(lower=0).tolist()
    pie_labels = pie_df[feature_col].astype(str).tolist()
    other_sum = max(0.0, float(importance_df["importance"].fillna(0).sum() - sum(pie_values)))
    if other_sum > 0:
        pie_values.append(other_sum)
        pie_labels.append("Others")
    if sum(pie_values) > 0:
        ax2.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 6})
    else:
        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            "All permutation importances are 0\n(perfectly separable test split)",
            ha="center",
            va="center",
            fontsize=8,
        )
    ax2.set_title("b  Top-10 contribution", loc="left", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ARTICLE_ROOT))
    parser.add_argument("--results-subdir", default="results")
    parser.add_argument("--embedding-file", default="supcon_consensus_embedding.csv")
    parser.add_argument("--model-subdir", default="")
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--fi-subsample-size", type=int, default=64)
    parser.add_argument("--fi-shuffle-sets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "results" / "supcon"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_subdir = str(args.model_subdir).strip() or f"autogluon_{args.results_subdir}"
    model_dir = root / "results" / "autogluon_supcon"

    known, unknown, latent_cols = load_embedding(root, args.results_subdir, args.embedding_file)
    known["target"] = known["Drug_Class"]
    y_binary = (known["Drug_Class"] == POSITIVE_CLASS).astype(int).to_numpy()
    clusters = known["Cluster"].to_numpy(dtype=int)
    stratify_key = build_stratify_key(y_binary, clusters, min_count=2, context="train_test_split")

    idx = np.arange(len(known))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_key,
    )

    train_df = known.iloc[train_idx][latent_cols + ["target"]].reset_index(drop=True)
    test_df = known.iloc[test_idx][latent_cols + ["target"]].reset_index(drop=True)

    predictor, fit_mode = fit_predictor(train_df, model_dir, args.time_limit)

    lb_full = predictor.leaderboard(test_df, silent=True)
    lb_full.to_csv(out_dir / "supcon_autogluon_leaderboard_full.csv", index=False)

    lb = lb_full[~lb_full["model"].str.contains("Ensemble|WeightedEnsemble", case=False, na=False)].copy()
    lb["category"] = lb["model"].map(categorize_model)
    lb = lb.sort_values(["score_test", "score_val"], ascending=False).reset_index(drop=True)
    lb.to_csv(out_dir / "supcon_autogluon_leaderboard_single_models.csv", index=False)

    best_model = str(lb.iloc[0]["model"])
    y_true_label = known.iloc[test_idx]["Drug_Class"].reset_index(drop=True)
    y_pred_label = predictor.predict(test_df[latent_cols], model=best_model).reset_index(drop=True)
    y_proba = predictor.predict_proba(test_df[latent_cols], model=best_model)
    prob_pos = y_proba[POSITIVE_CLASS].to_numpy(dtype=float)
    y_true = (y_true_label == POSITIVE_CLASS).astype(int).to_numpy()
    y_pred = (y_pred_label == POSITIVE_CLASS).astype(int).to_numpy()

    metrics = {
        "fit_mode": fit_mode,
        "best_model": best_model,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **compute_binary_metrics(y_true, y_pred, prob_pos),
    }
    with open(out_dir / "supcon_autogluon_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_df = pd.DataFrame(classification_report(y_true_label, y_pred_label, output_dict=True, zero_division=0)).T
    report_df.to_csv(out_dir / "supcon_autogluon_classification_report.csv")

    known.iloc[test_idx][["Drug_Name", "Drug_Class", "Cluster"]].assign(
        y_pred=y_pred_label.to_numpy(),
        prob_antineoplastic=prob_pos,
        prob_non_antineoplastic=(1.0 - prob_pos),
    ).to_csv(out_dir / "supcon_autogluon_test_predictions.csv", index=False)

    cm = confusion_matrix(y_true_label, y_pred_label, labels=["Antineoplastic", "Non-Antineoplastic"])
    plot_confusion(cm, ["Antineoplastic", "Non-Antineoplastic"], out_dir / "figure_supcon_autogluon_confusion_matrix")
    plot_roc_curve(y_true, prob_pos, out_dir / "figure_supcon_autogluon_roc")
    plot_model_overview(lb, out_dir / "figure_supcon_autogluon_model_overview")

    if len(unknown) > 0:
        unknown_proba = predictor.predict_proba(unknown[latent_cols], model=best_model)
        unknown_pred = predictor.predict(unknown[latent_cols], model=best_model)
        unknown.assign(
            predicted_class=unknown_pred.to_numpy(),
            prob_antineoplastic=unknown_proba[POSITIVE_CLASS].to_numpy(dtype=float),
            prob_non_antineoplastic=unknown_proba["Non-Antineoplastic"].to_numpy(dtype=float),
        ).sort_values("prob_antineoplastic", ascending=False).to_csv(
            out_dir / "supcon_autogluon_unknown_predictions.csv",
            index=False,
        )

    summary = {
        "model": "SupCon latent + AutoGluon",
        "fit_mode": fit_mode,
        "best_model": best_model,
        "n_known": int(len(known)),
        "n_unknown": int(len(unknown)),
        "class_distribution": known["Drug_Class"].value_counts().to_dict(),
        "test_metrics": metrics,
    }
    with open(out_dir / "supcon_autogluon_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    importance_path = out_dir / "supcon_autogluon_feature_importance.csv"
    try:
        importance_df = predictor.feature_importance(
            test_df,
            model=best_model,
            subsample_size=min(int(args.fi_subsample_size), int(len(test_df))),
            num_shuffle_sets=int(args.fi_shuffle_sets),
        )
        importance_export = importance_df.reset_index().rename(columns={"index": "feature"})
        importance_export.to_csv(importance_path, index=False)
        plot_feature_importance(importance_export, out_dir / "figure_supcon_autogluon_feature_importance")
    except Exception as exc:
        print(f"[WARN] Feature importance failed: {exc}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
