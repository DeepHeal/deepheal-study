#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import prototorch as pt
import seaborn as sns
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import LGMLVQ
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
POSITIVE_CLASS = "Antineoplastic"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
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


def load_embedding(embedding_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    emb = pd.read_csv(embedding_path)
    latent_cols = [c for c in emb.columns if c.startswith("latent_")]
    known = emb[emb["Drug_Class"].isin(["Antineoplastic", "Non-Antineoplastic"])].copy().reset_index(drop=True)
    unknown = emb[emb["Drug_Class"] == "Unknown"].copy().reset_index(drop=True)
    if len(latent_cols) == 0:
        raise ValueError(f"No latent columns found in {embedding_path}")
    known["y"] = (known["Drug_Class"] == POSITIVE_CLASS).astype(int)
    class_names = ["Non-Antineoplastic", "Antineoplastic"]
    return known, unknown, latent_cols, class_names


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
):
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


def create_model(hparams: dict, train_dataset: TensorDataset) -> LGMLVQ:
    return LGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_dataset),
    )


def trainer_for(max_epochs: int) -> pl.Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def predict_with_scores(model: LGMLVQ, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    prob_pos: list[np.ndarray] = []
    proto_labels = model.prototype_labels.detach().cpu().numpy()
    classes = sorted(np.unique(proto_labels).tolist())
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            batch_pred = model.predict(batch_x).detach().cpu().numpy()
            distances = model.compute_distances(batch_x).detach().cpu().numpy()
            class_scores = []
            for cls in classes:
                cls_dist = distances[:, proto_labels == cls]
                class_scores.append(-cls_dist.min(axis=1))
            class_scores = np.column_stack(class_scores)
            class_probs = torch.softmax(torch.tensor(class_scores, dtype=torch.float32), dim=1).cpu().numpy()
            pos_idx = classes.index(1)
            preds.append(np.atleast_1d(batch_pred))
            prob_pos.append(np.atleast_1d(class_probs[:, pos_idx]))
    return np.concatenate(preds), np.concatenate(prob_pos)


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


def cross_validate_lgmlvq(
    x_train: np.ndarray,
    y_train: np.ndarray,
    clusters_train: np.ndarray,
    hparams: dict,
    n_splits: int,
    n_repeats: int,
    batch_size: int,
    max_epochs: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        stratify_key = build_stratify_key(
            y_train,
            clusters_train,
            min_count=n_splits,
            context=f"CV repeat {repeat + 1}",
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, stratify_key), start=1):
            x_fold_train = x_train[tr_idx]
            x_fold_val = x_train[va_idx]
            y_fold_train = y_train[tr_idx]
            y_fold_val = y_train[va_idx]

            fold_train_dataset = TensorDataset(torch.FloatTensor(x_fold_train), torch.LongTensor(y_fold_train))
            fold_val_dataset = TensorDataset(torch.FloatTensor(x_fold_val), torch.LongTensor(y_fold_val))
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True)
            fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)

            model = create_model(hparams, fold_train_dataset)
            trainer_for(max_epochs=max_epochs).fit(model, fold_train_loader)
            pred, prob_pos = predict_with_scores(model, fold_val_loader)
            metrics = compute_binary_metrics(y_fold_val, pred, prob_pos)
            rows.append({"repeat": repeat + 1, "fold": fold, **metrics})
            print(
                f"[CV] repeat={repeat + 1}/{n_repeats} fold={fold}/{n_splits} "
                f"accuracy={metrics['accuracy']:.4f} mcc={metrics['mcc']:.4f}"
            )
    return pd.DataFrame(rows)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_base: Path, title_prefix: str) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title_prefix} + LGMLVQ")
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_roc(y_true: np.ndarray, prob_pos: np.ndarray, out_base: Path, title_prefix: str) -> None:
    set_style()
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = roc_auc_score(y_true, prob_pos)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#d73027", linewidth=2, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title_prefix} + LGMLVQ ROC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_base)


def feature_relevance(model: LGMLVQ, latent_cols: list[str]) -> pd.DataFrame:
    omega = model.omega_matrix.detach().cpu().numpy()
    if omega.ndim == 2:
        omega = omega[np.newaxis, :, :]
    lambda_diag = np.diagonal(np.matmul(np.transpose(omega, (0, 2, 1)), omega), axis1=1, axis2=2)
    relevance = lambda_diag.mean(axis=0)
    return (
        pd.DataFrame({"feature": latent_cols, "relevance": relevance})
        .sort_values("relevance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_relevance(relevance_df: pd.DataFrame, out_base: Path, title_prefix: str) -> None:
    set_style()
    top = relevance_df.head(20).sort_values("relevance", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.barh(top["feature"], top["relevance"], color="#2c7fb8", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean relevance")
    ax.set_ylabel("")
    ax.set_title(f"LGMLVQ relevance on {title_prefix.lower()} latents")
    fig.tight_layout()
    save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ARTICLE_ROOT))
    parser.add_argument("--results-subdir", default="results_average_deepprofile_ensemble_1000pc_full600")
    parser.add_argument("--embedding-file", default="average_consensus_embedding.csv")
    parser.add_argument("--output-prefix", default="average")
    parser.add_argument("--title-prefix", default="Average ensemble embedding")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    root = Path(args.root)
    out_dir = root / "results" / "average"
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = out_dir / args.embedding_file

    known, unknown, latent_cols, class_names = load_embedding(embedding_path)
    x = known[latent_cols].to_numpy(dtype=np.float32)
    y = known["y"].to_numpy(dtype=int)
    clusters = known["Cluster"].to_numpy(dtype=int)

    stratify_key = build_stratify_key(y, clusters, min_count=2, context="train_test_split")
    idx = np.arange(len(known))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_key,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    clusters_train = clusters[train_idx]

    train_loader, test_loader, train_dataset, _ = create_dataloaders(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=args.batch_size,
    )

    hparams = {
        "input_dim": int(x_train.shape[1]),
        "latent_dim": int(x_train.shape[1]),
        "distribution": {"num_classes": len(class_names), "per_class": 1},
        "proto_lr": 0.01,
        "bb_lr": 0.01,
    }

    cv_df = cross_validate_lgmlvq(
        x_train=x_train,
        y_train=y_train,
        clusters_train=clusters_train,
        hparams=hparams,
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
    cv_df.to_csv(out_dir / f"{args.output_prefix}_lgmlvq_cv_scores.csv", index=False)
    cv_summary = {
        "n_splits": int(args.cv_splits),
        "n_repeats": int(args.cv_repeats),
        "accuracy_mean": float(cv_df["accuracy"].mean()),
        "accuracy_std": float(cv_df["accuracy"].std(ddof=0)),
        "mcc_mean": float(cv_df["mcc"].mean()),
        "mcc_std": float(cv_df["mcc"].std(ddof=0)),
        "auroc_mean": float(cv_df["auroc"].mean()),
        "auroc_std": float(cv_df["auroc"].std(ddof=0)),
    }
    with open(out_dir / f"{args.output_prefix}_lgmlvq_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    final_model = create_model(hparams, train_dataset)
    trainer_for(max_epochs=args.max_epochs).fit(final_model, train_loader)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "hparams": hparams,
            "class_names": class_names,
            "latent_cols": latent_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        out_dir / f"{args.output_prefix}_lgmlvq_model.pt",
    )

    test_pred, test_prob_pos = predict_with_scores(final_model, test_loader)
    test_metrics = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **compute_binary_metrics(y_test, test_pred, test_prob_pos),
    }
    with open(out_dir / f"{args.output_prefix}_lgmlvq_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    report_df = pd.DataFrame(classification_report(y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0)).T
    report_df.to_csv(out_dir / f"{args.output_prefix}_lgmlvq_classification_report.csv")

    known.iloc[test_idx][["Drug_Name", "Drug_Class", "Cluster"]].assign(
        y_true=y_test,
        y_pred=test_pred,
        prob_antineoplastic=test_prob_pos,
    ).to_csv(out_dir / f"{args.output_prefix}_lgmlvq_test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, test_pred)
    plot_confusion(cm, class_names, out_dir / f"figure_{args.output_prefix}_lgmlvq_confusion_matrix", args.title_prefix)
    plot_roc(y_test, test_prob_pos, out_dir / f"figure_{args.output_prefix}_lgmlvq_roc", args.title_prefix)

    relevance_df = feature_relevance(final_model, latent_cols)
    relevance_df.to_csv(out_dir / f"{args.output_prefix}_lgmlvq_feature_relevance.csv", index=False)
    plot_feature_relevance(relevance_df, out_dir / f"figure_{args.output_prefix}_lgmlvq_feature_relevance", args.title_prefix)

    if len(unknown) > 0:
        x_unknown = scaler.transform(unknown[latent_cols].to_numpy(dtype=np.float32))
        unknown_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_unknown), torch.zeros(len(unknown), dtype=torch.long)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        unk_pred, unk_prob_pos = predict_with_scores(final_model, unknown_loader)
        unk_class = np.where(unk_pred == 1, POSITIVE_CLASS, "Non-Antineoplastic")
        unknown.assign(
            predicted_class=unk_class,
            prob_antineoplastic=unk_prob_pos,
            prob_non_antineoplastic=1.0 - unk_prob_pos,
        ).sort_values("prob_antineoplastic", ascending=False).to_csv(
            out_dir / f"{args.output_prefix}_lgmlvq_unknown_predictions.csv",
            index=False,
        )

    summary = {
        "model": f"{args.title_prefix} + LGMLVQ",
        "embedding_file": str(embedding_path),
        "input_features": int(len(latent_cols)),
        "n_known": int(len(known)),
        "n_unknown": int(len(unknown)),
        "class_distribution": known["Drug_Class"].value_counts().to_dict(),
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }
    with open(out_dir / f"{args.output_prefix}_lgmlvq_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
