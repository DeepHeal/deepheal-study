#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from sklearn.model_selection import train_test_split


OUTPUT_DIR_NAME = 'autogluon_expanded_model_zoo_comparison'
BASE_SCRIPT = 'feature_space_comparison.py'

MODEL_REQUESTS = [
    {
        "ag_key": "RF",
        "user_model_class": "RFModel",
        "display_name": "RandomForest",
        "expected_autogluon_model": "RandomForest",
        "category": "Traditional",
    },
    {
        "ag_key": "KNN",
        "user_model_class": "KNNModel",
        "display_name": "KNeighbors",
        "expected_autogluon_model": "KNeighbors",
        "category": "Traditional",
    },
    {
        "ag_key": "GBM",
        "user_model_class": "LGBModel",
        "display_name": "LightGBM",
        "expected_autogluon_model": "LightGBM",
        "category": "GBDT",
    },
    {
        "ag_key": "CAT",
        "user_model_class": "CatBoostModel",
        "display_name": "CatBoost",
        "expected_autogluon_model": "CatBoost",
        "category": "GBDT",
    },
    {
        "ag_key": "XGB",
        "user_model_class": "XGBoostModel",
        "display_name": "XGBoost",
        "expected_autogluon_model": "XGBoost",
        "category": "GBDT",
    },
    {
        "ag_key": "REALMLP",
        "user_model_class": "RealMLPModel",
        "display_name": "RealMLP",
        "expected_autogluon_model": "RealMLP",
        "category": "Deep Learning",
    },
    {
        "ag_key": "TABM",
        "user_model_class": "TabMModel",
        "display_name": "TabM",
        "expected_autogluon_model": "TabM",
        "category": "Foundation",
    },
    {
        "ag_key": "TABICL",
        "user_model_class": "TabICLModel",
        "display_name": "TabICL",
        "expected_autogluon_model": "TabICL",
        "category": "Foundation",
    },
    {
        "ag_key": "REALTABPFN-V2",
        "user_model_class": "TabPFNV2Model",
        "display_name": "TabPFNv2",
        "expected_autogluon_model": "RealTabPFN-v2",
        "category": "Foundation",
        "note": "AutoGluon 1.5.0 registers this as REALTABPFN-V2, not TABPFNV2.",
    },
    {
        "ag_key": "REALTABPFN-V2.5",
        "user_model_class": "RealTabPFNv25Model",
        "display_name": "RealTabPFN-v2.5",
        "expected_autogluon_model": "RealTabPFN-v2.5",
        "category": "Foundation",
        "note": "AutoGluon warns that TabPFN-2.5 is noncommercial.",
    },
    {
        "ag_key": "NN_TORCH",
        "user_model_class": "TabularNeuralNetTorchModel",
        "display_name": "NeuralNetTorch",
        "expected_autogluon_model": "NeuralNetTorch",
        "category": "Deep Learning",
    },
    {
        "ag_key": "FT_TRANSFORMER",
        "user_model_class": "FTTransformerModel",
        "display_name": "FTTransformer",
        "expected_autogluon_model": "FTTransformer",
        "category": "Deep Learning",
    },
    {
        "ag_key": "TABDPT",
        "user_model_class": "TabDPTModel",
        "display_name": "TabDPT",
        "expected_autogluon_model": "TabDPT",
        "category": "Foundation",
    },
]

AG_TO_DISPLAY = {item["ag_key"]: item["display_name"] for item in MODEL_REQUESTS}
MODEL_TO_REQUEST = {item["expected_autogluon_model"].lower(): item for item in MODEL_REQUESTS}
DISPLAY_ORDER = [item["display_name"] for item in MODEL_REQUESTS]
INTERNAL_FEATURE_LIMITS = {
    # FTTransformer exposes this inside AutoMM fit validation rather than the tabular registry metadata.
    "FT_TRANSFORMER": 300,
}
FIGURE_FEATURE_LABELS = {
    "pca_200": "PCA-200",
    "supcon_latent": "SupCon\nlatent",
    "average_latent": "Average\nlatent",
    "full_protein": "Full\nprotein",
}
PRIMARY_METRIC_CONFIG = {
    "roc_auc": {
        "column": "auroc",
        "label": "Held-out AUROC",
        "short_label": "AUROC",
        "slug": "auroc",
        "family_title": "Single-model AUROC by model family",
        "heatmap_title": "Expanded AutoGluon model zoo comparison",
        "ylim": (0.35, 1.04),
        "heatmap_vmin": 0.45,
    },
    "accuracy": {
        "column": "accuracy",
        "label": "Held-out accuracy",
        "short_label": "accuracy",
        "slug": "accuracy",
        "family_title": "Single-model accuracy by model family",
        "heatmap_title": "Expanded AutoGluon model zoo accuracy comparison",
        "ylim": (0.45, 1.04),
        "heatmap_vmin": 0.45,
    },
}


def load_base_module(root: Path) -> ModuleType:
    script = root / 'src/deepheal/benchmarking' / BASE_SCRIPT
    spec = importlib.util.spec_from_file_location("deepheal_feature_space_comparison", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base script from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_windows_nvml() -> str | None:
    nvml = Path(__import__("os").environ.get("WINDIR", "")) / "System32" / "nvml.dll"
    if not nvml.exists():
        return None
    try:
        import pynvml

        pynvml.nvmlLib = ctypes.CDLL(str(nvml))
        return str(nvml)
    except Exception as exc:  # pragma: no cover - recorded in summary for local diagnostics.
        return f"patch_failed: {exc!r}"


def registered_model_metadata() -> dict[str, dict[str, object]]:
    registry = ag_model_registry.key_to_cls_map()
    metadata: dict[str, dict[str, object]] = {}
    for request in MODEL_REQUESTS:
        ag_key = request["ag_key"]
        cls = registry.get(ag_key)
        if cls is None:
            metadata[ag_key] = {
                "registered": False,
                "max_features": None,
                "class_name": None,
                "module": None,
            }
            continue
        max_features = None
        try:
            model = cls()
            aux = model._get_default_auxiliary_params()
            max_features = aux.get("max_features")
        except Exception:
            max_features = None
        metadata[ag_key] = {
            "registered": True,
            "max_features": max_features,
            "class_name": cls.__name__,
            "module": cls.__module__,
        }
    return metadata


def hyperparameters_for_feature_count(
    feature_count: int,
    metadata: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    hyperparameters: dict[str, dict[str, object]] = {}
    skipped: list[dict[str, object]] = []
    for request in MODEL_REQUESTS:
        ag_key = request["ag_key"]
        model_meta = metadata[ag_key]
        if not model_meta["registered"]:
            skipped.append(
                {
                    **request,
                    "status": "skipped_unregistered",
                    "reason": f"AutoGluon key {ag_key} is not registered.",
                }
            )
            continue
        max_features = model_meta.get("max_features")
        if ag_key in INTERNAL_FEATURE_LIMITS:
            internal_limit = INTERNAL_FEATURE_LIMITS[ag_key]
            max_features = internal_limit if max_features is None else min(int(max_features), internal_limit)
        if max_features is not None and feature_count > int(max_features):
            skipped.append(
                {
                    **request,
                    "status": "skipped_feature_limit",
                    "reason": f"ag.max_features={max_features} < feature_count={feature_count}.",
                }
            )
            continue
        if ag_key == "TABDPT":
            hyperparameters[ag_key] = {"ag_args_fit": {"num_gpus": 0}}
        else:
            hyperparameters[ag_key] = {}
    return hyperparameters, skipped


def model_request_for_autogluon_name(model_name: str) -> dict[str, object] | None:
    return MODEL_TO_REQUEST.get(model_name.lower())


def fit_expanded_predictor(
    train_df: pd.DataFrame,
    model_dir: Path,
    hyperparameters: dict[str, dict[str, object]],
    time_limit: int,
    verbosity: int,
    positive_class: str,
    eval_metric: str,
    reuse_existing: bool,
) -> TabularPredictor:
    if reuse_existing and (model_dir / "predictor.pkl").exists():
        print(f"[INFO] Reusing existing predictor at {model_dir}")
        return TabularPredictor.load(str(model_dir))
    if model_dir.exists():
        shutil.rmtree(model_dir)
    return TabularPredictor(
        label="target",
        path=str(model_dir),
        problem_type="binary",
        eval_metric=eval_metric,
        positive_class=positive_class,
        verbosity=verbosity,
    ).fit(
        train_data=train_df,
        hyperparameters=hyperparameters,
        num_bag_folds=0,
        num_stack_levels=0,
        time_limit=time_limit,
        presets="best_quality",
        raise_on_no_models_fitted=False,
    )


def evaluate_feature_space(
    base: ModuleType,
    feature_space: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    out_dir: Path,
    metadata: dict[str, dict[str, object]],
    time_limit: int,
    verbosity: int,
    eval_metric: str,
    reuse_existing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_count = len(feature_cols)
    hyperparameters, skipped = hyperparameters_for_feature_count(feature_count, metadata)
    if not hyperparameters:
        raise RuntimeError(f"No trainable expanded model keys remain for {feature_space}.")

    print(
        f"[INFO] Training expanded model zoo for {feature_space}: "
        f"{feature_count} features, {len(hyperparameters)} requested models, eval_metric={eval_metric}"
    )
    model_dir = out_dir / "predictors" / feature_space
    train_df = base.assemble_autogluon_frame(train_meta, feature_df, feature_cols)
    test_df = base.assemble_autogluon_frame(test_meta, feature_df, feature_cols)

    predictor = fit_expanded_predictor(
        train_df=train_df,
        model_dir=model_dir,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        verbosity=verbosity,
        positive_class=base.POSITIVE_CLASS,
        eval_metric=eval_metric,
        reuse_existing=reuse_existing,
    )

    leaderboard_full = predictor.leaderboard(test_df, silent=True)
    leaderboard_full.insert(0, "feature_space", feature_space)
    leaderboard_full.insert(1, "feature_space_label", base.FEATURE_SPACE_LABELS[feature_space])
    leaderboard_full["feature_count"] = feature_count

    single = leaderboard_full[
        ~leaderboard_full["model"].str.contains("Ensemble|WeightedEnsemble", case=False, na=False)
    ].copy()
    single = single.sort_values(["score_test", "score_val"], ascending=False).reset_index(drop=True)

    x_test = test_df[feature_cols]
    y_true_label = test_meta["Drug_Class"].reset_index(drop=True)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    fitted_expected: set[str] = set()
    for row in single.itertuples():
        request = model_request_for_autogluon_name(str(row.model))
        display_name = request["display_name"] if request else str(row.model)
        category = request["category"] if request else base.categorize_model(str(row.model))
        y_pred_label = predictor.predict(x_test, model=str(row.model)).reset_index(drop=True)
        prob_pos = base.positive_probabilities(predictor, x_test, model=str(row.model))
        metrics = base.compute_binary_metrics(y_true_label, y_pred_label, prob_pos)
        metric_rows.append(
            {
                "feature_space": feature_space,
                "feature_space_label": base.FEATURE_SPACE_LABELS[feature_space],
                "feature_count": feature_count,
                "ag_key": request["ag_key"] if request else None,
                "user_model_class": request["user_model_class"] if request else None,
                "display_model": display_name,
                "autogluon_model": str(row.model),
                "category": category,
                "score_test": float(row.score_test),
                "score_val": float(row.score_val) if not pd.isna(row.score_val) else np.nan,
                "fit_time": float(row.fit_time) if not pd.isna(row.fit_time) else np.nan,
                "pred_time_test": float(row.pred_time_test) if not pd.isna(row.pred_time_test) else np.nan,
                **metrics,
            }
        )
        if request:
            fitted_expected.add(request["ag_key"])
        pred_df = test_meta[["Drug_ID", "Sample_ID", "Drug_Name", "Drug_Class", "Cluster"]].copy().reset_index(drop=True)
        pred_df.insert(0, "feature_space", feature_space)
        pred_df.insert(1, "feature_space_label", base.FEATURE_SPACE_LABELS[feature_space])
        pred_df["ag_key"] = request["ag_key"] if request else None
        pred_df["display_model"] = display_name
        pred_df["autogluon_model"] = str(row.model)
        pred_df["y_pred"] = y_pred_label.to_numpy()
        pred_df["prob_antineoplastic"] = prob_pos
        pred_df["prob_non_antineoplastic"] = 1.0 - prob_pos
        prediction_rows.append(pred_df)

    coverage_rows: list[dict[str, object]] = []
    skipped_by_key = {item["ag_key"]: item for item in skipped}
    for request in MODEL_REQUESTS:
        ag_key = request["ag_key"]
        if ag_key in fitted_expected:
            status = "fitted"
            reason = ""
        elif ag_key in skipped_by_key:
            status = str(skipped_by_key[ag_key]["status"])
            reason = str(skipped_by_key[ag_key]["reason"])
        else:
            status = "missing_after_fit"
            reason = "AutoGluon did not return a fitted model in the leaderboard."
        coverage_rows.append(
            {
                "feature_space": feature_space,
                "feature_space_label": base.FEATURE_SPACE_LABELS[feature_space],
                "feature_count": feature_count,
                "ag_key": ag_key,
                "user_model_class": request["user_model_class"],
                "display_model": request["display_name"],
                "expected_autogluon_model": request["expected_autogluon_model"],
                "category": request["category"],
                "status": status,
                "reason": reason,
                "registered": bool(metadata[ag_key]["registered"]),
                "max_features": metadata[ag_key]["max_features"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows)
    return metrics_df, predictions_df, leaderboard_full, coverage_df


def split_cohort_for_run(
    base: ModuleType,
    cohort: pd.DataFrame,
    seed: int,
    split_order: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if split_order == "sorted":
        train_meta, test_meta, split_info = base.split_cohort(cohort, seed=seed)
        split_info["split_order"] = "sorted_by_drug_id_after_split"
        return train_meta, test_meta, split_info

    y_binary = (cohort["Drug_Class"] == base.POSITIVE_CLASS).astype(int).to_numpy()
    clusters = cohort["Cluster"].astype(int).to_numpy()
    stratify_key = base.build_stratify_key(
        y_binary,
        clusters,
        min_count=2,
        context="expanded_accuracy_split_order",
    )
    indices = np.arange(len(cohort))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_key,
    )
    train_meta = cohort.iloc[train_idx].reset_index(drop=True)
    test_meta = cohort.iloc[test_idx].reset_index(drop=True)
    return train_meta, test_meta, {
        "seed": int(seed),
        "test_size": 0.2,
        "n_labeled": int(len(cohort)),
        "n_train": int(len(train_meta)),
        "n_test": int(len(test_meta)),
        "stratification": "Drug_Class binary label plus Cluster with sparse-strata merge",
        "split_order": "train_test_split_order_preserved",
        "train_drug_ids": sorted(train_meta["Drug_ID"].tolist()),
        "test_drug_ids": sorted(test_meta["Drug_ID"].tolist()),
        "test_class_counts": test_meta["Drug_Class"].value_counts().to_dict(),
    }


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="ticks", context="paper")


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_metric_heatmap(
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    base: ModuleType,
    out_dir: Path,
    eval_metric: str,
) -> None:
    apply_figure_style()
    metric_config = PRIMARY_METRIC_CONFIG[eval_metric]
    pivot = metrics.pivot_table(
        index="display_model",
        columns="feature_space",
        values=metric_config["column"],
        aggfunc="first",
    ).reindex(index=DISPLAY_ORDER, columns=base.FEATURE_SPACE_ORDER)
    labels = pivot.copy()
    annot = labels.map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    for row in coverage.itertuples():
        if row.status != "fitted":
            annot.loc[row.display_model, row.feature_space] = "N/A"

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt="",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": metric_config["label"], "shrink": 0.75},
        vmin=metric_config["heatmap_vmin"],
        vmax=1.0,
    )
    for i, model in enumerate(DISPLAY_ORDER):
        for j, feature_space in enumerate(base.FEATURE_SPACE_ORDER):
            if pd.isna(pivot.loc[model, feature_space]):
                ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center", fontsize=6.5, color="#4B5563")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([FIGURE_FEATURE_LABELS[x] for x in base.FEATURE_SPACE_ORDER], rotation=0)
    ax.set_title(metric_config["heatmap_title"])
    fig.tight_layout()
    save_figure(fig, out_dir / f"figure_expanded_model_zoo_{metric_config['slug']}_heatmap")


def plot_model_family_points(metrics: pd.DataFrame, base: ModuleType, out_dir: Path, eval_metric: str) -> None:
    apply_figure_style()
    metric_config = PRIMARY_METRIC_CONFIG[eval_metric]
    plot_df = metrics.copy()
    plot_df["figure_feature_space_label"] = plot_df["feature_space"].map(FIGURE_FEATURE_LABELS)
    plot_df["feature_space_label"] = pd.Categorical(
        plot_df["figure_feature_space_label"],
        categories=[FIGURE_FEATURE_LABELS[x] for x in base.FEATURE_SPACE_ORDER],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    sns.stripplot(
        data=plot_df,
        x="feature_space_label",
        y=metric_config["column"],
        hue="category",
        dodge=True,
        size=4.2,
        linewidth=0.35,
        edgecolor="black",
        alpha=0.9,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric_config["label"])
    ax.set_ylim(*metric_config["ylim"])
    ax.set_title(metric_config["family_title"])
    ax.legend(frameon=False, title="", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.45, alpha=0.75)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.26)
    save_figure(fig, out_dir / f"figure_expanded_model_zoo_{metric_config['slug']}_family_points")


def verify_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coverage: pd.DataFrame,
    split_info: dict[str, object],
    base: ModuleType,
) -> dict[str, object]:
    feature_spaces = sorted(metrics["feature_space"].unique().tolist())
    if feature_spaces != sorted(base.FEATURE_SPACE_ORDER):
        raise AssertionError(f"Unexpected feature spaces in metrics: {feature_spaces}")

    expected_test_ids = list(split_info["test_drug_ids"])
    for (feature_space, model), sub in predictions.groupby(["feature_space", "display_model"]):
        test_ids = sorted(sub["Drug_ID"].tolist())
        if test_ids != expected_test_ids:
            raise AssertionError(f"Test Drug_ID mismatch for {feature_space} / {model}")

    required_metrics = ["auroc", "auprc", "mcc", "f1_pos", "accuracy"]
    missing = metrics[required_metrics].isna().sum().to_dict()
    if any(count > 0 for count in missing.values()):
        raise AssertionError(f"Missing expanded model metrics: {missing}")

    unexpected_missing = coverage[coverage["status"] == "missing_after_fit"]
    if len(unexpected_missing) > 0:
        missing_models = unexpected_missing[["feature_space", "display_model", "reason"]].to_dict(orient="records")
        raise AssertionError(f"Supported requested models missing after fit: {missing_models}")

    fitted_counts = coverage[coverage["status"] == "fitted"].groupby("feature_space").size().to_dict()
    skipped_counts = coverage[coverage["status"].str.startswith("skipped")].groupby("feature_space").size().to_dict()
    return {
        "feature_spaces_present": feature_spaces,
        "shared_test_ids_verified": True,
        "non_missing_metrics": required_metrics,
        "fitted_model_counts": fitted_counts,
        "skipped_model_counts": skipped_counts,
        "coverage_status_counts": coverage["status"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ARTICLE_ROOT)
    parser.add_argument("--output-subdir", default=OUTPUT_DIR_NAME)
    parser.add_argument("--time-limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--eval-metric", choices=sorted(PRIMARY_METRIC_CONFIG), default="roc_auc")
    parser.add_argument(
        "--split-order",
        choices=["sorted", "train_test_split"],
        default="sorted",
        help=(
            "Use sorted train/test rows after splitting, or preserve the raw train_test_split row order. "
            "The latter matches the older Average-only AutoGluon comparison's internal validation behavior."
        ),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    base = load_base_module(root)
    out_dir = root / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictors").mkdir(parents=True, exist_ok=True)

    nvml_patch = patch_windows_nvml()
    metadata = registered_model_metadata()

    meta = base.load_labels(root)
    features, feature_cols, provenance = base.build_feature_spaces(root, meta)
    cohort = base.make_shared_labeled_cohort(meta, features)
    train_meta, test_meta, split_info = split_cohort_for_run(
        base=base,
        cohort=cohort,
        seed=args.seed,
        split_order=args.split_order,
    )

    metrics_frames = []
    prediction_frames = []
    leaderboard_frames = []
    coverage_frames = []
    for feature_space in base.FEATURE_SPACE_ORDER:
        metrics_df, predictions_df, leaderboard_df, coverage_df = evaluate_feature_space(
            base=base,
            feature_space=feature_space,
            feature_df=features[feature_space],
            feature_cols=feature_cols[feature_space],
            train_meta=train_meta,
            test_meta=test_meta,
            out_dir=out_dir,
            metadata=metadata,
            time_limit=args.time_limit,
            verbosity=args.verbosity,
            eval_metric=args.eval_metric,
            reuse_existing=args.reuse_existing,
        )
        metrics_frames.append(metrics_df)
        prediction_frames.append(predictions_df)
        leaderboard_frames.append(leaderboard_df)
        coverage_frames.append(coverage_df)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)

    metrics["feature_space"] = pd.Categorical(metrics["feature_space"], base.FEATURE_SPACE_ORDER, ordered=True)
    metrics["display_model"] = pd.Categorical(metrics["display_model"], DISPLAY_ORDER, ordered=True)
    metrics = metrics.sort_values(["feature_space", "display_model"]).reset_index(drop=True)
    metrics["feature_space"] = metrics["feature_space"].astype(str)
    metrics["display_model"] = metrics["display_model"].astype(str)

    verification = verify_outputs(metrics, predictions, coverage, split_info, base)

    metrics.to_csv(out_dir / "expanded_model_zoo_test_metrics.csv", index=False)
    predictions.to_csv(out_dir / "expanded_model_zoo_test_predictions.csv", index=False)
    leaderboard.to_csv(out_dir / "expanded_model_zoo_leaderboard_full.csv", index=False)
    coverage.to_csv(out_dir / "expanded_model_zoo_model_coverage.csv", index=False)

    plot_model_metric_heatmap(metrics, coverage, base, out_dir, eval_metric=args.eval_metric)
    plot_model_family_points(metrics, base, out_dir, eval_metric=args.eval_metric)

    summary = {
        "analysis": "Expanded AutoGluon model-zoo comparison for DeepHeal feature spaces",
        "root": str(root),
        "output_dir": str(out_dir),
        "autogluon_version": importlib.metadata.version("autogluon.tabular"),
        "eval_metric": args.eval_metric,
        "primary_metric_column": PRIMARY_METRIC_CONFIG[args.eval_metric]["column"],
        "split_order": args.split_order,
        "time_limit_per_feature_space_seconds": int(args.time_limit),
        "seed": int(args.seed),
        "nvml_patch": nvml_patch,
        "model_requests": MODEL_REQUESTS,
        "registered_model_metadata": metadata,
        "split_info": split_info,
        "feature_space_order": base.FEATURE_SPACE_ORDER,
        "feature_space_expected_counts": base.FEATURE_SPACE_EXPECTED_COUNTS,
        "feature_provenance": provenance,
        "transductive_notice": (
            "The cached DeepHeal SupCon latent representation was learned on the full 875-drug cohort, "
            "and all 448 labeled drugs contributed labels during SupCon representation learning. "
            "Within-cohort SupCon latent classification should therefore be treated as transductive/exploratory."
        ),
        "verification": verification,
    }
    with open(out_dir / "expanded_model_zoo_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({"output_dir": str(out_dir), "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
