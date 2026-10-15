#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.metadata
import importlib.util
import json
import pickle
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


BASE_SCRIPT = 'feature_space_comparison.py'
EXPANDED_SCRIPT = 'model_zoo_comparison.py'
OUTPUT_DIR_NAME = 'autogluon_protargetminer_four_spaces'
POSITIVE_CLASS = "Antineoplastic"
NEGATIVE_CLASS = "Non-Antineoplastic"

PANEL_SPECS = [
    {
        "dataset": "supplementary_data1",
        "scope": "A549 original 55-drug dataset",
        "panel_slug": "protargetminer_supplementary_data1_a549_original_55drugs",
        "expected_compounds": 55,
    },
    {
        "dataset": "supplementary_data4",
        "scope": "A549 deep 9-drug dataset",
        "panel_slug": "protargetminer_supplementary_data4_a549_deep_9drugs",
        "expected_compounds": 9,
    },
    {
        "dataset": "supplementary_data5",
        "scope": "MCF-7 deep 9-drug dataset",
        "panel_slug": "protargetminer_supplementary_data5_mcf7_deep_9drugs",
        "expected_compounds": 9,
    },
    {
        "dataset": "supplementary_data6",
        "scope": "RKO deep 9-drug dataset",
        "panel_slug": "protargetminer_supplementary_data6_rko_deep_9drugs",
        "expected_compounds": 9,
    },
]


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_module(root: Path) -> ModuleType:
    return load_module("deepheal_feature_space_comparison_for_external", root / 'src/deepheal/benchmarking' / BASE_SCRIPT)


def load_expanded_module(root: Path) -> ModuleType:
    return load_module("deepheal_expanded_model_zoo_for_external", root / 'src/deepheal/benchmarking' / EXPANDED_SCRIPT)


def read_conversion_manifest(root: Path) -> dict[str, object]:
    path = (
        root / 'data/external/deepcovermoa_protargetminer_all/processed/conversion_manifest.json'
    )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def supplement_manifest_by_dataset(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(item["dataset"]): item for item in manifest["supplementary_outputs"]}


def values_frame_with_meta(
    compounds: pd.DataFrame,
    values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(values, columns=feature_names)
    out.insert(0, "compound_norm", compounds["compound_norm"].to_numpy())
    out.insert(0, "Compound", compounds["Compound"].to_numpy())
    return out


def load_external_feature_spaces(
    root: Path,
    panel_slug: str,
    feature_cols: dict[str, list[str]],
    pca,
    external_feature_root: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    feature_root = external_feature_root or root / "results" / "external_panel_predictions"
    table_dir = feature_root / panel_slug / "tables"
    aligned_path = table_dir / f"{panel_slug}_aligned_threshold015.csv"
    supcon_path = table_dir / f"{panel_slug}_supcon_projected_embeddings.csv"
    average_path = table_dir / f"{panel_slug}_average_projected_embeddings.csv"
    coverage_path = table_dir / f"{panel_slug}_coverage_summary.csv"

    for path in [aligned_path, supcon_path, average_path, coverage_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required external-panel feature file is missing: {path}")

    aligned = pd.read_csv(aligned_path)
    if not {"Compound", "compound_norm"}.issubset(aligned.columns):
        raise ValueError(f"Aligned matrix is missing Compound/compound_norm columns: {aligned_path}")
    compound_meta = aligned[["Compound", "compound_norm"]].copy()
    raw_cols = [col for col in aligned.columns if col not in {"Compound", "compound_norm"}]
    if len(raw_cols) != len(feature_cols["full_protein"]):
        raise ValueError(
            f"{panel_slug} aligned feature count {len(raw_cols)} does not match "
            f"full-protein count {len(feature_cols['full_protein'])}"
        )
    raw_values = aligned[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    pca_values = pca.transform(raw_values).astype(np.float32)[:, : len(feature_cols["pca_200"])]
    full_df = values_frame_with_meta(compound_meta, raw_values, feature_cols["full_protein"])
    pca_df = values_frame_with_meta(compound_meta, pca_values, feature_cols["pca_200"])

    supcon = pd.read_csv(supcon_path)
    average = pd.read_csv(average_path)
    for name, df, cols, path in [
        ("supcon_latent", supcon, feature_cols["supcon_latent"], supcon_path),
        ("average_latent", average, feature_cols["average_latent"], average_path),
    ]:
        missing = [col for col in ["Compound", "compound_norm", *cols] if col not in df.columns]
        if missing:
            raise ValueError(f"{name} file is missing {len(missing)} required columns in {path}: {missing[:5]}")

    coverage = pd.read_csv(coverage_path).iloc[0].to_dict()
    external_features = {
        "pca_200": pca_df,
        "supcon_latent": supcon[["Compound", "compound_norm", *feature_cols["supcon_latent"]]].copy(),
        "average_latent": average[["Compound", "compound_norm", *feature_cols["average_latent"]]].copy(),
        "full_protein": full_df,
    }
    return external_features, coverage


def train_feature_space_predictors(
    base: ModuleType,
    expanded: ModuleType,
    features: dict[str, pd.DataFrame],
    feature_cols: dict[str, list[str]],
    cohort: pd.DataFrame,
    out_dir: Path,
    metadata: dict[str, dict[str, object]],
    time_limit: int,
    verbosity: int,
    eval_metric: str,
    reuse_existing: bool,
) -> tuple[dict[str, TabularPredictor], pd.DataFrame, pd.DataFrame]:
    predictors: dict[str, TabularPredictor] = {}
    train_rows: list[dict[str, object]] = []
    coverage_frames: list[pd.DataFrame] = []

    for feature_space in base.FEATURE_SPACE_ORDER:
        cols = feature_cols[feature_space]
        hyperparameters, skipped = expanded.hyperparameters_for_feature_count(len(cols), metadata)
        if not hyperparameters:
            raise RuntimeError(f"No AutoGluon models remain after feature-limit filtering for {feature_space}")

        train_df = base.assemble_autogluon_frame(cohort, features[feature_space], cols)
        model_dir = out_dir / "predictors" / feature_space
        print(
            f"[INFO] Training {feature_space}: {len(cols)} features, "
            f"{len(hyperparameters)} model requests, n={len(train_df)}"
        )
        predictor = expanded.fit_expanded_predictor(
            train_df=train_df,
            model_dir=model_dir,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
            verbosity=verbosity,
            positive_class=POSITIVE_CLASS,
            eval_metric=eval_metric,
            reuse_existing=reuse_existing,
        )
        predictors[feature_space] = predictor
        leaderboard = predictor.leaderboard(train_df, silent=True)
        best_model = str(getattr(predictor, "model_best", "") or leaderboard.iloc[0]["model"])

        train_rows.append(
            {
                "feature_space": feature_space,
                "feature_space_label": base.FEATURE_SPACE_LABELS[feature_space],
                "feature_count": int(len(cols)),
                "n_train_labeled": int(len(train_df)),
                "eval_metric": eval_metric,
                "best_model": best_model,
                "model_dir": str(model_dir),
                "n_requested_models_after_feature_filter": int(len(hyperparameters)),
                "n_skipped_models_by_feature_filter": int(len(skipped)),
            }
        )

        coverage_rows = []
        skipped_by_key = {item["ag_key"]: item for item in skipped}
        fitted_models = set(leaderboard["model"].astype(str))
        for request in expanded.MODEL_REQUESTS:
            ag_key = request["ag_key"]
            expected = request["expected_autogluon_model"]
            status = "fitted" if any(model.startswith(expected) for model in fitted_models) else "not_returned"
            reason = ""
            if ag_key in skipped_by_key:
                status = str(skipped_by_key[ag_key]["status"])
                reason = str(skipped_by_key[ag_key]["reason"])
            coverage_rows.append(
                {
                    "feature_space": feature_space,
                    "feature_count": int(len(cols)),
                    "ag_key": ag_key,
                    "display_model": request["display_name"],
                    "expected_autogluon_model": expected,
                    "status": status,
                    "reason": reason,
                }
            )
        coverage_frames.append(pd.DataFrame(coverage_rows))

    return predictors, pd.DataFrame(train_rows), pd.concat(coverage_frames, ignore_index=True)


def load_saved_feature_space_predictors(
    root: Path,
    base: ModuleType,
    source_output_subdir: str,
) -> tuple[dict[str, TabularPredictor], pd.DataFrame, pd.DataFrame]:
    source_dir = root / "results" / source_output_subdir
    predictor_root = source_dir / "predictors"
    if not predictor_root.exists():
        raise FileNotFoundError(f"Missing predictor source directory: {predictor_root}")

    predictors: dict[str, TabularPredictor] = {}
    for feature_space in base.FEATURE_SPACE_ORDER:
        predictor_dir = predictor_root / feature_space
        if not predictor_dir.exists():
            raise FileNotFoundError(f"Missing saved predictor directory: {predictor_dir}")
        predictors[feature_space] = TabularPredictor.load(str(predictor_dir), require_version_match=False)

    train_summary_path = source_dir / "external_autogluon_training_summary.csv"
    model_coverage_path = source_dir / "external_autogluon_model_coverage.csv"
    if not train_summary_path.exists():
        raise FileNotFoundError(train_summary_path)
    if not model_coverage_path.exists():
        raise FileNotFoundError(model_coverage_path)

    train_summary = pd.read_csv(train_summary_path)
    model_coverage = pd.read_csv(model_coverage_path)
    observed_spaces = set(train_summary["feature_space"].astype(str))
    expected_spaces = set(base.FEATURE_SPACE_ORDER)
    if observed_spaces != expected_spaces:
        raise AssertionError(f"Unexpected saved predictor feature spaces: {sorted(observed_spaces)}")
    return predictors, train_summary, model_coverage


def predict_external_panels(
    base: ModuleType,
    predictors: dict[str, TabularPredictor],
    feature_cols: dict[str, list[str]],
    root: Path,
    panel_specs: list[dict[str, object]],
    manifest_by_dataset: dict[str, dict[str, object]],
    external_feature_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(root / "results" / "supcon" / "supcon_pca.pkl", "rb") as handle:
        pca = pickle.load(handle)

    prediction_frames: list[pd.DataFrame] = []
    panel_rows: list[dict[str, object]] = []

    for spec in panel_specs:
        dataset = str(spec["dataset"])
        panel_slug = str(spec["panel_slug"])
        manifest_item = manifest_by_dataset[dataset]
        external_features, coverage = load_external_feature_spaces(
            root,
            panel_slug,
            feature_cols,
            pca,
            external_feature_root=external_feature_root,
        )
        expected_compounds = int(spec["expected_compounds"])
        observed_compounds = int(len(external_features["full_protein"]))
        if observed_compounds != expected_compounds:
            raise AssertionError(f"{panel_slug} expected {expected_compounds} compounds, observed {observed_compounds}")

        panel_rows.append(
            {
                "dataset": dataset,
                "scope": spec["scope"],
                "panel_slug": panel_slug,
                "n_compounds": observed_compounds,
                "source_rows": int(manifest_item["source_rows"]),
                "expected_rows": int(manifest_item["expected_rows"]),
                "n_input_proteins": int(coverage["n_input_proteins"]),
                "n_shared_features": int(coverage["n_shared_features"]),
                "hct116_feature_overlap_pct": float(coverage["hct116_feature_overlap_pct"]),
            }
        )

        for feature_space in base.FEATURE_SPACE_ORDER:
            cols = feature_cols[feature_space]
            panel_df = external_features[feature_space]
            predictor = predictors[feature_space]
            leaderboard = predictor.leaderboard(silent=True)
            best_model = str(getattr(predictor, "model_best", "") or leaderboard.iloc[0]["model"])
            x = panel_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            pred = predictor.predict(x, model=best_model).reset_index(drop=True)
            proba = predictor.predict_proba(x, model=best_model).reset_index(drop=True)
            out = panel_df[["Compound", "compound_norm"]].copy().reset_index(drop=True)
            out.insert(0, "scope", str(spec["scope"]))
            out.insert(0, "dataset", dataset)
            out.insert(0, "panel_slug", panel_slug)
            out["feature_space"] = feature_space
            out["feature_space_label"] = base.FEATURE_SPACE_LABELS[feature_space]
            out["feature_count"] = int(len(cols))
            out["autogluon_model"] = best_model
            out["predicted_class"] = pred.to_numpy()
            out["prob_antineoplastic"] = proba[POSITIVE_CLASS].to_numpy(dtype=float)
            out["prob_non_antineoplastic"] = proba[NEGATIVE_CLASS].to_numpy(dtype=float)
            prediction_frames.append(out)

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(panel_rows)


def build_wide_predictions(long_df: pd.DataFrame, feature_space_order: list[str]) -> pd.DataFrame:
    index_cols = ["dataset", "scope", "panel_slug", "Compound", "compound_norm"]
    class_wide = long_df.pivot(index=index_cols, columns="feature_space", values="predicted_class").reset_index()
    prob_wide = long_df.pivot(index=index_cols, columns="feature_space", values="prob_antineoplastic").reset_index()
    class_wide = class_wide.rename(columns={fs: f"{fs}_predicted_class" for fs in feature_space_order})
    prob_wide = prob_wide.rename(columns={fs: f"{fs}_prob_antineoplastic" for fs in feature_space_order})
    wide = class_wide.merge(prob_wide, on=index_cols, how="inner")

    class_cols = [f"{fs}_predicted_class" for fs in feature_space_order]
    prob_cols = [f"{fs}_prob_antineoplastic" for fs in feature_space_order]
    wide["positive_feature_space_votes"] = (wide[class_cols] == POSITIVE_CLASS).sum(axis=1)
    wide["mean_prob_antineoplastic"] = wide[prob_cols].mean(axis=1)
    wide["consensus_rule"] = "Antineoplastic if at least 3 of 4 feature spaces call Antineoplastic"
    wide["DeepHeal_feature_space_consensus_call"] = np.where(
        wide["positive_feature_space_votes"] >= 3,
        POSITIVE_CLASS,
        NEGATIVE_CLASS,
    )
    ordered = (
        index_cols
        + class_cols
        + ["positive_feature_space_votes"]
        + prob_cols
        + ["mean_prob_antineoplastic", "consensus_rule", "DeepHeal_feature_space_consensus_call"]
    )
    return wide[ordered].sort_values(["dataset", "mean_prob_antineoplastic"], ascending=[True, False]).reset_index(drop=True)


def summarize_predictions(wide_df: pd.DataFrame, feature_space_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scope), sub in wide_df.groupby(["dataset", "scope"], sort=False):
        row: dict[str, object] = {
            "dataset": dataset,
            "scope": scope,
            "n_compounds": int(len(sub)),
            "consensus_antineoplastic": int((sub["DeepHeal_feature_space_consensus_call"] == POSITIVE_CLASS).sum()),
            "consensus_non_antineoplastic": int((sub["DeepHeal_feature_space_consensus_call"] == NEGATIVE_CLASS).sum()),
            "mean_consensus_prob_antineoplastic": float(sub["mean_prob_antineoplastic"].mean()),
        }
        for feature_space in feature_space_order:
            col = f"{feature_space}_predicted_class"
            prob_col = f"{feature_space}_prob_antineoplastic"
            row[f"{feature_space}_antineoplastic_calls"] = int((sub[col] == POSITIVE_CLASS).sum())
            row[f"{feature_space}_mean_prob_antineoplastic"] = float(sub[prob_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_separate_feature_space_performance(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scope, feature_space, feature_label), sub in long_df.groupby(
        ["dataset", "scope", "feature_space", "feature_space_label"],
        sort=False,
    ):
        n_compounds = int(len(sub))
        n_positive = int((sub["predicted_class"] == POSITIVE_CLASS).sum())
        rows.append(
            {
                "dataset": dataset,
                "scope": scope,
                "feature_space": feature_space,
                "feature_space_label": feature_label,
                "n_compounds": n_compounds,
                "n_predicted_antineoplastic": n_positive,
                "accuracy_assuming_all_antineoplastic": float(n_positive / n_compounds),
                "mean_prob_antineoplastic": float(sub["prob_antineoplastic"].mean()),
                "median_prob_antineoplastic": float(sub["prob_antineoplastic"].median()),
                "autogluon_model": str(sub["autogluon_model"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def verify_outputs(
    autogluon_version: str,
    base: ModuleType,
    train_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> dict[str, object]:
    if autogluon_version != "1.5.0":
        raise AssertionError(f"Expected AutoGluon 1.5.0, observed {autogluon_version}")

    feature_counts = dict(zip(train_summary["feature_space"], train_summary["feature_count"]))
    if feature_counts != base.FEATURE_SPACE_EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected training feature counts: {feature_counts}")

    expected_total_compounds = int(panel_summary["n_compounds"].sum())
    expected_long_rows = expected_total_compounds * len(base.FEATURE_SPACE_ORDER)
    if len(wide_df) != expected_total_compounds:
        raise AssertionError(f"Expected {expected_total_compounds} wide prediction rows, observed {len(wide_df)}")
    if len(long_df) != expected_long_rows:
        raise AssertionError(f"Expected {expected_long_rows} long prediction rows, observed {len(long_df)}")

    if long_df["predicted_class"].isna().any():
        raise AssertionError("Missing predicted_class values in long predictions")
    if long_df["prob_antineoplastic"].isna().any():
        raise AssertionError("Missing prob_antineoplastic values in long predictions")
    if not long_df["prob_antineoplastic"].between(0.0, 1.0).all():
        raise AssertionError("Probability values outside [0, 1]")

    panel_counts = panel_summary.set_index("dataset")["n_compounds"].to_dict()
    expected_panel_counts = {str(spec["dataset"]): int(spec["expected_compounds"]) for spec in PANEL_SPECS}
    if panel_counts != expected_panel_counts:
        raise AssertionError(f"Unexpected panel counts: {panel_counts}")

    return {
        "autogluon_version": autogluon_version,
        "feature_counts": feature_counts,
        "panel_counts": panel_counts,
        "long_prediction_rows": int(len(long_df)),
        "wide_prediction_rows": int(len(wide_df)),
        "no_missing_predictions": True,
        "probabilities_in_unit_interval": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AutoGluon 1.5.0 DeepHeal feature-space predictors and classify ProTargetMiner panels."
    )
    parser.add_argument("--root", type=Path, default=ARTICLE_ROOT)
    parser.add_argument("--output-subdir", default=OUTPUT_DIR_NAME)
    parser.add_argument("--time-limit", type=int, default=1200)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--eval-metric", default="accuracy", choices=["accuracy", "roc_auc"])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--predictor-source-subdir",
        default="",
        help=(
            "Load saved predictors and training/model-coverage summaries from this results subdirectory "
            "instead of fitting predictors under --output-subdir."
        ),
    )
    parser.add_argument(
        "--external-feature-root",
        type=Path,
        default=None,
        help=(
            "Directory containing external-panel feature folders. Defaults to "
            "results/external_panel_predictions under --root."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictors").mkdir(parents=True, exist_ok=True)

    base = load_base_module(root)
    expanded = load_expanded_module(root)
    nvml_patch = expanded.patch_windows_nvml()
    metadata = expanded.registered_model_metadata()
    manifest = read_conversion_manifest(root)
    manifest_by_dataset = supplement_manifest_by_dataset(manifest)

    meta = base.load_labels(root)
    features, feature_cols, provenance = base.build_feature_spaces(root, meta)
    cohort = base.make_shared_labeled_cohort(meta, features).reset_index(drop=True)

    if args.predictor_source_subdir:
        predictors, train_summary, model_coverage = load_saved_feature_space_predictors(
            root=root,
            base=base,
            source_output_subdir=args.predictor_source_subdir,
        )
    else:
        predictors, train_summary, model_coverage = train_feature_space_predictors(
            base=base,
            expanded=expanded,
            features=features,
            feature_cols=feature_cols,
            cohort=cohort,
            out_dir=out_dir,
            metadata=metadata,
            time_limit=args.time_limit,
            verbosity=args.verbosity,
            eval_metric=args.eval_metric,
            reuse_existing=args.reuse_existing,
        )

    long_df, panel_summary = predict_external_panels(
        base=base,
        predictors=predictors,
        feature_cols=feature_cols,
        root=root,
        panel_specs=PANEL_SPECS,
        manifest_by_dataset=manifest_by_dataset,
        external_feature_root=args.external_feature_root.resolve() if args.external_feature_root else None,
    )
    wide_df = build_wide_predictions(long_df, base.FEATURE_SPACE_ORDER)
    dataset_summary = summarize_predictions(wide_df, base.FEATURE_SPACE_ORDER)
    performance = build_separate_feature_space_performance(long_df)

    autogluon_version = importlib.metadata.version("autogluon.tabular")
    verification = verify_outputs(
        autogluon_version=autogluon_version,
        base=base,
        train_summary=train_summary,
        panel_summary=panel_summary,
        long_df=long_df,
        wide_df=wide_df,
    )

    train_summary.to_csv(out_dir / "external_autogluon_training_summary.csv", index=False)
    model_coverage.to_csv(out_dir / "external_autogluon_model_coverage.csv", index=False)
    panel_summary.to_csv(out_dir / "external_panel_feature_coverage_summary.csv", index=False)
    long_df.to_csv(out_dir / "external_panel_feature_space_predictions_long.csv", index=False)
    wide_df.to_csv(out_dir / "external_panel_feature_space_predictions_wide.csv", index=False)
    dataset_summary.to_csv(out_dir / "external_panel_prediction_summary_by_dataset.csv", index=False)
    performance.to_csv(out_dir / "external_panel_separate_feature_space_performance.csv", index=False)

    workbook_path = out_dir / "external_panel_antineoplastic_predictions.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        dataset_summary.to_excel(writer, sheet_name="dataset_summary", index=False)
        wide_df.to_excel(writer, sheet_name="wide_predictions", index=False)
        long_df.to_excel(writer, sheet_name="long_predictions", index=False)
        train_summary.to_excel(writer, sheet_name="training_summary", index=False)
        model_coverage.to_excel(writer, sheet_name="model_coverage", index=False)
        panel_summary.to_excel(writer, sheet_name="feature_coverage", index=False)

    summary = {
        "analysis": "AutoGluon 1.5.0 external ProTargetMiner antineoplastic prediction across four DeepHeal feature spaces",
        "root": str(root),
        "output_dir": str(out_dir),
        "autogluon_version": autogluon_version,
        "eval_metric": args.eval_metric,
        "time_limit_seconds": int(args.time_limit),
        "n_labeled_training_drugs": int(len(cohort)),
        "class_distribution_labeled_training": cohort["Drug_Class"].value_counts().to_dict(),
        "feature_space_order": base.FEATURE_SPACE_ORDER,
        "feature_space_expected_counts": base.FEATURE_SPACE_EXPECTED_COUNTS,
        "feature_provenance": provenance,
        "panel_specs": PANEL_SPECS,
        "nvml_patch": nvml_patch,
        "transductive_notice": (
            "The cached DeepHeal SupCon latent representation was learned on the full 875-drug cohort, "
            "and all 448 labeled drugs contributed labels during SupCon representation learning. "
            "External SupCon predictions should be interpreted with this representation-learning caveat."
        ),
        "verification": verification,
        "outputs": {
            "training_summary_csv": str(out_dir / "external_autogluon_training_summary.csv"),
            "model_coverage_csv": str(out_dir / "external_autogluon_model_coverage.csv"),
            "panel_coverage_csv": str(out_dir / "external_panel_feature_coverage_summary.csv"),
            "predictions_long_csv": str(out_dir / "external_panel_feature_space_predictions_long.csv"),
            "predictions_wide_csv": str(out_dir / "external_panel_feature_space_predictions_wide.csv"),
            "dataset_summary_csv": str(out_dir / "external_panel_prediction_summary_by_dataset.csv"),
            "workbook": str(workbook_path),
        },
    }
    if args.predictor_source_subdir:
        summary["predictor_source_subdir"] = args.predictor_source_subdir
        summary["predictor_source_dir"] = str(root / "results" / args.predictor_source_subdir)
    if args.external_feature_root:
        summary["external_feature_root"] = str(args.external_feature_root.resolve())
    with open(out_dir / "external_panel_autogluon_four_spaces_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({"output_dir": str(out_dir), "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
