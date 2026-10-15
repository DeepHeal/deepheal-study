#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


PANEL_NAME = "Jurkat decryptE 144-drug proteome 10000 nM"
PANEL_SLUG = "jurkat_decrypte_144_10000nm"
OUTPUT_DIR_NAME = 'autogluon_jurkat_decrypte_144_four_spaces'
SOURCE_PREDICTOR_SUBDIR = 'autogluon_protargetminer_four_spaces'
POSITIVE_CLASS = "Antineoplastic"
NEGATIVE_CLASS = "Non-Antineoplastic"


def repo_root_from_script() -> Path:
    return ARTICLE_ROOT


def package_dir_from_script() -> Path:
    return ARTICLE_ROOT


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_pipeline_path(root: Path) -> None:
    pipeline_dir = root / 'src/deepheal/benchmarking'
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))


def load_base_module(root: Path):
    add_pipeline_path(root)
    return load_module("deepheal_feature_space_base_jurkat", root / 'src/deepheal/benchmarking/feature_space_comparison.py')


def load_feature_columns(root: Path, base) -> dict[str, list[str]]:
    meta = base.load_labels(root)
    _, feature_cols, _ = base.build_feature_spaces(root, meta)
    return feature_cols


def load_saved_predictors(root: Path, base) -> dict[str, TabularPredictor]:
    predictor_root = root / "results" / SOURCE_PREDICTOR_SUBDIR / "predictors"
    predictors: dict[str, TabularPredictor] = {}
    for feature_space in base.FEATURE_SPACE_ORDER:
        predictor_dir = predictor_root / feature_space
        if not predictor_dir.exists():
            raise FileNotFoundError(f"Missing saved predictor directory: {predictor_dir}")
        predictors[feature_space] = TabularPredictor.load(str(predictor_dir), require_version_match=False)
    return predictors


def load_prepared_features(package_dir: Path, feature_cols: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    aligned_path = table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv"
    pca_path = table_dir / f"{PANEL_SLUG}_pca_200_features.csv"
    supcon_path = table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv"
    average_path = table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv"
    for path in [aligned_path, pca_path, supcon_path, average_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required prepared feature file is missing: {path}")

    aligned = pd.read_csv(aligned_path)
    if not {"Compound", "compound_norm"}.issubset(aligned.columns):
        raise ValueError(f"Aligned matrix is missing Compound/compound_norm columns: {aligned_path}")
    raw_cols = [col for col in aligned.columns if col not in {"Compound", "compound_norm"}]
    if len(raw_cols) != len(feature_cols["full_protein"]):
        raise ValueError(
            f"Aligned full-protein feature count {len(raw_cols)} does not match "
            f"expected {len(feature_cols['full_protein'])}"
        )
    full_df = pd.DataFrame(
        aligned[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        columns=feature_cols["full_protein"],
    )
    full_df.insert(0, "compound_norm", aligned["compound_norm"].to_numpy())
    full_df.insert(0, "Compound", aligned["Compound"].to_numpy())

    pca = pd.read_csv(pca_path)
    supcon = pd.read_csv(supcon_path)
    average = pd.read_csv(average_path)
    for feature_space, df in [
        ("pca_200", pca),
        ("supcon_latent", supcon),
        ("average_latent", average),
    ]:
        required = ["Compound", "compound_norm", *feature_cols[feature_space]]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{feature_space} is missing {len(missing)} required columns: {missing[:5]}")

    return {
        "pca_200": pca[["Compound", "compound_norm", *feature_cols["pca_200"]]].copy(),
        "supcon_latent": supcon[["Compound", "compound_norm", *feature_cols["supcon_latent"]]].copy(),
        "average_latent": average[["Compound", "compound_norm", *feature_cols["average_latent"]]].copy(),
        "full_protein": full_df,
    }


def predict_feature_spaces(base, predictors, feature_cols, features) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature_space in base.FEATURE_SPACE_ORDER:
        cols = feature_cols[feature_space]
        predictor = predictors[feature_space]
        best_model = str(getattr(predictor, "model_best", "") or predictor.leaderboard(silent=True).iloc[0]["model"])
        x = features[feature_space][cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pred = predictor.predict(x, model=best_model).reset_index(drop=True)
        proba = predictor.predict_proba(x, model=best_model).reset_index(drop=True)
        if POSITIVE_CLASS not in proba.columns or NEGATIVE_CLASS not in proba.columns:
            raise ValueError(f"Predictor for {feature_space} did not return expected class probability columns")
        out = features[feature_space][["Compound", "compound_norm"]].copy().reset_index(drop=True)
        out.insert(1, "compound", out["Compound"])
        out.insert(0, "panel_slug", PANEL_SLUG)
        out.insert(0, "panel_name", PANEL_NAME)
        out["feature_space"] = feature_space
        out["feature_space_label"] = base.FEATURE_SPACE_LABELS[feature_space]
        out["feature_count"] = int(len(cols))
        out["autogluon_model"] = best_model
        out["model"] = best_model
        out["decision_threshold"] = float(getattr(predictor, "decision_threshold", 0.5))
        out["predicted_class"] = pred.to_numpy()
        out["prob_antineoplastic"] = proba[POSITIVE_CLASS].to_numpy(dtype=float)
        out["prob_non_antineoplastic"] = proba[NEGATIVE_CLASS].to_numpy(dtype=float)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def build_wide_predictions(long_df: pd.DataFrame, feature_space_order: list[str], label_match: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["panel_name", "panel_slug", "Compound", "compound", "compound_norm"]
    class_wide = long_df.pivot(index=index_cols, columns="feature_space", values="predicted_class").reset_index()
    prob_wide = long_df.pivot(index=index_cols, columns="feature_space", values="prob_antineoplastic").reset_index()
    class_wide = class_wide.rename(columns={fs: f"{fs}_predicted_class" for fs in feature_space_order})
    prob_wide = prob_wide.rename(columns={fs: f"{fs}_prob_antineoplastic" for fs in feature_space_order})
    wide = class_wide.merge(prob_wide, on=index_cols, how="inner")

    class_cols = [f"{fs}_predicted_class" for fs in feature_space_order]
    prob_cols = [f"{fs}_prob_antineoplastic" for fs in feature_space_order]
    wide["positive_feature_space_votes"] = (wide[class_cols] == POSITIVE_CLASS).sum(axis=1)
    wide["positive_vote_count"] = wide["positive_feature_space_votes"]
    wide["mean_prob_antineoplastic"] = wide[prob_cols].mean(axis=1)
    wide["consensus_rule"] = "Antineoplastic if at least 3 of 4 feature spaces call Antineoplastic"
    wide["DeepHeal_feature_space_consensus_call"] = np.where(
        wide["positive_feature_space_votes"] >= 3,
        POSITIVE_CLASS,
        NEGATIVE_CLASS,
    )
    wide["consensus_call"] = wide["DeepHeal_feature_space_consensus_call"]
    if len(label_match) > 0:
        wide = wide.merge(label_match, on=["Compound", "compound_norm"], how="left")

    ordered = (
        index_cols
        + class_cols
        + ["positive_feature_space_votes", "positive_vote_count"]
        + prob_cols
        + ["mean_prob_antineoplastic", "consensus_rule", "DeepHeal_feature_space_consensus_call", "consensus_call"]
        + [
            col
            for col in [
                "deepheal_label_match",
                "matched_deepheal_drug_name",
                "matched_deepheal_drug_class",
                "matched_deepheal_sample_id",
                "matched_deepheal_cluster",
            ]
            if col in wide.columns
        ]
    )
    return wide[ordered].sort_values(["mean_prob_antineoplastic", "Compound"], ascending=[False, True]).reset_index(drop=True)


def summarize_predictions(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (feature_space, feature_label), sub in long_df.groupby(["feature_space", "feature_space_label"], sort=False):
        n_compounds = int(len(sub))
        n_positive = int((sub["predicted_class"] == POSITIVE_CLASS).sum())
        rows.append(
            {
                "panel_name": PANEL_NAME,
                "panel_slug": PANEL_SLUG,
                "feature_space": feature_space,
                "feature_space_label": feature_label,
                "feature_count": int(sub["feature_count"].iloc[0]),
                "n_compounds": n_compounds,
                "n_predicted_antineoplastic": n_positive,
                "n_predicted_non_antineoplastic": int(n_compounds - n_positive),
                "fraction_predicted_antineoplastic": float(n_positive / n_compounds),
                "mean_prob_antineoplastic": float(sub["prob_antineoplastic"].mean()),
                "median_prob_antineoplastic": float(sub["prob_antineoplastic"].median()),
                "autogluon_model": str(sub["autogluon_model"].iloc[0]),
                "model": str(sub["model"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_consensus(wide_df: pd.DataFrame) -> pd.DataFrame:
    vote_counts = wide_df["positive_feature_space_votes"].value_counts().sort_index()
    rows = [
        {
            "positive_feature_space_votes": int(votes),
            "positive_vote_count": int(votes),
            "n_compounds": int(count),
            "fraction_compounds": float(count / len(wide_df)),
        }
        for votes, count in vote_counts.items()
    ]
    return pd.DataFrame(rows)


def verify_outputs(autogluon_version: str, base, long_df: pd.DataFrame, wide_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, object]:
    if autogluon_version != "1.5.0":
        raise AssertionError(f"Expected AutoGluon 1.5.0, observed {autogluon_version}")
    feature_counts = dict(zip(summary_df["feature_space"], summary_df["feature_count"]))
    if feature_counts != base.FEATURE_SPACE_EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected feature counts: {feature_counts}")
    if len(wide_df) != 144:
        raise AssertionError(f"Expected 144 wide prediction rows, observed {len(wide_df)}")
    expected_long = 144 * len(base.FEATURE_SPACE_ORDER)
    if len(long_df) != expected_long:
        raise AssertionError(f"Expected {expected_long} long prediction rows, observed {len(long_df)}")
    if long_df["predicted_class"].isna().any():
        raise AssertionError("Missing predicted_class values")
    if long_df["prob_antineoplastic"].isna().any() or long_df["prob_non_antineoplastic"].isna().any():
        raise AssertionError("Missing probability values")
    if not long_df["prob_antineoplastic"].between(0.0, 1.0).all():
        raise AssertionError("prob_antineoplastic values outside [0, 1]")
    if not long_df["prob_non_antineoplastic"].between(0.0, 1.0).all():
        raise AssertionError("prob_non_antineoplastic values outside [0, 1]")
    if set(long_df["predicted_class"].unique()) - {POSITIVE_CLASS, NEGATIVE_CLASS}:
        raise AssertionError(f"Unexpected predicted classes: {sorted(long_df['predicted_class'].unique())}")
    return {
        "autogluon_version": autogluon_version,
        "feature_counts": {key: int(value) for key, value in feature_counts.items()},
        "long_prediction_rows": int(len(long_df)),
        "wide_prediction_rows": int(len(wide_df)),
        "no_missing_predictions": True,
        "probabilities_in_unit_interval": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict decryptE Jurkat compounds with saved four-space AutoGluon predictors.")
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--package-dir", type=Path, default=package_dir_from_script())
    parser.add_argument("--output-subdir", default=OUTPUT_DIR_NAME)
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = args.package_dir.resolve()
    out_dir = package_dir / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(root)
    feature_cols = load_feature_columns(root, base)
    predictors = load_saved_predictors(root, base)
    features = load_prepared_features(package_dir, feature_cols)
    long_df = predict_feature_spaces(base, predictors, feature_cols, features)

    label_match_path = package_dir / "data" / "processed" / "jurkat_decrypte_deepheal_label_match.csv"
    label_match = pd.read_csv(label_match_path) if label_match_path.exists() else pd.DataFrame()
    wide_df = build_wide_predictions(long_df, base.FEATURE_SPACE_ORDER, label_match)
    feature_summary = summarize_predictions(long_df)
    consensus_summary = summarize_consensus(wide_df)

    autogluon_version = importlib.metadata.version("autogluon.tabular")
    verification = verify_outputs(autogluon_version, base, long_df, wide_df, feature_summary)

    long_path = out_dir / "jurkat_decrypte_144_feature_space_predictions_long.csv"
    wide_path = out_dir / "jurkat_decrypte_144_feature_space_predictions_wide.csv"
    feature_summary_path = out_dir / "jurkat_decrypte_144_feature_space_summary.csv"
    consensus_summary_path = out_dir / "jurkat_decrypte_144_consensus_vote_summary.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8")
    feature_summary.to_csv(feature_summary_path, index=False, encoding="utf-8")
    consensus_summary.to_csv(consensus_summary_path, index=False, encoding="utf-8")

    workbook_path = out_dir / "jurkat_decrypte_144_antineoplastic_predictions.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        feature_summary.to_excel(writer, sheet_name="feature_summary", index=False)
        consensus_summary.to_excel(writer, sheet_name="consensus_summary", index=False)
        wide_df.to_excel(writer, sheet_name="wide_predictions", index=False)
        long_df.to_excel(writer, sheet_name="long_predictions", index=False)
        if len(label_match) > 0:
            label_match.to_excel(writer, sheet_name="deepheal_label_match", index=False)

    summary_payload = {
        "analysis": "decryptE Jurkat 144-drug antineoplastic prediction across four DeepHeal feature spaces",
        "panel_name": PANEL_NAME,
        "panel_slug": PANEL_SLUG,
        "root": str(root),
        "package_dir": str(package_dir),
        "output_dir": str(out_dir),
        "predictor_source_dir": str(root / "results" / SOURCE_PREDICTOR_SUBDIR / "predictors"),
        "autogluon_version": autogluon_version,
        "feature_space_order": base.FEATURE_SPACE_ORDER,
        "feature_space_expected_counts": base.FEATURE_SPACE_EXPECTED_COUNTS,
        "n_compounds": int(len(wide_df)),
        "reporting_note": (
            "These are external prediction calls for 144 decryptE Jurkat perturbations. "
            "fraction_predicted_antineoplastic is a model-call fraction, not a measured binary accuracy metric."
        ),
        "verification": verification,
        "outputs": {
            "predictions_long_csv": str(long_path),
            "predictions_wide_csv": str(wide_path),
            "feature_summary_csv": str(feature_summary_path),
            "consensus_summary_csv": str(consensus_summary_path),
            "workbook": str(workbook_path),
        },
    }
    summary_json = out_dir / "jurkat_decrypte_144_autogluon_four_spaces_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
