#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


PANEL_NAME = "Ruprecht Nat Chem Biol 2020 Supplementary Data 1 lung cancer cell lines"
PANEL_SLUG = "ruprecht_2020_suppdata1_lung_cell_lines"
OUTPUT_DIR_NAME = 'autogluon_lung_cell_lines_four_spaces'
SOURCE_PREDICTOR_SUBDIR = 'autogluon_protargetminer_four_spaces'
POSITIVE_CLASS = "Antineoplastic"
NEGATIVE_CLASS = "Non-Antineoplastic"
EXPECTED_MODEL_BEST = "WeightedEnsemble_L2"
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


def add_pipeline_path(root: Path) -> None:
    pipeline_dir = root / 'src/deepheal/benchmarking'
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))


def load_module(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_module(root: Path):
    add_pipeline_path(root)
    return load_module(
        "deepheal_feature_space_base_lung_cell_lines",
        root / 'src/deepheal/benchmarking/feature_space_comparison.py',
    )


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


def meta_columns() -> list[str]:
    return [
        "profile_id",
        "cell_line",
        "inhibitor_key",
        "inhibitor",
        "Compound",
        "compound_norm",
        "source_sheet",
        "source_log2fc_column",
    ]


def load_prepared_features(package_dir: Path, feature_cols: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    table_dir = package_dir / "results" / "external_panel_predictions" / PANEL_SLUG / "tables"
    aligned_path = table_dir / f"{PANEL_SLUG}_aligned_threshold015.csv"
    pca_path = table_dir / f"{PANEL_SLUG}_pca_200_features.csv"
    supcon_path = table_dir / f"{PANEL_SLUG}_supcon_projected_embeddings.csv"
    average_path = table_dir / f"{PANEL_SLUG}_average_projected_embeddings.csv"
    for path in [aligned_path, pca_path, supcon_path, average_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required prepared feature file is missing: {path}")

    dtype_meta = {col: str for col in meta_columns()}
    aligned = pd.read_csv(aligned_path, dtype=dtype_meta)
    required_meta = set(meta_columns())
    missing_meta = sorted(required_meta - set(aligned.columns))
    if missing_meta:
        raise ValueError(f"Aligned matrix is missing metadata columns: {missing_meta}")

    raw_cols = [col for col in aligned.columns if col not in required_meta]
    if len(raw_cols) != len(feature_cols["full_protein"]):
        raise ValueError(
            f"Aligned full-protein feature count {len(raw_cols)} does not match "
            f"expected {len(feature_cols['full_protein'])}"
        )
    full_df = pd.concat(
        [
            aligned[meta_columns()].reset_index(drop=True),
            pd.DataFrame(
                aligned[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                columns=feature_cols["full_protein"],
            ),
        ],
        axis=1,
    )

    pca = pd.read_csv(pca_path, dtype=dtype_meta)
    supcon = pd.read_csv(supcon_path, dtype=dtype_meta)
    average = pd.read_csv(average_path, dtype=dtype_meta)
    for feature_space, df in [
        ("pca_200", pca),
        ("supcon_latent", supcon),
        ("average_latent", average),
    ]:
        required = [*meta_columns(), *feature_cols[feature_space]]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{feature_space} is missing {len(missing)} required columns: {missing[:5]}")

    return {
        "pca_200": pca[[*meta_columns(), *feature_cols["pca_200"]]].copy(),
        "supcon_latent": supcon[[*meta_columns(), *feature_cols["supcon_latent"]]].copy(),
        "average_latent": average[[*meta_columns(), *feature_cols["average_latent"]]].copy(),
        "full_protein": full_df,
    }


def predict_feature_spaces(base, predictors, feature_cols, features) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature_space in base.FEATURE_SPACE_ORDER:
        cols = feature_cols[feature_space]
        predictor = predictors[feature_space]
        best_model = str(getattr(predictor, "model_best", "") or predictor.leaderboard(silent=True).iloc[0]["model"])
        if best_model != EXPECTED_MODEL_BEST:
            raise AssertionError(f"{feature_space} expected {EXPECTED_MODEL_BEST}, observed {best_model}")
        x = features[feature_space][cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pred = predictor.predict(x, model=best_model).reset_index(drop=True)
        proba = predictor.predict_proba(x, model=best_model).reset_index(drop=True)
        if POSITIVE_CLASS not in proba.columns or NEGATIVE_CLASS not in proba.columns:
            raise ValueError(f"Predictor for {feature_space} did not return expected class probability columns")
        out = features[feature_space][meta_columns()].copy().reset_index(drop=True)
        out["panel_name"] = PANEL_NAME
        out["panel_slug"] = PANEL_SLUG
        out["feature_space"] = feature_space
        out["feature_space_label"] = base.FEATURE_SPACE_LABELS[feature_space]
        out["feature_count"] = int(len(cols))
        out["autogluon_model"] = best_model
        out["decision_threshold"] = float(getattr(predictor, "decision_threshold", 0.5))
        out["predicted_class"] = pred.to_numpy()
        out["prob_antineoplastic"] = proba[POSITIVE_CLASS].to_numpy(dtype=float)
        out["prob_non_antineoplastic"] = proba[NEGATIVE_CLASS].to_numpy(dtype=float)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def build_wide_predictions(long_df: pd.DataFrame, feature_space_order: list[str], label_match: pd.DataFrame) -> pd.DataFrame:
    index_cols = [*meta_columns(), "panel_name", "panel_slug"]
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

    if len(label_match) > 0:
        keep = [
            col
            for col in [
                "profile_id",
                "deepheal_label_match",
                "matched_deepheal_drug_name",
                "matched_deepheal_drug_class",
                "matched_deepheal_sample_id",
                "matched_deepheal_cluster",
            ]
            if col in label_match.columns
        ]
        wide = wide.merge(label_match[keep], on="profile_id", how="left")

    label_cols = [
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
    ordered = (
        [*meta_columns(), "panel_name", "panel_slug"]
        + class_cols
        + ["positive_feature_space_votes"]
        + prob_cols
        + ["mean_prob_antineoplastic", "consensus_rule", "DeepHeal_feature_space_consensus_call"]
        + label_cols
    )
    return wide[ordered].sort_values(["cell_line", "inhibitor_key"]).reset_index(drop=True)


def summarize_predictions(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in long_df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        n_profiles = int(len(sub))
        n_positive = int((sub["predicted_class"] == POSITIVE_CLASS).sum())
        row.update(
            {
                "n_profiles": n_profiles,
                "n_predicted_antineoplastic": n_positive,
                "n_predicted_non_antineoplastic": int(n_profiles - n_positive),
                "fraction_predicted_antineoplastic": float(n_positive / n_profiles),
                "mean_prob_antineoplastic": float(sub["prob_antineoplastic"].mean()),
                "median_prob_antineoplastic": float(sub["prob_antineoplastic"].median()),
            }
        )
        if "feature_space" in group_cols:
            row["feature_count"] = int(sub["feature_count"].iloc[0])
            row["autogluon_model"] = str(sub["autogluon_model"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_consensus(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        wide_df["positive_feature_space_votes"]
        .value_counts()
        .sort_index()
        .rename_axis("positive_feature_space_votes")
        .reset_index(name="n_profiles")
    )
    overall["fraction_profiles"] = overall["n_profiles"] / len(wide_df)

    rows: list[dict[str, object]] = []
    for (cell_line, votes), sub in wide_df.groupby(["cell_line", "positive_feature_space_votes"], sort=False):
        rows.append(
            {
                "cell_line": cell_line,
                "positive_feature_space_votes": int(votes),
                "n_profiles": int(len(sub)),
                "fraction_cell_line_profiles": float(len(sub) / (wide_df["cell_line"] == cell_line).sum()),
            }
        )
    return overall, pd.DataFrame(rows).sort_values(["cell_line", "positive_feature_space_votes"]).reset_index(drop=True)


def verify_outputs(
    autogluon_version: str,
    base,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> dict[str, object]:
    if autogluon_version != "1.5.0":
        raise AssertionError(f"Expected AutoGluon 1.5.0, observed {autogluon_version}")
    feature_counts = dict(zip(feature_summary["feature_space"], feature_summary["feature_count"]))
    if feature_counts != base.FEATURE_SPACE_EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected feature counts: {feature_counts}")
    profile_counts = wide_df["cell_line"].value_counts().reindex(EXPECTED_PROFILE_COUNTS).to_dict()
    profile_counts = {str(key): int(value) for key, value in profile_counts.items()}
    if profile_counts != EXPECTED_PROFILE_COUNTS:
        raise AssertionError(f"Unexpected profile counts: {profile_counts}")
    if len(wide_df) != 262:
        raise AssertionError(f"Expected 262 wide prediction rows, observed {len(wide_df)}")
    if len(long_df) != 262 * len(base.FEATURE_SPACE_ORDER):
        raise AssertionError(f"Expected 1048 long prediction rows, observed {len(long_df)}")
    if long_df["predicted_class"].isna().any():
        raise AssertionError("Missing predicted_class values")
    if long_df["prob_antineoplastic"].isna().any() or long_df["prob_non_antineoplastic"].isna().any():
        raise AssertionError("Missing probability values")
    if not long_df["prob_antineoplastic"].between(0.0, 1.0).all():
        raise AssertionError("prob_antineoplastic values outside [0, 1]")
    if not long_df["prob_non_antineoplastic"].between(0.0, 1.0).all():
        raise AssertionError("prob_non_antineoplastic values outside [0, 1]")
    observed_classes = set(long_df["predicted_class"].unique())
    if observed_classes - {POSITIVE_CLASS, NEGATIVE_CLASS}:
        raise AssertionError(f"Unexpected predicted classes: {sorted(observed_classes)}")
    observed_models = set(long_df["autogluon_model"].unique())
    if observed_models != {EXPECTED_MODEL_BEST}:
        raise AssertionError(f"Unexpected AutoGluon models: {sorted(observed_models)}")
    return {
        "autogluon_version": autogluon_version,
        "feature_counts": {key: int(value) for key, value in feature_counts.items()},
        "profile_counts_by_cell_line": profile_counts,
        "long_prediction_rows": int(len(long_df)),
        "wide_prediction_rows": int(len(wide_df)),
        "no_missing_predictions": True,
        "predicted_classes": sorted(observed_classes),
        "probabilities_in_unit_interval": True,
        "autogluon_models": sorted(observed_models),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Ruprecht 2020 Supplementary Data 1 lung cell-line profiles with saved four-space AutoGluon predictors."
    )
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

    label_match_path = package_dir / "data" / "processed" / "suppdata1_lung_cell_line_deepheal_label_match.csv"
    label_match = pd.read_csv(label_match_path) if label_match_path.exists() else pd.DataFrame()
    wide_df = build_wide_predictions(long_df, base.FEATURE_SPACE_ORDER, label_match)
    feature_summary = summarize_predictions(
        long_df,
        ["feature_space", "feature_space_label"],
    )
    cell_line_summary = summarize_predictions(
        long_df,
        ["cell_line", "feature_space", "feature_space_label"],
    )
    compound_summary = summarize_predictions(
        long_df,
        ["Compound", "inhibitor_key", "feature_space", "feature_space_label"],
    )
    consensus_summary, consensus_by_cell_line = summarize_consensus(wide_df)

    autogluon_version = importlib.metadata.version("autogluon.tabular")
    verification = verify_outputs(autogluon_version, base, long_df, wide_df, feature_summary)

    long_path = out_dir / "lung_cell_line_feature_space_predictions_long.csv"
    wide_path = out_dir / "lung_cell_line_feature_space_predictions_wide.csv"
    feature_summary_path = out_dir / "lung_cell_line_feature_space_summary.csv"
    cell_line_summary_path = out_dir / "lung_cell_line_cell_line_feature_space_summary.csv"
    compound_summary_path = out_dir / "lung_cell_line_compound_feature_space_summary.csv"
    consensus_summary_path = out_dir / "lung_cell_line_consensus_vote_summary.csv"
    consensus_by_cell_line_path = out_dir / "lung_cell_line_consensus_vote_summary_by_cell_line.csv"

    long_df.to_csv(long_path, index=False, encoding="utf-8")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8")
    feature_summary.to_csv(feature_summary_path, index=False, encoding="utf-8")
    cell_line_summary.to_csv(cell_line_summary_path, index=False, encoding="utf-8")
    compound_summary.to_csv(compound_summary_path, index=False, encoding="utf-8")
    consensus_summary.to_csv(consensus_summary_path, index=False, encoding="utf-8")
    consensus_by_cell_line.to_csv(consensus_by_cell_line_path, index=False, encoding="utf-8")

    workbook_path = out_dir / "lung_cell_line_antineoplastic_predictions.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        feature_summary.to_excel(writer, sheet_name="feature_summary", index=False)
        cell_line_summary.to_excel(writer, sheet_name="cell_line_summary", index=False)
        compound_summary.to_excel(writer, sheet_name="compound_summary", index=False)
        consensus_summary.to_excel(writer, sheet_name="consensus_summary", index=False)
        consensus_by_cell_line.to_excel(writer, sheet_name="consensus_by_cell_line", index=False)
        wide_df.to_excel(writer, sheet_name="wide_predictions", index=False)
        long_df.to_excel(writer, sheet_name="long_predictions", index=False)
        if len(label_match) > 0:
            label_match.to_excel(writer, sheet_name="deepheal_label_match", index=False)

    summary_payload = {
        "analysis": "Ruprecht 2020 Supplementary Data 1 lung cancer cell-line antineoplastic prediction across four DeepHeal feature spaces",
        "source_article": "https://www.nature.com/articles/s41589-020-0572-3",
        "panel_name": PANEL_NAME,
        "panel_slug": PANEL_SLUG,
        "root": str(root),
        "package_dir": str(package_dir),
        "output_dir": str(out_dir),
        "predictor_source_dir": str(root / "results" / SOURCE_PREDICTOR_SUBDIR / "predictors"),
        "autogluon_version": autogluon_version,
        "feature_space_order": base.FEATURE_SPACE_ORDER,
        "feature_space_expected_counts": base.FEATURE_SPACE_EXPECTED_COUNTS,
        "n_profiles": int(len(wide_df)),
        "reporting_note": (
            "fraction_predicted_antineoplastic is an external prediction summary, not a binary accuracy metric, "
            "because this Supplementary Data 1 panel is not curated as a balanced positive/negative benchmark."
        ),
        "verification": verification,
        "outputs": {
            "predictions_long_csv": str(long_path),
            "predictions_wide_csv": str(wide_path),
            "feature_summary_csv": str(feature_summary_path),
            "cell_line_summary_csv": str(cell_line_summary_path),
            "compound_summary_csv": str(compound_summary_path),
            "consensus_summary_csv": str(consensus_summary_path),
            "consensus_by_cell_line_csv": str(consensus_by_cell_line_path),
            "workbook": str(workbook_path),
        },
    }
    summary_json = out_dir / "lung_cell_line_autogluon_four_spaces_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
