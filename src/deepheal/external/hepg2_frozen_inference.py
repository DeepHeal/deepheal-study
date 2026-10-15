"""Frozen 18-profile HepG2 projection, archive validation and single-model inference.

Run directly with the AutoGluon Python environment. No fit/training calls occur.
All outputs are versioned; every resumable source-model cache is content-bound.
"""
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import hashlib
import importlib.metadata
import importlib.util
import io
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import torch

ROOT = ARTICLE_ROOT
PACKAGE = ROOT / 'results/hepg2'
OUT = PACKAGE / 'vae_transfer'
TABLES = OUT / "tables"
QA = OUT / "qa"
CACHE = OUT / "cache"
MODEL_NAME = "RealTabPFN-v2.5"
POSITIVE = "Antineoplastic"
THRESHOLD = 0.5
EXPECTED_COMPOUNDS = ["TCS ERK 11e", "kb NB 142-70", "L-779450", "NVP-BEP800", "AZ 628",
    "BAY 61-3606", "SB 225002", "BIX 01294", "G-1", "WH-4-023", "PF-431396", "KPT-330",
    "Momelotinib", "Fedratinib", "PHA 767491", "AS 1517499", "OTSSP167", "MI-773"]
BANKS = {
    "average": ROOT / 'results/drug_repurposing/models/average_replayed',
    "supcon": ROOT / "results/source_model_checkpoints/supcon",
}
FIDELITY_LIMITS = {"average": 0.9333319, "supcon": 0.930612}
DIMENSIONS = {"average": 194, "supcon": 257}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def status(state, **kwargs):
    record = {"state": state, "updated": datetime.now().astimezone().isoformat(), **kwargs}
    json_write(OUT / "status.json", record)
    with (OUT / "logs/progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def load_helpers():
    path = ROOT / 'src/deepheal/vae_common.py'
    spec = importlib.util.spec_from_file_location("hepg2_frozen_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_inputs(helper):
    features, drugs, _, _ = helper.projection.load_source_feature_data(ROOT)
    drugs = drugs.sort_index()
    assert len(features) == len(set(features)) == 9960 and len(drugs) == 875
    raw_paths = [ROOT / 'data/raw/hepg2_compounds_log2fc.csv', ROOT / "data/raw/MI-773_log2FC.csv"]
    frames, provenance, coverage = [], [], []
    for path in raw_paths:
        frame = pd.read_csv(path)
        assert frame.columns[0] == "protein_ids"
        ids = frame.pop("protein_ids").map(helper.projection.norm)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric.index = ids
        duplicate_count = int(ids.duplicated().sum())
        numeric = numeric.groupby(level=0, sort=False).mean()
        aligned = numeric.reindex(features)
        for compound in aligned:
            observed = np.array([protein in numeric.index for protein in features])
            values = aligned[compound].to_numpy(dtype=float)
            finite = np.isfinite(values)
            coverage.append({"Compound": compound, "source_file": str(path), "n_source_proteins": len(numeric),
                "n_source_duplicate_ids": duplicate_count, "n_exact_mapped_features": int(observed.sum()),
                "n_finite_mapped_features": int((observed & finite).sum()),
                "n_missing_features": int((~observed).sum()),
                "n_nonfinite_observed_features": int((observed & ~finite).sum()),
                "n_zero_filled_features": int((~finite).sum()), "mapped_fraction": float(observed.mean())})
        frames.append(aligned)
        provenance.append({"role": "HepG2 input", "path": str(path), "sha256": sha256(path)})
    raw = pd.concat(frames, axis=1)
    assert raw.columns.tolist() == EXPECTED_COMPOUNDS and raw.columns.is_unique
    values = raw.T.to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = 0
    values[np.abs(values) <= 0.15] = 0
    assert values.shape == (18, 9960) and np.isfinite(values).all()
    coverage = pd.DataFrame(coverage)
    coverage["n_nonzero_after_threshold"] = np.count_nonzero(values, axis=1)
    coverage.to_csv(TABLES / "hepg2_input_coverage.csv", index=False, encoding="utf-8")
    pd.DataFrame({"feature_index": range(9960), "protein_id": features}).to_csv(TABLES / "feature_order.csv", index=False)
    pd.DataFrame(values, index=pd.Index(EXPECTED_COMPOUNDS, name="Compound"), columns=features).to_csv(
        TABLES / "hepg2_aligned_threshold015.csv.gz", encoding="utf-8")
    return features, drugs, values, provenance


def encode_branch(branch, features, drugs, query, helper, provenance):
    status("running", phase="model_manifest", branch=branch)
    summary_path = ROOT / f"results/{branch}/{branch}_model_bank_summary.csv"
    node_path = ROOT / f"results/{branch}/{branch}_ensemble_node_labels.csv"
    pca_path = ROOT / f"results/{branch}/{branch}_pca.pkl"
    archive_path = ROOT / f"results/{branch}/{branch}_consensus_embedding_including_marginal.csv"
    source_summary = pd.read_csv(summary_path)
    nodes = pd.read_csv(node_path)
    mapping = {(int(d), int(r)): g.sort_values("source_node_idx").ensemble_label.to_numpy(dtype=int)
        for (d, r), g in nodes.groupby(["source_latent_dim", "source_run"])}
    expected_stems = {helper.bank_stem(row): row for row in source_summary.to_dict("records")}
    paths = sorted(BANKS[branch].glob("*.pt"))
    assert len(paths) == len(expected_stems) == len(mapping) == 600
    assert {path.stem for path in paths} == set(expected_stems)
    dim = int(nodes.ensemble_label.max()) + 1
    assert dim == DIMENSIONS[branch]
    frozen_manifest_path = ROOT / f'results/drug_repurposing/results/{branch}_checkpoint_manifest.csv'
    frozen_manifest = pd.read_csv(frozen_manifest_path)
    expected_hashes = {Path(row.checkpoint).name: row.sha256 for row in frozen_manifest.itertuples()}
    manifest = []
    for path in paths:
        file_hash = sha256(path)
        assert file_hash == expected_hashes[path.name], f"Frozen checkpoint changed: {path}"
        manifest.append({"branch": branch, "checkpoint": str(path), "sha256": file_hash})
    pd.DataFrame(manifest).to_csv(QA / f"{branch}_checkpoint_manifest.csv", index=False)
    assets = [summary_path, node_path, pca_path, archive_path, frozen_manifest_path]
    for path in assets:
        provenance.append({"role": branch + " frozen asset", "path": str(path), "sha256": sha256(path)})
    pca = helper.projection.load_pickle(pca_path)
    native = drugs.to_numpy(dtype=np.float32)
    if branch == "supcon":
        native = helper.projection.load_supcon_drug_matrix(ROOT, features).loc[drugs.index].to_numpy(dtype=np.float32)
    raw = np.vstack([native, query])
    transformed = pca.transform(raw).astype(np.float32)
    binding = {"branch": branch, "compound_order": EXPECTED_COMPOUNDS, "reference_order": drugs.index.tolist(),
        "feature_order": features, "input_and_asset_manifest": provenance.copy(), "checkpoints": manifest,
        "transformed_shape": list(transformed.shape), "transformed_sha256": hashlib.sha256(transformed.tobytes()).hexdigest(),
        "encoding_recipe": "eval_posterior_mean_unscaled_uncentered_consensus_v1"}
    binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode("utf-8")).hexdigest()
    cache = CACHE / branch / binding_hash
    cache.mkdir(parents=True, exist_ok=True)
    json_write(cache / "binding.json", binding)
    total = np.zeros((len(raw), dim), dtype=np.float64)
    counts = np.zeros(dim, dtype=np.int64)
    technical = []
    for index, path in enumerate(paths):
        stem = path.stem
        row = expected_stems[stem]
        labels = mapping[int(row["source_latent_dim"]), int(row["source_run"])]
        assert len(labels) == int(row["source_latent_dim"]) and (labels >= 0).all() and (labels < dim).all()
        cached_path = cache / (stem + ".npz")
        probe = index in [0, 199, 399, 599]
        if cached_path.exists() and not probe:
            with np.load(cached_path) as stored:
                latent = stored["latent"]
                assert str(stored["binding_hash"]) == binding_hash
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert int(payload["input_dim"]) == transformed.shape[1]
            assert helper.bank_stem(payload["source_summary"]) == stem
            model = helper.build_model(branch, payload)
            latent = helper.encode(model, transformed)
            if probe:
                sample = transformed[[0, 200, 874, 875, 892]]
                separate = helper.encode(model, sample, 1)
                together = helper.encode(model, sample, 5)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                reloaded = helper.build_model(branch, payload)
                reloaded.load_state_dict(torch.load(buffer, map_location=helper.DEVICE, weights_only=True))
                serialized = helper.encode(reloaded, sample, 5)
                technical.append({"branch": branch, "checkpoint": path.name,
                    "batch_max_abs_error": float(np.max(np.abs(separate - together))),
                    "serialization_max_abs_error": float(np.max(np.abs(serialized - together))),
                    "batch_pass": bool(np.allclose(separate, together, atol=2e-4, rtol=2e-4)),
                    "serialization_pass": bool(np.allclose(serialized, together, atol=1e-6, rtol=1e-6))})
                del reloaded
            temporary = cached_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, latent=latent, binding_hash=np.array(binding_hash))
            temporary.replace(cached_path)
            del model, payload
        assert latent.shape == (893, len(labels)) and np.isfinite(latent).all()
        for source_node, consensus_node in enumerate(labels):
            total[:, consensus_node] += latent[:, source_node]
            counts[consensus_node] += 1
        if (index + 1) % 25 == 0:
            status("running", phase="frozen_encoding", branch=branch, completed_models=index + 1, total_models=600)
    assert (counts > 0).all()
    consensus = (total / counts).astype(np.float32)
    archive = pd.read_csv(archive_path)
    archive["drug_norm"] = archive.Drug_Name.map(helper.projection.norm)
    columns = [f"latent_{i:03d}" for i in range(dim)]
    historical = archive.set_index("drug_norm").loc[drugs.index, columns].to_numpy(dtype=np.float32)
    reference = consensus[:875]
    fidelity = helper.rowcos(reference, historical)
    pd.DataFrame({"branch": branch, "drug_norm": drugs.index, "cosine_vs_archive": fidelity,
        "rmse_vs_archive": np.sqrt(np.mean((reference - historical)**2, axis=1))}).to_csv(QA / f"{branch}_archive_fidelity.csv", index=False)
    pd.DataFrame(technical).to_csv(QA / f"{branch}_technical_checks.csv", index=False)
    result = {"branch": branch, "n_checkpoints": 600, "consensus_dimensions": dim,
        "pca_dimensions": transformed.shape[1], "median_cosine_vs_archive": float(np.median(fidelity)),
        "fidelity_threshold": FIDELITY_LIMITS[branch], "archive_pass": bool(np.isfinite(fidelity).all() and np.median(fidelity) >= FIDELITY_LIMITS[branch]),
        "technical_pass": all(row["batch_pass"] and row["serialization_pass"] for row in technical),
        "cache_binding_sha256": binding_hash, "n_reference_profiles": 875, "n_query_profiles": 18,
        "checkpoint_bank": str(BANKS[branch]), "posterior": "mean", "centered": False, "refitted": False}
    json_write(QA / f"{branch}_projection_validation.json", result)
    assert result["archive_pass"] and result["technical_pass"], f"Projection gate failed: {result}"
    output = pd.DataFrame(consensus[875:], columns=columns)
    output.insert(0, "Compound", EXPECTED_COMPOUNDS)
    output.to_csv(TABLES / f"hepg2_{branch}_projected_embeddings.csv", index=False, float_format="%.9g")
    return output, result


def classifier_inference(branch, embeddings, provenance):
    from autogluon.tabular import TabularPredictor
    import autogluon.common.utils.utils as ag_utils

    # The installed environment includes metadata-less distributions. Skip those
    # when AutoGluon records its runtime inventory; this does not change inference.
    def safe_package_versions():
        versions = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                versions[name.lower()] = dist.version
        return versions
    ag_utils.get_package_versions = safe_package_versions
    predictor_dir = ROOT / f'results/autogluon_protargetminer_four_spaces/predictors/{branch}_latent'
    for path in sorted(predictor_dir.rglob("*")):
        if path.is_file() and (path.name == "predictor.pkl" or (MODEL_NAME in path.parts)):
            provenance.append({"role": branch + " classifier", "path": str(path), "sha256": sha256(path)})
    predictor = TabularPredictor.load(str(predictor_dir))
    assert MODEL_NAME in predictor.model_names()
    assert set(predictor.class_labels) == {"Non-Antineoplastic", POSITIVE}
    assert predictor.decision_threshold == THRESHOLD
    features = predictor.features()
    assert features == [f"latent_{i:03d}" for i in range(DIMENSIONS[branch])]
    x = embeddings[features]
    scores = predictor.predict_proba(x, model=MODEL_NAME)[POSITIVE].to_numpy(dtype=float)
    predicted = predictor.predict(x, model=MODEL_NAME, decision_threshold=THRESHOLD).astype(str).to_numpy()
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    assert np.array_equal(predicted == POSITIVE, scores > THRESHOLD)
    out = pd.DataFrame({"Compound": EXPECTED_COMPOUNDS, "branch": branch,
        "feature_space": branch + "_latent", "feature_count": DIMENSIONS[branch],
        "analysis_role": "primary" if branch == "average" else "sensitivity", "autogluon_model": MODEL_NAME,
        "prob_antineoplastic": scores, "predicted_class": predicted, "decision_threshold": THRESHOLD,
        "is_recovered_positive": predicted == POSITIVE, "truth_policy": "assumed_all_positive",
        "metric_context": "positive_panel_recovery", "score_calibration": "uncalibrated"})
    summary = {"branch": branch, "feature_space": branch + "_latent", "feature_count": DIMENSIONS[branch],
        "analysis_role": out.analysis_role.iloc[0], "autogluon_model": MODEL_NAME, "decision_threshold": THRESHOLD,
        "n_profiles": 18, "n_predicted_antineoplastic": int((predicted == POSITIVE).sum()),
        "positive_panel_recovery": float((predicted == POSITIVE).mean()),
        "mean_prob_antineoplastic": float(np.mean(scores)), "median_prob_antineoplastic": float(np.median(scores))}
    json_write(QA / f"{branch}_classifier_validation.json", {"model": MODEL_NAME, "features": features,
        "class_labels": predictor.class_labels, "threshold": THRESHOLD, "schema_pass": True,
        "threshold_consistency_pass": True, "autogluon_version": importlib.metadata.version("autogluon.tabular")})
    return out, summary


def main():
    for folder in [TABLES, QA, CACHE, OUT / "logs"]:
        folder.mkdir(parents=True, exist_ok=True)
    status("running", phase="input_preparation")
    helper = load_helpers()
    features, drugs, query, provenance = prepare_inputs(helper)
    for path in [ROOT / "data/raw/875_Compounds_Average_Data.csv",
        ROOT / "data/raw/1752_Experiments_Data_With_Replicate_for_VAE.csv",
        ROOT / "data/raw/1628_Experiments_Data.csv",
        ROOT / "data/raw/drug_hct116_classification_including_marginal_with_cluster.csv",
        Path(__file__), ROOT / 'src/deepheal/vae_common.py',
        ROOT / 'src/deepheal/projection.py',
        ROOT / 'src/deepheal/models/average_ensemble.py',
        ROOT / 'src/deepheal/models/supcon_vae.py']:
        provenance.append({"role": "reference or inference code", "path": str(path), "sha256": sha256(path)})
    validation, embeddings = {}, {}
    # Both branches must pass the archive gate before any classifier result is released.
    for branch in ["average", "supcon"]:
        embeddings[branch], validation[branch] = encode_branch(branch, features, drugs, query, helper, provenance)
    predictions, summaries = [], []
    for branch in ["average", "supcon"]:
        status("running", phase="saved_classifier_inference", branch=branch)
        pred, summary = classifier_inference(branch, embeddings[branch], provenance)
        predictions.append(pred)
        summaries.append(summary)
    pd.concat(predictions, ignore_index=True).to_csv(TABLES / "hepg2_vae_predictions_long.csv", index=False)
    pd.DataFrame(summaries).to_csv(TABLES / "hepg2_vae_summary.csv", index=False)
    pd.DataFrame(provenance).drop_duplicates("path").to_csv(QA / "source_manifest.csv", index=False)
    result = {"n_profiles": 18, "n_prediction_rows": 36, "branches": validation,
        "classifier_summaries": summaries, "device": str(helper.DEVICE), "torch_version": torch.__version__,
        "experiment_conditions_user_confirmed": {"concentration_uM": 10, "duration_h": 24, "biological_replicates": 2},
        "refitted": False, "source_manifest": str(QA / "source_manifest.csv")}
    json_write(OUT / "run_summary.json", result)
    status("complete", phase="inference_and_validation_complete", summaries=summaries)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "logs").mkdir(exist_ok=True)
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise
