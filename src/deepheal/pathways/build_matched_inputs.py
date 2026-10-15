from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom

from pathway_comparison import COMPARISON_SPECS, DATA_DIR, DEEPHEAL_RESULTS, FISHER_ALTERNATIVE, MATCHED_BASELINE_DIR, MANUSCRIPT_ROOT, METHOD_META, PATHWAY_GMT_DIR, SOURCE_VAE_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = ARTICLE_ROOT / "src/deepheal/benchmarking/comparator_methods.py"
FIG2_SCRIPT = ARTICLE_ROOT / "src/deepheal/pathways/pathway_methods.py"
TOP_N = 31


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(path)


def top_gene_sets(top_df: pd.DataFrame, normalize) -> dict[str, set[str]]:
    df = top_df.copy()
    df["node"] = df["node"].astype(str)
    df["gene_name"] = df["gene_name"].astype(str).map(normalize)
    df = df[df["gene_name"].str.lower() != "nan"].copy()
    return {
        node: set(frame.sort_values("rank")["gene_name"].tolist())
        for node, frame in df.groupby("node", sort=False)
    }


def write_fdr_outputs(
    method: str,
    top_df: pd.DataFrame,
    fig2,
    canonical_pathways: dict[str, set[str]],
    oncogenic_pathways: dict[str, set[str]],
    universe_genes: Iterable[str],
) -> None:
    node_sets = top_gene_sets(top_df, fig2.normalize_id)
    fdr_c2 = fig2.run_fisher_enrichment(
        method_name=method,
        node_sets=node_sets,
        pathways=canonical_pathways,
        universe_genes=universe_genes,
    )
    fdr_c6 = fig2.run_fisher_enrichment(
        method_name=method,
        node_sets=node_sets,
        pathways=oncogenic_pathways,
        universe_genes=universe_genes,
    )
    save_table(top_df, MATCHED_BASELINE_DIR / f"{method}_top31_genes_per_latent.csv")
    save_table(fdr_c2, MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c2_canonical.csv")
    save_table(fdr_c6, MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c6_oncogenic.csv")


def build_linear_baselines(
    dim: int,
    avg: dict[str, object],
    benchmark,
    fig2,
    canonical_pathways: dict[str, set[str]],
    oncogenic_pathways: dict[str, set[str]],
    universe_genes: Iterable[str],
    seed: int,
    force: bool,
) -> None:
    x_scaled = avg["x_scaled"]
    gene_names = avg["gene_names"]
    gene_keys = avg["gene_keys"]

    specs = []
    pca_method = f"pca_{dim}L"
    pca_model = benchmark.PCA(n_components=dim, random_state=seed)
    pca_model.fit(x_scaled)
    specs.append((pca_method, "PCA", pca_model.components_))

    ica_method = f"ica_{dim}L"
    ica_model = benchmark.FastICA(
        n_components=dim,
        random_state=seed,
        whiten="unit-variance",
        max_iter=2000,
        tol=0.005,
    )
    ica_model.fit(x_scaled)
    specs.append((ica_method, "ICA", ica_model.components_))

    rp_method = f"rp_{dim}L"
    rp_model = benchmark.GaussianRandomProjection(n_components=dim, random_state=seed)
    rp_model.fit(x_scaled)
    specs.append((rp_method, "RP", rp_model.components_))

    for method, prefix, score_matrix in specs:
        c2_path = MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c2_canonical.csv"
        c6_path = MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c6_oncogenic.csv"
        if c2_path.exists() and c6_path.exists() and not force:
            print(f"[skip] {method}")
            continue
        node_names = [f"{prefix}_{dim}L_node{i:03d}" for i in range(dim)]
        top_df = benchmark.build_top_gene_table(score_matrix, node_names, gene_names, gene_keys)
        write_fdr_outputs(method, top_df, fig2, canonical_pathways, oncogenic_pathways, universe_genes)


def build_autoencoder_baselines(
    dim: int,
    avg: dict[str, object],
    benchmark,
    fig2,
    canonical_pathways: dict[str, set[str]],
    oncogenic_pathways: dict[str, set[str]],
    universe_genes: Iterable[str],
    seed: int,
    force: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dropout: float,
    max_ig_samples: int,
    ig_steps: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triad_pathways = {name: genes for name, genes in canonical_pathways.items()}
    for prefix, noisy in [("ae", False), ("dae", True)]:
        method = f"{prefix}_{dim}L"
        c2_path = MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c2_canonical.csv"
        c6_path = MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c6_oncogenic.csv"
        if c2_path.exists() and c6_path.exists() and not force:
            print(f"[skip] {method}")
            continue
        print(f"[build] {method} on {device}")
        rep = benchmark.build_ae_like_representation(
            method_name=method,
            x_pca_pre=avg["x_pca_pre"],
            pca_pre=avg["pca_pre"],
            gene_names=avg["gene_names"],
            gene_keys=avg["gene_keys"],
            sample_ids=avg["sample_ids"],
            pathways=triad_pathways,
            latent_dim=dim,
            seed=seed,
            noisy=noisy,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_ig_samples=max_ig_samples,
            ig_steps=ig_steps,
        )
        prefix_label = prefix.upper()
        node_rename = {f"{method}_{i}": f"{prefix_label}_{dim}L_node{i:03d}" for i in range(dim)}
        top_genes = rep["top_genes"].copy()
        top_genes["node"] = top_genes["node"].map(node_rename)
        gene_scores = rep["gene_scores"].rename(index=node_rename)
        gene_scores_path = MATCHED_BASELINE_DIR / f"{method}_gene_scores.csv"
        save_table(gene_scores.reset_index().rename(columns={"index": "node"}), gene_scores_path)
        write_fdr_outputs(method, top_genes, fig2, canonical_pathways, oncogenic_pathways, universe_genes)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def build_pathway_sparse(pathways: dict[str, set[str]], universe_genes: list[str]):
    clean_universe = []
    seen = set()
    for gene in universe_genes:
        if gene and str(gene).lower() != "nan" and gene not in seen:
            clean_universe.append(gene)
            seen.add(gene)
    gene_to_idx = {gene: idx for idx, gene in enumerate(clean_universe)}
    pathway_names = sorted(pathways)
    rows = []
    cols = []
    for col, pathway in enumerate(pathway_names):
        for gene in pathways[pathway]:
            idx = gene_to_idx.get(gene)
            if idx is not None:
                rows.append(idx)
                cols.append(col)
    data = np.ones(len(rows), dtype=np.int8)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(clean_universe), len(pathway_names)), dtype=np.int16)
    pathway_sizes = np.asarray(matrix.sum(axis=0)).ravel().astype(int)
    valid = pathway_sizes >= 5
    matrix = matrix[:, valid]
    pathway_names = [name for name, keep in zip(pathway_names, valid) if keep]
    pathway_sizes = pathway_sizes[valid]
    return gene_to_idx, pathway_names, matrix, pathway_sizes


def top_gene_sparse_for_model(
    weights: np.ndarray,
    gene_names: list[str],
    gene_to_idx: dict[str, int],
    normalize,
) -> tuple[csr_matrix, np.ndarray]:
    rows = []
    cols = []
    top_counts = []
    for row_idx, scores in enumerate(weights):
        top_idx = np.argpartition(np.abs(scores), -TOP_N)[-TOP_N:]
        genes = []
        for idx in top_idx:
            gene = normalize(gene_names[int(idx)])
            if gene and gene.lower() != "nan" and gene in gene_to_idx:
                genes.append(gene)
        unique_genes = sorted(set(genes))
        top_counts.append(len(unique_genes))
        for gene in unique_genes:
            rows.append(row_idx)
            cols.append(gene_to_idx[gene])
    data = np.ones(len(rows), dtype=np.int8)
    matrix = csr_matrix((data, (rows, cols)), shape=(weights.shape[0], len(gene_to_idx)), dtype=np.int16)
    return matrix, np.asarray(top_counts, dtype=int)


def build_source_vae_comparator(
    branch: str,
    fig2,
    canonical_pathways: dict[str, set[str]],
    universe_genes: list[str],
    force: bool,
) -> None:
    if FISHER_ALTERNATIVE != "greater":
        raise ValueError("Source-VAE sparse hypergeometric recurrence analysis implements one-sided over-representation only.")
    method = f"source_vae_{branch}"
    model_out = SOURCE_VAE_DIR / f"{method}_model_detection_c2_canonical.csv"
    pathway_out = SOURCE_VAE_DIR / f"{method}_pathway_detection_c2_canonical.csv"
    if model_out.exists() and pathway_out.exists() and not force:
        print(f"[skip] {method}")
        return

    weights_path = DEEPHEAL_RESULTS / branch / f"{branch}_joined_weights_pca_noscale_cv.npy"
    metadata_path = DEEPHEAL_RESULTS / branch / f"{branch}_joined_node_metadata_pca_noscale_cv.csv"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing source VAE inputs for {branch}")

    meta = pd.read_csv(metadata_path, encoding="utf-8")
    weights = np.load(weights_path, mmap_mode="r")
    if weights.shape[0] != len(meta):
        raise ValueError(f"{branch} weights rows {weights.shape[0]} do not match metadata rows {len(meta)}")

    gene_to_idx, pathway_names, pathway_matrix, pathway_sizes = build_pathway_sparse(canonical_pathways, universe_genes)
    model_rows = []
    pathway_detected_counts = np.zeros(len(pathway_names), dtype=int)
    n_models = 0
    universe_size = len(gene_to_idx)

    for (latent_dim, run), frame in meta.groupby(["source_latent_dim", "source_run"], sort=False):
        indices = frame["global_node_idx"].to_numpy(dtype=int)
        model_weights = np.asarray(weights[indices, :], dtype=np.float32)
        top_sparse, top_counts = top_gene_sparse_for_model(model_weights, universe_genes, gene_to_idx, fig2.normalize_id)
        overlaps = (top_sparse @ pathway_matrix).toarray().astype(int)
        pvals = np.ones_like(overlaps, dtype=float)
        for j, pathway_size in enumerate(pathway_sizes):
            pvals[:, j] = hypergeom.sf(overlaps[:, j] - 1, universe_size, int(pathway_size), top_counts)
        fdr = np.ones_like(pvals, dtype=float)
        for j in range(pvals.shape[1]):
            fdr[:, j] = bh_adjust(pvals[:, j])
        detected = (fdr < 0.05).any(axis=0)
        pathway_detected_counts += detected.astype(int)
        n_models += 1
        model_rows.append(
            {
                "method": method,
                "method_label": str(METHOD_META[method]["display"]),
                "method_tick": str(METHOD_META[method]["tick"]),
                "model_key": f"L{int(latent_dim)}_run{int(run):03d}",
                "source_latent_dim": int(latent_dim),
                "source_run": int(run),
                "n_source_nodes": int(len(indices)),
                "n_pathways_tested": int(len(pathway_names)),
                "n_detected_pathways": int(detected.sum()),
                "pct_pathways_detected": float(detected.mean() * 100.0),
            }
        )
        if n_models % 50 == 0:
            print(f"[{method}] models {n_models}/600")

    pathway_df = pd.DataFrame(
        {
            "method": method,
            "pathway": pathway_names,
            "n_source_models_detected": pathway_detected_counts,
            "pct_source_models_detected": pathway_detected_counts / max(n_models, 1) * 100.0,
            "n_source_models_total": n_models,
        }
    ).sort_values(["n_source_models_detected", "pathway"], ascending=[False, True])
    save_table(pd.DataFrame(model_rows), model_out)
    save_table(pathway_df, pathway_out)


def build_all(args: argparse.Namespace) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MATCHED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_VAE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = import_script(BENCHMARK_SCRIPT, "deepheal_competitor_benchmark")
    fig2 = import_script(FIG2_SCRIPT, "deepheal_pathway_comparison")
    benchmark.set_seed(args.seed)

    gmt_dir = PATHWAY_GMT_DIR
    c2_path = fig2.ensure_gmt(gmt_dir, "c2.cp.v6.2.symbols.gmt")
    c6_path = fig2.ensure_gmt(gmt_dir, "c6.all.v6.2.symbols.gmt")
    universe_genes = fig2.load_universe_gene_names(MANUSCRIPT_ROOT)
    canonical_pathways = fig2.parse_gmt(c2_path, collection="canonical")
    oncogenic_pathways = fig2.parse_gmt(c6_path, collection="oncogenic")

    if args.baselines:
        avg = benchmark.load_average_matrix(MANUSCRIPT_ROOT)
        for dim in args.dims:
            print(f"[matched baselines] {dim}L")
            build_linear_baselines(
                dim=dim,
                avg=avg,
                benchmark=benchmark,
                fig2=fig2,
                canonical_pathways=canonical_pathways,
                oncogenic_pathways=oncogenic_pathways,
                universe_genes=universe_genes,
                seed=args.seed,
                force=args.force,
            )
            build_autoencoder_baselines(
                dim=dim,
                avg=avg,
                benchmark=benchmark,
                fig2=fig2,
                canonical_pathways=canonical_pathways,
                oncogenic_pathways=oncogenic_pathways,
                universe_genes=universe_genes,
                seed=args.seed,
                force=args.force,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                max_ig_samples=args.max_ig_samples,
                ig_steps=args.ig_steps,
            )

    if args.source_vae:
        for spec in COMPARISON_SPECS.values():
            method = str(spec["source_vae_method"])
            branch = str(METHOD_META[method]["source_branch"])
            print(f"[source VAE comparator] {branch}")
            build_source_vae_comparator(
                branch=branch,
                fig2=fig2,
                canonical_pathways=canonical_pathways,
                universe_genes=universe_genes,
                force=args.force,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched DeepHeal pathway-comparison inputs.")
    parser.add_argument("--dims", type=int, nargs="+", default=[257, 194])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=750)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-ig-samples", type=int, default=100)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-vae", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_all(parse_args())
