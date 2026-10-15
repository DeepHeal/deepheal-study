#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import math
import pickle
import random
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import fisher_exact


CLASS_TO_INT = {"Non-Antineoplastic": 0, "Antineoplastic": 1, "Unknown": -1}
DATABASE_PREFIXES = OrderedDict(
    {
        "REACTOME": "REACTOME_",
        "BIOCARTA": "BIOCARTA_",
        "KEGG": "KEGG_",
    }
)


def norm(s: object) -> str:
    return str(s).strip().replace("\xa0", " ")


def to_drug(sample_id: str) -> str:
    x = norm(sample_id)
    if x.endswith("_1") or x.endswith("_2"):
        return x.rsplit("_", 1)[0]
    return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SupConModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)


class IntegratedGradients:
    def __init__(self, model: SupConModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain_batch(
        self,
        samples: np.ndarray,
        output_index: int,
        reference: np.ndarray | None = None,
        num_steps: int = 50,
    ) -> np.ndarray:
        n, d = samples.shape
        samples_t = torch.tensor(samples, dtype=torch.float32, device=self.device)
        ref_t = (
            torch.zeros((1, d), dtype=torch.float32, device=self.device)
            if reference is None
            else torch.tensor(reference.reshape(1, -1), dtype=torch.float32, device=self.device)
        )

        steps = num_steps + 1
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device).view(1, steps, 1)
        s = samples_t.unsqueeze(1)
        r = ref_t.unsqueeze(0)
        diff = s - r

        interp_flat = (r + alphas * diff).reshape(n * steps, d)
        interp_flat = interp_flat.detach().requires_grad_(True)

        z_mean, _ = self.model.encode(interp_flat)
        target = z_mean[:, output_index]
        grads = torch.autograd.grad(outputs=target.sum(), inputs=interp_flat, create_graph=False)[0]
        grads = grads.reshape(n, steps, d)
        avg_grads = grads.mean(dim=1)
        attrs = diff.squeeze(1) * avg_grads
        return attrs.detach().cpu().numpy()


def compute_gene_attributions(
    model: SupConModel,
    data_pca: np.ndarray,
    latent_dim: int,
    pca_components: np.ndarray | None,
    device: torch.device,
    seed: int,
    num_steps: int = 50,
    max_samples: int = 100,
    batch_size: int = 25,
) -> np.ndarray:
    ig = IntegratedGradients(model, device)
    n_samples = data_pca.shape[0]
    rng = np.random.default_rng(seed)
    if max_samples and max_samples < n_samples:
        idx = rng.choice(n_samples, max_samples, replace=False)
        subset = data_pca[idx]
    else:
        subset = data_pca

    n = subset.shape[0]
    n_genes = pca_components.shape[1] if pca_components is not None else data_pca.shape[1]
    weights = np.zeros((n_genes, latent_dim), dtype=np.float32)

    for lat in range(latent_dim):
        all_attrs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            attrs = ig.explain_batch(subset[start:end], lat, num_steps=num_steps)
            if pca_components is not None:
                attrs = attrs @ pca_components
            all_attrs.append(attrs)
        all_attrs = np.concatenate(all_attrs, axis=0)
        weights[:, lat] = np.mean(np.abs(all_attrs), axis=0)
        print(f"[IG] latent {lat + 1}/{latent_dim}")

    return weights


def parse_gmt(filepath: Path):
    pathway_names, pathway_genes, pathway_descriptions = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway_names.append(parts[0])
            pathway_descriptions.append(parts[1])
            pathway_genes.append([g.strip() for g in parts[2:] if g.strip()])
    return pathway_names, pathway_genes, pathway_descriptions


def create_pathway_matrix(pathway_names, pathway_genes, gene_names):
    gene_names_arr = np.array(gene_names)
    n_genes = len(gene_names)
    n_pathways = len(pathway_names)
    pathway_matrix = np.zeros((n_genes, n_pathways), dtype=np.int32)
    pathway_lengths = []

    for i in range(n_pathways):
        genes = pathway_genes[i]
        pathway_lengths.append(len(genes))
        for gene in genes:
            idx = np.where(gene_names_arr == gene)[0]
            if len(idx) != 0:
                pathway_matrix[idx[0], i] = 1

    col_sums = pathway_matrix.sum(axis=0)
    valid_mask = col_sums > 0
    return pathway_matrix, pathway_lengths, valid_mask


def fishers_test_official(pathway_matrix, ensemble_weights, top_n: int) -> np.ndarray:
    n_nodes = ensemble_weights.shape[0]
    n_pathways = pathway_matrix.shape[1]
    n_genes_total = pathway_matrix.shape[0]
    p_vals = np.zeros((n_nodes, n_pathways), dtype=np.float64)

    print(f"[PATHWAY] Fisher tests: {n_nodes} nodes x {n_pathways} pathways, top_n={top_n}")
    for i in range(n_nodes):
        gene_indices = ensemble_weights[i, :].argsort()[-top_n:][::-1]
        for j in range(n_pathways):
            pathway_indices = np.where(pathway_matrix[:, j] == 1)[0]
            a = len(np.intersect1d(pathway_indices, gene_indices))
            c = top_n - a
            b = len(pathway_indices) - a
            d = n_genes_total - a - b - c
            table = np.array([[a, b], [c, d]])
            _, pvalue = fisher_exact(table)
            p_vals[i, j] = pvalue
        if (i + 1) % max(1, n_nodes // 8) == 0 or i == n_nodes - 1:
            print(f"[PATHWAY] node {i + 1}/{n_nodes}")
    return p_vals


def apply_fdr_per_pathway(p_vals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    corrected = np.zeros_like(p_vals)
    for j in range(p_vals.shape[1]):
        vals = p_vals[:, j]
        order = np.argsort(vals)
        ranked = vals[order]
        n = len(ranked)
        adj = ranked * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty_like(adj)
        out[order] = np.clip(adj, 0.0, 1.0)
        corrected[:, j] = out
    return corrected


def build_drug_level_targets(root: Path, scaler, pca) -> tuple[list[str], list[str], np.ndarray]:
    data = pd.read_csv(root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv")
    hdr1628 = pd.read_csv(root / "data" / "raw" / "1628_Experiments_Data.csv", nrows=0).columns.tolist()[2:]
    rep_count_1628 = Counter([to_drug(x) for x in hdr1628])

    meta = pd.read_csv(root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv")
    meta["Drug_Name"] = meta["Drug_Name"].map(norm)
    meta["Drug_Class"] = meta["Drug_Class"].map(norm)
    meta = meta[meta["Drug_Class"].isin(CLASS_TO_INT.keys())].copy()
    dmeta = meta.set_index("Drug_Name")

    feats = [c for c in data.columns if c != "Sample_ID"]
    data["Sample_ID"] = data["Sample_ID"].map(norm)
    data["Drug_Name"] = data["Sample_ID"].map(to_drug)
    x_all = data[feats].to_numpy(dtype=np.float32)
    drugs = data["Drug_Name"].to_numpy()

    grouped: dict[str, list[np.ndarray]] = {}
    for i, drug in enumerate(drugs):
        grouped.setdefault(drug, []).append(x_all[i])

    drug_names = []
    targets = []
    for drug, reps in grouped.items():
        if drug not in dmeta.index:
            continue
        if rep_count_1628.get(drug, 0) >= 2 and len(reps) >= 2:
            target = (reps[0] + reps[1]) / 2.0
        else:
            target = reps[0]
        drug_names.append(drug)
        targets.append(target.astype(np.float32))

    target_arr = np.vstack(targets)
    target_pca = pca.transform(scaler.transform(target_arr)).astype(np.float32)
    return drug_names, feats, target_pca


def save_summary_tables(
    corrected_pvals: np.ndarray,
    node_names: list[str],
    pathway_names: list[str],
    out_dir: Path,
    stem: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    sig_mask = corrected_pvals < alpha

    rows = []
    for i, node in enumerate(node_names):
        sig_idx = np.where(sig_mask[i, :])[0]
        if len(sig_idx) == 0:
            continue
        sorted_idx = sig_idx[np.argsort(corrected_pvals[i, sig_idx])]
        for rank, j in enumerate(sorted_idx[:10], start=1):
            rows.append(
                {
                    "node": node,
                    "latent_idx": i,
                    "pathway": pathway_names[j],
                    "fdr": corrected_pvals[i, j],
                    "rank": rank,
                }
            )
    top_df = pd.DataFrame(rows)
    top_df.to_csv(out_dir / f"{stem}_latent_top_pathways.csv", index=False)

    pw_freq = sig_mask.sum(axis=0)
    freq_df = pd.DataFrame(
        {
            "pathway": pathway_names,
            "n_significant_nodes": pw_freq,
            "pct_nodes": pw_freq / len(node_names) * 100.0,
            "min_fdr": corrected_pvals.min(axis=0),
        }
    )
    freq_df = freq_df.sort_values(["n_significant_nodes", "min_fdr"], ascending=[False, True])
    freq_df = freq_df[freq_df["n_significant_nodes"] > 0].reset_index(drop=True)
    freq_df.to_csv(out_dir / f"{stem}_pathway_frequency.csv", index=False)
    pd.DataFrame(corrected_pvals, index=node_names, columns=pathway_names).to_csv(
        out_dir / f"{stem}_fdr_matrix.csv"
    )
    pd.DataFrame(sig_mask.astype(int), index=node_names, columns=pathway_names).to_csv(
        out_dir / f"{stem}_significant_mask.csv"
    )
    return freq_df


def plot_frequency(freq_df: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
    if len(freq_df) == 0:
        return

    top = freq_df.head(20).sort_values(["n_significant_nodes", "min_fdr"], ascending=[True, False])
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
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.barh(
        top["pathway"],
        top["n_significant_nodes"],
        color="#1f78b4",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Number of significant latent nodes")
    ax.set_ylabel("Pathway")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_{stem}_pathway_frequency.png", dpi=300)
    fig.savefig(out_dir / f"figure_{stem}_pathway_frequency.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ARTICLE_ROOT))
    ap.add_argument("--results-subdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--ig-max-samples", type=int, default=100)
    ap.add_argument("--ig-batch", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = root / "results" / "supcon"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    with open(out_dir / "supcon_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / "supcon_pca.pkl", "rb") as f:
        pca = pickle.load(f)

    model = SupConModel(input_dim=int(pca.n_components_), latent_dim=64, proj_dim=32).to(device)
    state_dict = torch.load(out_dir / "supcon_consensus_vae.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    drug_names, feature_ids, target_pca = build_drug_level_targets(root, scaler, pca)
    print(f"[INFO] drug-level targets={target_pca.shape}")

    gene_weights = compute_gene_attributions(
        model=model,
        data_pca=target_pca,
        latent_dim=64,
        pca_components=pca.components_,
        device=device,
        seed=a.seed,
        num_steps=a.ig_steps,
        max_samples=a.ig_max_samples,
        batch_size=a.ig_batch,
    )

    node_names = [f"SupCon_{i}" for i in range(gene_weights.shape[1])]
    weights_df = pd.DataFrame(gene_weights.T, index=node_names, columns=[norm(x) for x in feature_ids])
    weights_df.to_csv(out_dir / "supcon_gene_weights_raw_uniprot.csv")

    mapping_df = pd.read_csv(root / "data" / "raw" / "Uniprot_ID_to_Gene_name_mapping.csv")
    uid_to_gene = {norm(uid): norm(gene) for uid, gene in zip(mapping_df["UniprotID"], mapping_df["Gene Name"])}
    mapped_cols = [uid_to_gene.get(col, col) for col in weights_df.columns]
    weights_df_mapped = weights_df.copy()
    weights_df_mapped.columns = mapped_cols
    weights_df_mapped = weights_df_mapped.T.groupby(level=0).mean().T
    weights_df_mapped = weights_df_mapped.loc[
        :,
        [c for c in weights_df_mapped.columns if c and str(c).lower() != "nan"],
    ]
    weights_df_mapped.to_csv(out_dir / "supcon_gene_weights.csv")

    gene_names = list(weights_df_mapped.columns)
    ensemble_weights = weights_df_mapped.values

    top_gene_rows = []
    for node_idx, node_name in enumerate(node_names):
        row = weights_df_mapped.loc[node_name].to_numpy(dtype=float)
        order = np.argsort(row)[::-1][:31]
        for rank, idx in enumerate(order, start=1):
            top_gene_rows.append(
                {
                    "node": node_name,
                    "latent_idx": node_idx,
                    "rank": rank,
                    "gene_name": gene_names[idx],
                    "score": float(row[idx]),
                }
            )
    pd.DataFrame(top_gene_rows).to_csv(out_dir / "supcon_latent_top_genes.csv", index=False)

    analyses = [
        {
            "stem": "supcon",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c2.cp.v6.2.symbols.gmt",
            "uncorr": out_dir / "supcon_uncorrected_pvals_cp.csv",
            "title": "Top enriched canonical pathways across SupCon latent dimensions",
        },
        {
            "stem": "supcon_c6",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c6.all.v6.2.symbols.gmt",
            "uncorr": out_dir / "supcon_uncorrected_pvals_c6.csv",
            "title": "Top oncogenic signatures across SupCon latent dimensions",
        },
    ]

    for analysis in analyses:
        pw_names, pw_genes, _ = parse_gmt(analysis["gmt"])
        pathway_matrix, _, valid_mask = create_pathway_matrix(pw_names, pw_genes, gene_names)
        keep_idx = np.where(valid_mask)[0]
        pathway_matrix = pathway_matrix[:, keep_idx]
        pw_names_filtered = [pw_names[i] for i in keep_idx]
        avg_len = np.mean(pathway_matrix.sum(axis=0))
        top_n = int(np.ceil(avg_len))
        print(f"[INFO] {analysis['stem']} pathway_matrix={pathway_matrix.shape}, top_n={top_n}")

        p_vals = fishers_test_official(pathway_matrix, ensemble_weights, top_n=top_n)
        corrected = apply_fdr_per_pathway(p_vals, alpha=a.alpha)
        pd.DataFrame(p_vals, index=node_names, columns=pw_names_filtered).to_csv(analysis["uncorr"])
        freq_df = save_summary_tables(
            corrected,
            node_names,
            pw_names_filtered,
            out_dir,
            stem=analysis["stem"],
            alpha=a.alpha,
        )
        plot_frequency(freq_df, out_dir, stem=analysis["stem"], title=analysis["title"])
        if len(freq_df) > 0:
            print(f"[INFO] Top pathways for {analysis['stem']}:")
            print(freq_df.head(15).to_string(index=False))
        else:
            print(f"[WARN] No significant pathways identified for {analysis['stem']} at the current settings.")


if __name__ == "__main__":
    main()
