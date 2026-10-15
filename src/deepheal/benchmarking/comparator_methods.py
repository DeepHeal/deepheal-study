from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import random

import re

from itertools import combinations

from pathlib import Path

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pandas as pd

import torch

from scipy.stats import fisher_exact

from sklearn.decomposition import FastICA, PCA

from sklearn.preprocessing import StandardScaler

from sklearn.random_projection import GaussianRandomProjection

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_id(value: object) -> str:
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+_", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_average_matrix(root: Path) -> Dict[str, object]:
    matrix = pd.read_csv(root / "data" / "raw" / "875_Compounds_Average_Data.csv", encoding="latin1")
    gene_names = matrix["Gene Name"].astype(str).str.strip().tolist()
    uniprot_ids = matrix["UniprotID"].astype(str).str.strip().tolist()
    gene_keys = [f"{gene}|{uniprot}" for gene, uniprot in zip(gene_names, uniprot_ids)]

    sample_ids = [normalize_id(col) for col in matrix.columns[2:]]
    data = matrix.iloc[:, 2:].copy()
    data.columns = sample_ids
    x = data.T.fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x).astype(np.float32)

    pca_pre = PCA(
        n_components=min(1000, x_scaled.shape[0], x_scaled.shape[1]),
        random_state=0,
    )
    x_pca_pre = pca_pre.fit_transform(x_scaled).astype(np.float32)

    return {
        "matrix": matrix,
        "sample_ids": sample_ids,
        "gene_names": gene_names,
        "uniprot_ids": uniprot_ids,
        "gene_keys": gene_keys,
        "x_scaled": x_scaled,
        "pca_pre": pca_pre,
        "x_pca_pre": x_pca_pre,
    }

def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out

def summarize_fdr_matrix(method_name: str, fdr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = fdr_df.drop(columns="node")
    sig = values.lt(0.05)

    per_latent = pd.DataFrame(
        {
            "method": method_name,
            "node": fdr_df["node"],
            "n_significant_pathways": sig.sum(axis=1).astype(int),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "method": method_name,
                "latent_dims": int(len(fdr_df)),
                "avg_sig_pathways_per_latent": float(per_latent["n_significant_pathways"].mean()),
                "median_sig_pathways_per_latent": float(per_latent["n_significant_pathways"].median()),
                "annotated_latent_pct": float((per_latent["n_significant_pathways"] > 0).mean() * 100.0),
                "unique_sig_pathways": int(sig.any(axis=0).sum()),
                "total_sig_tests": int(sig.values.sum()),
            }
        ]
    )
    return summary, per_latent

def pathway_frequency_table(fdr_df: pd.DataFrame) -> pd.DataFrame:
    values = fdr_df.drop(columns="node")
    sig = values.lt(0.05)
    out = pd.DataFrame(
        {
            "pathway": values.columns,
            "n_significant_nodes": sig.sum(axis=0).astype(int).values,
            "pct_nodes": sig.mean(axis=0).values * 100.0,
            "min_fdr": values.min(axis=0).values,
        }
    )
    return out[out["n_significant_nodes"] > 0].sort_values(
        ["n_significant_nodes", "min_fdr"],
        ascending=[False, True],
    )

def top_set_overlap(top_sets: List[set]) -> Tuple[float, float, float]:
    jaccard_scores = []
    for a, b in combinations(range(len(top_sets)), 2):
        union = len(top_sets[a] | top_sets[b])
        inter = len(top_sets[a] & top_sets[b])
        score = 0.0 if union == 0 else inter / union
        jaccard_scores.append(score)
    if not jaccard_scores:
        return 0.0, 0.0, 0.0
    arr = np.asarray(jaccard_scores, dtype=float)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())

def build_top_gene_table(
    score_matrix: np.ndarray,
    node_names: Sequence[str],
    gene_names: Sequence[str],
    gene_keys: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for node_name, scores in zip(node_names, score_matrix):
        abs_scores = np.abs(np.asarray(scores, dtype=float))
        top_idx = np.argsort(abs_scores)[-31:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "node": node_name,
                    "rank": rank,
                    "gene_name": gene_names[idx],
                    "feature_id": gene_keys[idx],
                    "abs_score": float(abs_scores[idx]),
                }
            )
    return pd.DataFrame(rows)

def build_redundancy_from_top_genes(method_name: str, top_genes_df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = []
    for _, frame in top_genes_df.groupby("node"):
        feature_sets.append(set(frame["feature_id"].astype(str)))
    mean_jaccard, median_jaccard, max_jaccard = top_set_overlap(feature_sets)
    return pd.DataFrame(
        [
            {
                "method": method_name,
                "latent_dims": int(len(feature_sets)),
                "mean_pairwise_jaccard_top31": mean_jaccard,
                "median_pairwise_jaccard_top31": median_jaccard,
                "max_pairwise_jaccard_top31": max_jaccard,
            }
        ]
    )

def threshold_curve(method_name: str, fdr_df: pd.DataFrame) -> pd.DataFrame:
    values = fdr_df.drop(columns="node")
    rows = []
    for threshold in THRESHOLDS:
        annotated_pct = float(values.lt(threshold).any(axis=1).mean() * 100.0)
        rows.append(
            {
                "method": method_name,
                "threshold": threshold,
                "annotated_latent_pct": annotated_pct,
            }
        )
    return pd.DataFrame(rows)

def run_pathway_analysis_from_scores(
    method_name: str,
    score_matrix: np.ndarray,
    node_names: Sequence[str],
    gene_names: Sequence[str],
    gene_keys: Sequence[str],
    pathways: Dict[str, set],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = {gene for gene in gene_names if gene and gene.lower() != "nan"}
    use_pathways = {name: (members & universe) for name, members in pathways.items()}
    use_pathways = {name: members for name, members in use_pathways.items() if len(members) >= 5}
    ordered_pathways = sorted(use_pathways)

    pval_df = pd.DataFrame(index=list(node_names), columns=ordered_pathways, dtype=float)
    for node_name, scores in zip(node_names, score_matrix):
        abs_scores = np.abs(np.asarray(scores, dtype=float))
        top_idx = np.argsort(abs_scores)[-31:]
        top_genes = {gene_names[idx] for idx in top_idx if gene_names[idx] and gene_names[idx].lower() != "nan"}

        n_top = len(top_genes)
        universe_size = len(universe)
        for pathway_name, pathway_genes in use_pathways.items():
            n_pathway = len(pathway_genes)
            overlap = len(top_genes & pathway_genes)
            contingency = [
                [overlap, n_top - overlap],
                [n_pathway - overlap, universe_size - n_pathway - n_top + overlap],
            ]
            pval_df.loc[node_name, pathway_name] = fisher_exact(
                contingency,
                alternative="greater",
            ).pvalue

    fdr_df = pval_df.copy()
    for pathway_name in fdr_df.columns:
        fdr_df[pathway_name] = bh_adjust(fdr_df[pathway_name].to_numpy(dtype=float))
    fdr_df = fdr_df.reset_index().rename(columns={"index": "node"})

    summary, per_latent = summarize_fdr_matrix(method_name, fdr_df)
    freq = pathway_frequency_table(fdr_df)
    top_genes_df = build_top_gene_table(score_matrix, node_names, gene_names, gene_keys)
    return fdr_df, summary, per_latent, freq, top_genes_df

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

def train_autoencoder(
    x_train: np.ndarray,
    latent_dim: int,
    seed: int,
    noisy: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> Autoencoder:
    set_seed(seed)
    model = Autoencoder(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    clean_tensor = torch.tensor(x_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(clean_tensor), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for (clean_batch,) in loader:
            clean_batch = clean_batch.to(device)
            noisy_batch = clean_batch + torch.randn_like(clean_batch) if noisy else clean_batch
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(noisy_batch)
            loss = criterion(reconstructed, clean_batch)
            loss.backward()
            optimizer.step()
    return model

def encode_array(model: Autoencoder, array: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = torch.tensor(array[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model.encode(batch).cpu().numpy())
    return np.vstack(outputs)

def integrated_gradients_for_encoder(
    model: Autoencoder,
    inputs: np.ndarray,
    target_index: int,
    device: torch.device,
    steps: int = 50,
) -> np.ndarray:
    model.eval()
    batch = torch.tensor(inputs, dtype=torch.float32, device=device)
    baseline = torch.zeros_like(batch)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]

    scaled = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (batch - baseline).unsqueeze(0)
    flat = scaled.reshape(-1, batch.shape[1]).requires_grad_(True)
    outputs = model.encode(flat)[:, target_index]
    total = outputs.sum()
    model.zero_grad(set_to_none=True)
    total.backward()
    grads = flat.grad.reshape(steps, batch.shape[0], batch.shape[1])
    avg_grads = grads.mean(dim=0)
    attributions = (batch - baseline) * avg_grads
    return attributions.detach().cpu().numpy()

def build_ae_like_representation(
    method_name: str,
    x_pca_pre: np.ndarray,
    pca_pre: PCA,
    gene_names: Sequence[str],
    gene_keys: Sequence[str],
    sample_ids: Sequence[str],
    pathways: Dict[str, set],
    latent_dim: int,
    seed: int,
    noisy: bool,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dropout: float,
    max_ig_samples: int,
    ig_steps: int,
) -> Dict[str, object]:
    model = train_autoencoder(
        x_train=x_pca_pre,
        latent_dim=latent_dim,
        seed=seed,
        noisy=noisy,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
    )
    embedding = encode_array(model, x_pca_pre, device=device)
    node_names = [f"{method_name}_{i}" for i in range(latent_dim)]
    feature_cols = node_names.copy()
    emb_df = pd.DataFrame(embedding, columns=feature_cols)
    emb_df.insert(0, "Sample_ID", list(sample_ids))

    subset_size = min(max_ig_samples, embedding.shape[0])
    subset_idx = np.linspace(0, embedding.shape[0] - 1, subset_size, dtype=int)
    subset_inputs = x_pca_pre[subset_idx]
    pca_components = pca_pre.components_.astype(np.float32)
    gene_scores = np.zeros((latent_dim, pca_components.shape[1]), dtype=np.float32)

    for latent_idx in range(latent_dim):
        attrs = integrated_gradients_for_encoder(
            model=model,
            inputs=subset_inputs,
            target_index=latent_idx,
            device=device,
            steps=ig_steps,
        )
        gene_attrs = attrs @ pca_components
        gene_scores[latent_idx] = np.mean(np.abs(gene_attrs), axis=0)

    fdr_df, summary, per_latent, freq, top_genes = run_pathway_analysis_from_scores(
        method_name=method_name,
        score_matrix=gene_scores,
        node_names=node_names,
        gene_names=gene_names,
        gene_keys=gene_keys,
        pathways=pathways,
    )
    redundancy = build_redundancy_from_top_genes(method_name, top_genes)
    return {
        "embedding": emb_df,
        "feature_cols": feature_cols,
        "gene_scores": pd.DataFrame(gene_scores, index=node_names, columns=list(gene_keys)),
        "fdr_df": fdr_df,
        "annotation_summary": summary,
        "per_latent": per_latent,
        "freq": freq,
        "top_genes": top_genes,
        "redundancy": redundancy,
        "threshold_curve": threshold_curve(method_name, fdr_df),
    }
