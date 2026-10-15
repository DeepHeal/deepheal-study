#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import copy
import importlib.util
import json
import math
import pickle
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import anderson
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, Dataset, TensorDataset


CLASS_TO_INT = {"Non-Antineoplastic": 0, "Antineoplastic": 1, "Unknown": -1}
KL_MAX_BETA = 0.1
KL_WARMUP_EPOCHS = 30
DROPOUT_OPTIONS = [0.0, 0.2, 0.4, 0.6]
INTERMEDIATE_LAYER_OPTIONS = [(50, 5), (100, 25), (250, 50), (250, 100), (300, 150)]
DEFAULT_FINAL_VAL_FRACTION = 0.1
THRESHOLD_LOG2FC = 0.15


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(s: object) -> str:
    text = str(s).strip().replace("\xa0", " ")
    text = re.sub(r"\s*_\s*", "_", text)
    return text


def parse_dims(text: str) -> list[int]:
    dims = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not dims:
        raise ValueError("No latent dimensions were provided")
    if any(x <= 0 for x in dims):
        raise ValueError("Latent dimensions must be positive")
    return dims


def parse_dropout_options(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("No dropout options were provided")
    if any(x < 0 or x >= 1 for x in values):
        raise ValueError("Dropout values must be in [0, 1).")
    return values


def parse_intermediate_configs(text: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    normalized = str(text).replace(",", ";")
    for chunk in normalized.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            left, right = chunk.split(":", 1)
        elif "x" in chunk.lower():
            left, right = chunk.lower().split("x", 1)
        else:
            raise ValueError(
                "Intermediate-layer configurations must use 'h1:h2' or 'h1xh2' format, "
                f"but received '{chunk}'."
            )
        configs.append((int(left.strip()), int(right.strip())))
    if not configs:
        raise ValueError("No intermediate-layer configurations were provided")
    if any(h1 <= 0 or h2 <= 0 for h1, h2 in configs):
        raise ValueError("Intermediate-layer widths must be positive integers")
    return configs


def dropout_tag(value: float) -> str:
    return f"{int(round(value * 100)):02d}"


class GMeans:
    """G-means clustering. See THIRD_PARTY_NOTICES.md for source and license."""

    def __init__(self, min_obs: int = 1, max_depth: int = 10, random_state: int | None = None, strictness: int = 3):
        self.max_depth = max_depth
        self.min_obs = min_obs
        self.random_state = random_state
        if strictness not in range(5):
            raise ValueError("strictness must be an integer from 0 to 4")
        self.strictness = strictness
        self.stopping_criteria: list[str] = []

    def _gaussian_check(self, vector: np.ndarray) -> bool:
        output = anderson(vector)
        return bool(output.statistic <= output.critical_values[self.strictness])

    def _recursive_clustering(self, data: np.ndarray, depth: int, index: np.ndarray) -> None:
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append("max_depth")
            return

        km = MiniBatchKMeans(n_clusters=2, random_state=self.random_state)
        km.fit(data)

        centers = km.cluster_centers_
        v = centers[0] - centers[1]
        denom = float(v.dot(v))
        if denom <= 1e-12:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append("degenerate_split")
            return

        x_prime = scale(data.dot(v) / denom)
        gaussian = self._gaussian_check(x_prime)
        if gaussian:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append("gaussian")
            return

        for label in sorted(set(km.labels_.tolist())):
            current_data = data[km.labels_ == label]
            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append("min_obs")
                return

            current_index = index[km.labels_ == label].copy()
            current_index[:, 1] = np.random.randint(0, 2_000_000_000)
            self._recursive_clustering(current_data, depth=depth, index=current_index)

    def fit(self, data: np.ndarray):
        data_index = np.column_stack(
            [
                np.arange(data.shape[0], dtype=np.int64),
                np.full(data.shape[0], -1, dtype=np.int64),
            ]
        )
        self.data_index = data_index
        self._recursive_clustering(data=data, depth=0, index=data_index)
        self.labels_ = np.asarray(self.data_index[:, 1])
        return self


class AverageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        source_col = "x_model" if "x_model" in df.columns else "x_pca"
        self.vectors = np.stack(df[source_col].to_list()).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.vectors))

    def __getitem__(self, idx: int):
        return torch.tensor(self.vectors[idx], dtype=torch.float32)


class AverageVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mu = nn.Linear(hidden_2, latent_dim)
        self.logvar = nn.Linear(hidden_2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_2, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def cache_stem(
    latent_dim: int,
    run_i: int,
    hidden_dims: tuple[int, int] | None = None,
    dropout_rate: float | None = None,
) -> str:
    stem = f"dim{latent_dim:03d}_run{run_i:03d}"
    if hidden_dims is not None:
        stem += f"_h{int(hidden_dims[0]):03d}_{int(hidden_dims[1]):03d}"
    if dropout_rate is not None:
        stem += f"_do{dropout_tag(dropout_rate)}"
    return stem


def make_node_names(n_latents: int) -> list[str]:
    return [f"Average_{i}" for i in range(n_latents)]


def map_gene_weights(raw_weights_df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path)
    uid_to_gene = {
        norm(uid): norm(gene)
        for uid, gene in zip(mapping_df["UniprotID"], mapping_df["Gene Name"])
    }
    mapped_cols = [uid_to_gene.get(norm(col), norm(col)) for col in raw_weights_df.columns]
    mapped = raw_weights_df.copy()
    mapped.columns = mapped_cols
    mapped = mapped.T.groupby(level=0).mean().T
    mapped = mapped.loc[:, [c for c in mapped.columns if c and str(c).lower() != "nan"]]
    return mapped


def save_top_genes(weights_df_mapped: pd.DataFrame, out_dir: Path) -> None:
    gene_names = list(weights_df_mapped.columns)
    rows: list[dict[str, object]] = []
    for node_idx, node_name in enumerate(weights_df_mapped.index):
        row = weights_df_mapped.loc[node_name].to_numpy(dtype=float)
        order = np.argsort(row)[::-1][:31]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "node": node_name,
                    "latent_idx": node_idx,
                    "rank": rank,
                    "gene_name": gene_names[idx],
                    "score": float(row[idx]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "average_latent_top_genes.csv", index=False)


def prepare_average_dataframe(
    root: Path,
    out_dir: Path,
    seed: int,
    pca_components: int,
    pca_reference: str,
) -> tuple[pd.DataFrame, None, PCA, list[str], dict[str, object]]:
    set_seed(seed)

    meta = pd.read_csv(root / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv")
    meta["Drug_Name"] = meta["Drug_Name"].map(norm)
    meta["Sample_ID"] = meta["Sample_ID"].map(norm)
    meta["Drug_Class"] = meta["Drug_Class"].map(norm)
    meta = meta[meta["Drug_Class"].isin(CLASS_TO_INT.keys())].copy()
    meta = meta.set_index("Sample_ID")

    ref_cols = pd.read_csv(root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv", nrows=0).columns.tolist()
    ref_feature_order = [norm(c) for c in ref_cols if c != "Sample_ID"]

    avg = pd.read_csv(root / "data" / "raw" / "875_Compounds_Average_Data.csv", encoding="latin-1")
    avg.columns = [norm(c) for c in avg.columns]
    avg["Gene Name"] = avg["Gene Name"].map(norm)
    avg["UniprotID"] = avg["UniprotID"].map(norm)
    sample_ids = [norm(c) for c in avg.columns.tolist()[2:]]

    avg_matrix = avg.set_index("UniprotID")[sample_ids]
    avg_matrix = avg_matrix.loc[ref_feature_order]
    avg_matrix = avg_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    avg_matrix = avg_matrix.where(avg_matrix.abs() > THRESHOLD_LOG2FC, 0.0)
    x_avg = avg_matrix.to_numpy(dtype=np.float32).T

    if pca_reference == "replicate":
        ref_data = pd.read_csv(root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv")
        ref_data.columns = [norm(c) for c in ref_data.columns]
        ref_matrix = ref_data[ref_feature_order].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        ref_matrix = ref_matrix.where(ref_matrix.abs() > THRESHOLD_LOG2FC, 0.0)
        pca_fit_matrix = ref_matrix.to_numpy(dtype=np.float32)
        pca_reference_note = (
            "PCA was fit directly on the thresholded replicate-aware 1752-sample log2FC matrix without StandardScaler, "
            "then applied to the 875 averaged profiles."
        )
    else:
        pca_fit_matrix = x_avg
        pca_reference_note = (
            "PCA was fit directly on the thresholded 875 replicate-averaged log2FC profiles without StandardScaler."
        )

    n_comp = min(pca_components, pca_fit_matrix.shape[0], pca_fit_matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(pca_fit_matrix)

    rows: list[dict[str, object]] = []
    for sample_id, x_raw in zip(sample_ids, x_avg):
        if sample_id not in meta.index:
            continue
        x_pca = pca.transform(x_raw.reshape(1, -1)).squeeze(0).astype(np.float32)
        rows.append(
            {
                "Sample_ID": sample_id,
                "Drug_Name": str(meta.at[sample_id, "Drug_Name"]),
                "Drug_Class": str(meta.at[sample_id, "Drug_Class"]),
                "Label_Int": int(CLASS_TO_INT[str(meta.at[sample_id, "Drug_Class"])]),
                "Cluster": int(meta.at[sample_id, "Cluster"]),
                "x": x_raw.astype(np.float32),
                "x_model": x_pca,
                "x_pca": x_pca,
            }
        )

    df = pd.DataFrame(rows).sort_values("Drug_Name").reset_index(drop=True)
    df[["Sample_ID", "Drug_Name", "Drug_Class", "Label_Int", "Cluster"]].to_csv(
        out_dir / "average_sample_manifest.csv",
        index=False,
    )

    with open(out_dir / "average_scaler.pkl", "wb") as f:
        pickle.dump(None, f)
    with open(out_dir / "average_pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    prep_summary = {
        "pca_reference": str(pca_reference),
        "pca_reference_note": pca_reference_note,
        "pca_components_requested": int(pca_components),
        "pca_components_actual": int(pca.n_components_),
        "used_standard_scaler": False,
        "used_pca": True,
        "model_input_dim": int(pca.n_components_),
        "n_samples_average": int(len(df)),
        "n_features_proteins": int(len(ref_feature_order)),
        "threshold_log2fc": float(THRESHOLD_LOG2FC),
    }
    with open(out_dir / "average_preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(prep_summary, f, indent=2)

    return df, None, pca, ref_feature_order, prep_summary


def train_average_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, int],
    dropout_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> tuple[AverageVAE, pd.DataFrame, dict[str, float | int]]:
    set_seed(seed)
    model = AverageVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    train_loader = DataLoader(AverageDataset(train_df), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(AverageDataset(val_df), batch_size=batch_size, shuffle=False, drop_last=False)

    best_val_recon = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    logs: list[dict[str, float | int]] = []

    for ep in range(1, epochs + 1):
        model.train()
        beta = min(KL_MAX_BETA, KL_MAX_BETA * ep / max(1, KL_WARMUP_EPOCHS))
        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        n_train_batches = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar = model(batch_x)
            loss_recon = F.mse_loss(recon, batch_x)
            loss_kl = kl(mu, logvar)
            loss = loss_recon + beta * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_sum += float(loss.item())
            train_recon_sum += float(loss_recon.item())
            train_kl_sum += float(loss_kl.item())
            n_train_batches += 1

        model.eval()
        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                recon, mu, logvar = model(batch_x)
                loss_recon = F.mse_loss(recon, batch_x)
                loss_kl = kl(mu, logvar)
                loss = loss_recon + beta * loss_kl
                val_total_sum += float(loss.item())
                val_recon_sum += float(loss_recon.item())
                val_kl_sum += float(loss_kl.item())
                n_val_batches += 1

        scheduler.step()
        epoch_log = {
            "epoch": ep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "kl_beta": float(beta),
            "loss_total": float(val_total_sum / max(1, n_val_batches)),
            "loss_recon": float(val_recon_sum / max(1, n_val_batches)),
            "loss_kl": float(val_kl_sum / max(1, n_val_batches)),
            "train_loss_total": float(train_total_sum / max(1, n_train_batches)),
            "train_loss_recon": float(train_recon_sum / max(1, n_train_batches)),
            "train_loss_kl": float(train_kl_sum / max(1, n_train_batches)),
            "val_loss_total": float(val_total_sum / max(1, n_val_batches)),
            "val_loss_recon": float(val_recon_sum / max(1, n_val_batches)),
            "val_loss_kl": float(val_kl_sum / max(1, n_val_batches)),
        }
        logs.append(epoch_log)

        if epoch_log["val_loss_recon"] < best_val_recon - min_delta:
            best_val_recon = float(epoch_log["val_loss_recon"])
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[TRAIN] early stopping at epoch {ep} "
                    f"(best epoch {best_epoch}, best val_recon {best_val_recon:.6f})"
                )
                break

    model.load_state_dict(best_state)
    return (
        model,
        pd.DataFrame(logs),
        {
            "best_loss": float(best_val_recon),
            "best_val_recon": float(best_val_recon),
            "best_epoch": int(best_epoch),
            "n_epochs_trained": int(len(logs)),
        },
    )


def extract_embeddings(model: AverageVAE, df: pd.DataFrame, device: torch.device, batch_size: int = 256) -> np.ndarray:
    source_col = "x_model" if "x_model" in df.columns else "x_pca"
    x = np.stack(df[source_col].to_list()).astype(np.float32)
    loader = DataLoader(torch.FloatTensor(x), batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            embeddings.append(mu.cpu().numpy())
    return np.vstack(embeddings)


def cross_validate_average_config(
    df: pd.DataFrame,
    device: torch.device,
    latent_dim: int,
    hidden_dims: tuple[int, int],
    dropout_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    n_splits: int,
) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    source_col = "x_model" if "x_model" in df.columns else "x_pca"
    input_dim = int(len(df.iloc[0][source_col]))
    rows: list[dict[str, float | int]] = []
    all_idx = np.arange(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        fold_seed = seed + latent_dim * 10_000 + fold * 100 + int(round(dropout_rate * 100))
        _, log_df, info = train_average_model(
            train_df=train_df,
            val_df=val_df,
            device=device,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            seed=fold_seed,
        )
        best_row = log_df.loc[log_df["epoch"] == int(info["best_epoch"])].iloc[0]
        rows.append(
            {
                "latent_dim": int(latent_dim),
                "dropout_rate": float(dropout_rate),
                "hidden_1": int(hidden_dims[0]),
                "hidden_2": int(hidden_dims[1]),
                "fold": int(fold),
                "best_epoch": int(info["best_epoch"]),
                "n_epochs_trained": int(info["n_epochs_trained"]),
                "best_loss": float(info["best_loss"]),
                "best_val_recon": float(info["best_val_recon"]),
                "best_val_total": float(best_row["val_loss_total"]),
                "best_train_recon": float(best_row["train_loss_recon"]),
            }
        )
    return pd.DataFrame(rows)


def tune_average_hyperparameters(
    df: pd.DataFrame,
    out_dir: Path,
    device: torch.device,
    dims: list[int],
    dropout_options: list[float],
    intermediate_options: list[tuple[int, int]],
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    n_splits: int,
    seed: int,
) -> tuple[float, dict[int, tuple[int, int]], pd.DataFrame, dict[str, object]]:
    cv_cache_dir = out_dir / "tuning_cache_pca_noscale_cv"
    cv_cache_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows: list[pd.DataFrame] = []

    for latent_dim in dims:
        for dropout_rate in dropout_options:
            for hidden_dims in intermediate_options:
                cache_path = cv_cache_dir / (
                    f"cv_dim{latent_dim:03d}_h{hidden_dims[0]:03d}_{hidden_dims[1]:03d}_do{dropout_tag(dropout_rate)}.csv"
                )
                if cache_path.exists():
                    fold_df = pd.read_csv(cache_path)
                else:
                    print(
                        f"[TUNE] latent_dim={latent_dim} hidden={hidden_dims} "
                        f"dropout={dropout_rate:.2f} folds={n_splits}"
                    )
                    fold_df = cross_validate_average_config(
                        df=df,
                        device=device,
                        latent_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        dropout_rate=dropout_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        min_delta=min_delta,
                        seed=seed,
                        n_splits=n_splits,
                    )
                    fold_df.to_csv(cache_path, index=False)
                tuning_rows.append(fold_df)

    tuning_df = pd.concat(tuning_rows, ignore_index=True)
    tuning_df.to_csv(out_dir / "average_hyperparameter_tuning_cv.csv", index=False)

    dropout_summary = (
        tuning_df.groupby("dropout_rate", as_index=False)["best_val_recon"]
        .agg(mean_best_val_recon="mean", std_best_val_recon="std", n_evals="count")
        .sort_values(["mean_best_val_recon", "dropout_rate"], ascending=[True, True])
        .reset_index(drop=True)
    )
    best_dropout = float(dropout_summary.iloc[0]["dropout_rate"])

    selected_hidden: dict[int, tuple[int, int]] = {}
    per_dim_summary_rows: list[dict[str, object]] = []
    best_dropout_df = tuning_df.loc[np.isclose(tuning_df["dropout_rate"], best_dropout)].copy()
    for latent_dim in dims:
        sub = (
            best_dropout_df.loc[best_dropout_df["latent_dim"] == latent_dim]
            .groupby(["hidden_1", "hidden_2"], as_index=False)["best_val_recon"]
            .agg(mean_best_val_recon="mean", std_best_val_recon="std", n_folds="count")
            .sort_values(["mean_best_val_recon", "hidden_1", "hidden_2"], ascending=[True, True, True])
            .reset_index(drop=True)
        )
        best_row = sub.iloc[0]
        selected_hidden[int(latent_dim)] = (int(best_row["hidden_1"]), int(best_row["hidden_2"]))
        per_dim_summary_rows.append(
            {
                "latent_dim": int(latent_dim),
                "selected_hidden_1": int(best_row["hidden_1"]),
                "selected_hidden_2": int(best_row["hidden_2"]),
                "mean_best_val_recon": float(best_row["mean_best_val_recon"]),
                "std_best_val_recon": float(best_row["std_best_val_recon"]) if pd.notna(best_row["std_best_val_recon"]) else 0.0,
                "n_folds": int(best_row["n_folds"]),
            }
        )

    selection = {
        "protocol": "5-fold cross-validation with validation reconstruction error",
        "dropout_options": [float(x) for x in dropout_options],
        "intermediate_layer_options": [[int(h1), int(h2)] for h1, h2 in intermediate_options],
        "tune_epochs": int(epochs),
        "tune_patience": int(patience),
        "tune_min_delta": float(min_delta),
        "tune_folds": int(n_splits),
        "selected_dropout_rate": best_dropout,
        "selected_hidden_dims_by_latent": {
            str(int(latent_dim)): [int(hidden[0]), int(hidden[1])] for latent_dim, hidden in selected_hidden.items()
        },
        "dropout_summary": dropout_summary.to_dict(orient="records"),
        "per_latent_summary": per_dim_summary_rows,
    }
    with open(out_dir / "average_hyperparameter_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    return best_dropout, selected_hidden, tuning_df, selection


def invalidate_stale_average_cache(
    out_dir: Path,
    expected_threshold: float,
    expected_dims: list[int],
    expected_dropout_options: list[float],
    expected_intermediate_options: list[tuple[int, int]],
    expected_pca_reference: str,
) -> None:
    manifest_path = out_dir / "average_cache_manifest.json"
    expected_manifest = {
        "threshold_log2fc": float(expected_threshold),
        "dims": [int(x) for x in expected_dims],
        "dropout_options": [float(x) for x in expected_dropout_options],
        "intermediate_layer_options": [[int(h1), int(h2)] for h1, h2 in expected_intermediate_options],
        "pca_reference": str(expected_pca_reference),
    }

    cached_manifest: dict[str, object] | None = None
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cached_manifest = None

    if cached_manifest == expected_manifest:
        return

    stale_paths = [
        out_dir / "tuning_cache_pca_noscale_cv",
        out_dir / "average_hyperparameter_selection.json",
        out_dir / "average_hyperparameter_tuning_cv.csv",
        out_dir / "average_joined_embeddings_pca_noscale_cv.npy",
        out_dir / "average_joined_weights_pca_noscale_cv.npy",
        out_dir / "average_joined_node_metadata_pca_noscale_cv.csv",
        out_dir / "average_model_bank_summary.csv",
        out_dir / "average_training_log_bank.csv",
        out_dir / "model_bank_cache_pca_noscale_cv",
    ]
    for path in stale_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")


def aggregate_embeddings(
    df: pd.DataFrame,
    ensemble_embeddings: np.ndarray,
    latent_dim: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        z = ensemble_embeddings[idx]
        rows.append(
            {
                "Sample_ID": row["Sample_ID"],
                "Drug_Name": row["Drug_Name"],
                "Drug_Class": row["Drug_Class"],
                "Label_Int": int(row["Label_Int"]),
                "Cluster": int(row["Cluster"]),
                "embedding_count": 1,
                "embedding_source": "deepheal_average_ensemble",
                **{f"latent_{i:03d}": float(v) for i, v in enumerate(z[:latent_dim])},
            }
        )
    return pd.DataFrame(rows)


def save_pathway_results(root: Path, out_dir: Path, pathway_mod, weights_df_mapped: pd.DataFrame, alpha: float) -> None:
    node_names = list(weights_df_mapped.index)
    gene_names = list(weights_df_mapped.columns)
    ensemble_weights = weights_df_mapped.to_numpy(dtype=float)

    analyses = [
        {
            "stem": "average",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c2.cp.v6.2.symbols.gmt",
            "uncorr": out_dir / "average_uncorrected_pvals_cp.csv",
            "title": "Top enriched canonical pathways across average-profile ensemble latent dimensions",
        },
        {
            "stem": "average_c6",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c6.all.v6.2.symbols.gmt",
            "uncorr": out_dir / "average_uncorrected_pvals_c6.csv",
            "title": "Top oncogenic signatures across average-profile ensemble latent dimensions",
        },
    ]

    for analysis in analyses:
        pw_names, pw_genes, _ = pathway_mod.parse_gmt(analysis["gmt"])
        pathway_matrix, _, valid_mask = pathway_mod.create_pathway_matrix(pw_names, pw_genes, gene_names)
        keep_idx = np.where(valid_mask)[0]
        pathway_matrix = pathway_matrix[:, keep_idx]
        pw_names_filtered = [pw_names[i] for i in keep_idx]
        avg_len = np.mean(pathway_matrix.sum(axis=0))
        top_n = int(np.ceil(avg_len))
        print(f"[INFO] {analysis['stem']} pathway_matrix={pathway_matrix.shape}, top_n={top_n}")

        p_vals = pathway_mod.fishers_test_official(pathway_matrix, ensemble_weights, top_n=top_n)
        corrected = pathway_mod.apply_fdr_per_pathway(p_vals, alpha=alpha)
        pd.DataFrame(p_vals, index=node_names, columns=pw_names_filtered).to_csv(analysis["uncorr"])
        freq_df = pathway_mod.save_summary_tables(
            corrected,
            node_names,
            pw_names_filtered,
            out_dir,
            stem=analysis["stem"],
            alpha=alpha,
        )
        pathway_mod.plot_frequency(freq_df, out_dir, stem=analysis["stem"], title=analysis["title"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ARTICLE_ROOT))
    ap.add_argument("--results-subdir", default="results/average")
    ap.add_argument("--dims", default="5,10,25,50,75,100")
    ap.add_argument("--runs-per-dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dropout-options", default="0,0.2,0.4,0.6")
    ap.add_argument("--intermediate-configs", default="50:5;100:25;250:50;250:100;300:150")
    ap.add_argument("--tune-folds", type=int, default=5)
    ap.add_argument("--tune-epochs", type=int, default=80)
    ap.add_argument("--tune-patience", type=int, default=12)
    ap.add_argument("--tune-min-delta", type=float, default=1e-4)
    ap.add_argument("--final-val-fraction", type=float, default=DEFAULT_FINAL_VAL_FRACTION)
    ap.add_argument("--pca-components", type=int, default=1000)
    ap.add_argument("--pca-reference", choices=["replicate", "average"], default="average")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--ig-steps", type=int, default=25)
    ap.add_argument("--ig-max-samples", type=int, default=64)
    ap.add_argument("--ig-batch", type=int, default=16)
    ap.add_argument("--gmeans-strictness", type=int, default=3)
    ap.add_argument("--gmeans-max-depth", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = root / "results" / "average"
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = parse_dims(a.dims)
    dropout_options = parse_dropout_options(a.dropout_options)
    intermediate_options = parse_intermediate_configs(a.intermediate_configs)
    total_models = len(dims) * a.runs_per_dim

    script_dir = Path(__file__).resolve().parent
    pathway_mod = load_module(script_dir / 'attribution_pathways.py', "supcon_pathway_deepheal_average")
    pathway_mod.set_seed(a.seed)
    set_seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] dims={dims}, runs_per_dim={a.runs_per_dim}")
    print(f"[INFO] dropout_options={dropout_options}")
    print(f"[INFO] intermediate_options={intermediate_options}")

    df, _, pca, feature_ids, prep_summary = prepare_average_dataframe(
        root=root,
        out_dir=out_dir,
        seed=a.seed,
        pca_components=a.pca_components,
        pca_reference=a.pca_reference,
    )
    invalidate_stale_average_cache(
        out_dir=out_dir,
        expected_threshold=float(prep_summary.get("threshold_log2fc", 0.15)),
        expected_dims=dims,
        expected_dropout_options=dropout_options,
        expected_intermediate_options=intermediate_options,
        expected_pca_reference=a.pca_reference,
    )
    input_dim = int(prep_summary["model_input_dim"])
    data_model = np.stack(df["x_model"].to_list()).astype(np.float32)

    selection_path = out_dir / "average_hyperparameter_selection.json"
    if selection_path.exists():
        tuning_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_dropout = float(tuning_selection["selected_dropout_rate"])
        selected_hidden = {
            int(k): (int(v[0]), int(v[1])) for k, v in tuning_selection["selected_hidden_dims_by_latent"].items()
        }
        print(f"[INFO] Reusing tuned hyperparameters from {selection_path.name}")
    else:
        selected_dropout, selected_hidden, _, tuning_selection = tune_average_hyperparameters(
            df=df,
            out_dir=out_dir,
            device=device,
            dims=dims,
            dropout_options=dropout_options,
            intermediate_options=intermediate_options,
            batch_size=a.batch_size,
            epochs=a.tune_epochs,
            patience=a.tune_patience,
            min_delta=a.tune_min_delta,
            n_splits=a.tune_folds,
            seed=a.seed,
        )
    print(f"[INFO] selected_dropout={selected_dropout:.2f}")
    print(f"[INFO] selected_hidden={selected_hidden}")

    joined_embeddings_path = out_dir / "average_joined_embeddings_pca_noscale_cv.npy"
    joined_weights_path = out_dir / "average_joined_weights_pca_noscale_cv.npy"
    node_meta_path = out_dir / "average_joined_node_metadata_pca_noscale_cv.csv"
    model_summary_path = out_dir / "average_model_bank_summary.csv"
    training_log_bank_path = out_dir / "average_training_log_bank.csv"
    bank_cache_dir = out_dir / "model_bank_cache_pca_noscale_cv"
    bank_cache_dir.mkdir(parents=True, exist_ok=True)

    if (
        joined_embeddings_path.exists()
        and joined_weights_path.exists()
        and node_meta_path.exists()
        and model_summary_path.exists()
    ):
        print("[INFO] Loading cached joined embeddings/weights from disk")
        joined_embeddings = np.load(joined_embeddings_path)
        joined_weights = np.load(joined_weights_path)
        node_meta_df = pd.read_csv(node_meta_path)
    else:
        embedding_parts: list[np.ndarray] = []
        gene_weight_parts: list[np.ndarray] = []
        model_rows: list[dict[str, object]] = []
        log_tables: list[pd.DataFrame] = []
        node_meta_rows: list[dict[str, object]] = []
        node_offset = 0

        for dim_i, latent_dim in enumerate(dims, start=1):
            for run_i in range(a.runs_per_dim):
                model_idx = (dim_i - 1) * a.runs_per_dim + run_i + 1
                model_seed = a.seed + latent_dim * 1000 + run_i
                hidden_dims = selected_hidden[int(latent_dim)]
                stem = cache_stem(latent_dim, run_i, hidden_dims=hidden_dims, dropout_rate=selected_dropout)
                emb_cache = bank_cache_dir / f"{stem}_embeddings.npy"
                weight_cache = bank_cache_dir / f"{stem}_weights.npy"
                summary_cache = bank_cache_dir / f"{stem}_summary.json"
                log_cache = bank_cache_dir / f"{stem}_log.csv"
                print(
                    f"[BANK] model {model_idx}/{total_models}: latent_dim={latent_dim}, "
                    f"run={run_i + 1}/{a.runs_per_dim}, seed={model_seed}, "
                    f"hidden={hidden_dims}, dropout={selected_dropout:.2f}"
                )

                if emb_cache.exists() and weight_cache.exists() and summary_cache.exists():
                    print(f"[CACHE] Reusing cached source model {stem}")
                    embeddings = np.load(emb_cache)
                    gene_weights_t = np.load(weight_cache)
                    with open(summary_cache, "r", encoding="utf-8") as f:
                        model_summary = json.load(f)
                    if log_cache.exists():
                        log_tables.append(pd.read_csv(log_cache))
                else:
                    train_idx, val_idx = train_test_split(
                        np.arange(len(df), dtype=int),
                        test_size=float(a.final_val_fraction),
                        random_state=model_seed,
                        shuffle=True,
                    )
                    train_df = df.iloc[train_idx].reset_index(drop=True)
                    val_df = df.iloc[val_idx].reset_index(drop=True)
                    model, log_df, train_info = train_average_model(
                        train_df=train_df,
                        val_df=val_df,
                        device=device,
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        dropout_rate=selected_dropout,
                        batch_size=a.batch_size,
                        epochs=a.epochs,
                        patience=a.patience,
                        min_delta=a.min_delta,
                        seed=model_seed,
                    )
                    log_df["source_latent_dim"] = latent_dim
                    log_df["source_run"] = run_i
                    log_df["hidden_1"] = int(hidden_dims[0])
                    log_df["hidden_2"] = int(hidden_dims[1])
                    log_df["dropout_rate"] = float(selected_dropout)
                    log_tables.append(log_df)
                    log_df.to_csv(log_cache, index=False)

                    embeddings = extract_embeddings(model, df, device=device, batch_size=max(128, a.batch_size))
                    gene_weights = pathway_mod.compute_gene_attributions(
                        model=model,
                        data_pca=data_model,
                        latent_dim=latent_dim,
                        pca_components=pca.components_,
                        device=device,
                        seed=model_seed,
                        num_steps=a.ig_steps,
                        max_samples=a.ig_max_samples,
                        batch_size=a.ig_batch,
                    )
                    gene_weights_t = gene_weights.T
                    np.save(emb_cache, embeddings)
                    np.save(weight_cache, gene_weights_t)
                    model_summary = {
                        "source_latent_dim": latent_dim,
                        "source_run": run_i,
                        "seed": model_seed,
                        "hidden_1": int(hidden_dims[0]),
                        "hidden_2": int(hidden_dims[1]),
                        "dropout_rate": float(selected_dropout),
                        "validation_fraction": float(a.final_val_fraction),
                        "best_epoch": int(train_info["best_epoch"]),
                        "n_epochs_trained": int(train_info["n_epochs_trained"]),
                        "best_loss": float(train_info["best_loss"]),
                        "best_val_recon": float(train_info["best_val_recon"]),
                    }
                    with open(summary_cache, "w", encoding="utf-8") as f:
                        json.dump(model_summary, f, indent=2)

                embedding_parts.append(embeddings)
                gene_weight_parts.append(gene_weights_t)
                model_rows.append(model_summary)
                for source_node in range(latent_dim):
                    node_meta_rows.append(
                        {
                            "global_node_idx": node_offset + source_node,
                            "source_latent_dim": latent_dim,
                            "source_run": run_i,
                            "source_node_idx": source_node,
                        }
                    )
                node_offset += latent_dim

        joined_embeddings = np.concatenate(embedding_parts, axis=1)
        joined_weights = np.vstack(gene_weight_parts)
        node_meta_df = pd.DataFrame(node_meta_rows)
        np.save(joined_embeddings_path, joined_embeddings)
        np.save(joined_weights_path, joined_weights)
        node_meta_df.to_csv(node_meta_path, index=False)
        pd.DataFrame(model_rows).to_csv(model_summary_path, index=False)
        if log_tables:
            pd.concat(log_tables, ignore_index=True).to_csv(training_log_bank_path, index=False)

    node_profiles = joined_embeddings.T.astype(np.float32)
    print(f"[INFO] joined_embeddings={joined_embeddings.shape}, joined_weights={joined_weights.shape}")

    gmeans = GMeans(
        strictness=a.gmeans_strictness,
        random_state=a.seed,
        max_depth=a.gmeans_max_depth,
    ).fit(node_profiles)
    gmeans_labels = np.asarray(gmeans.labels_)
    selected_dim = int(len(np.unique(gmeans_labels)))
    print(f"[INFO] GMeans selected_dim={selected_dim}")

    kmeans = KMeans(n_clusters=selected_dim, random_state=a.seed, n_init=20)
    kmeans.fit(node_profiles)
    ensemble_labels = np.asarray(kmeans.labels_, dtype=int)

    ensemble_embeddings = np.zeros((joined_embeddings.shape[0], selected_dim), dtype=np.float32)
    ensemble_weights = np.zeros((selected_dim, joined_weights.shape[1]), dtype=np.float32)
    cluster_rows: list[dict[str, object]] = []
    for label in range(selected_dim):
        idx = np.where(ensemble_labels == label)[0]
        ensemble_embeddings[:, label] = joined_embeddings[:, idx].mean(axis=1)
        ensemble_weights[label, :] = joined_weights[idx, :].mean(axis=0)
        cluster_rows.append(
            {
                "ensemble_label": label,
                "n_source_nodes": int(len(idx)),
            }
        )

    node_meta_df["gmeans_label_raw"] = gmeans_labels
    node_meta_df["ensemble_label"] = ensemble_labels
    node_meta_df.to_csv(out_dir / "average_ensemble_node_labels.csv", index=False)
    pd.DataFrame(cluster_rows).sort_values("ensemble_label").to_csv(
        out_dir / "average_ensemble_cluster_sizes.csv",
        index=False,
    )

    emb_df = aggregate_embeddings(df=df, ensemble_embeddings=ensemble_embeddings, latent_dim=selected_dim)
    emb_df.to_csv(out_dir / "average_consensus_embedding.csv", index=False)
    emb_df.to_csv(out_dir / "average_consensus_embedding_full.csv", index=False)

    node_names = make_node_names(selected_dim)
    raw_weights_df = pd.DataFrame(ensemble_weights, index=node_names, columns=feature_ids)
    raw_weights_df.to_csv(out_dir / "average_gene_weights_raw_uniprot.csv")
    weights_df_mapped = map_gene_weights(raw_weights_df, root / "data" / "raw" / "Uniprot_ID_to_Gene_name_mapping.csv")
    weights_df_mapped.to_csv(out_dir / "average_gene_weights.csv")
    save_top_genes(weights_df_mapped, out_dir)
    save_pathway_results(root, out_dir, pathway_mod, weights_df_mapped, alpha=a.alpha)

    summary = {
        "device": str(device),
        "dims": dims,
        "runs_per_dim": int(a.runs_per_dim),
        "total_source_models": int(total_models),
        "total_source_nodes": int(joined_embeddings.shape[1]),
        "selected_ensemble_dim": int(selected_dim),
        "gmeans_strictness": int(a.gmeans_strictness),
        "gmeans_max_depth": int(a.gmeans_max_depth),
        "pca_components_requested": int(a.pca_components),
        "pca_components_actual": prep_summary["pca_components_actual"],
        "pca_reference": prep_summary["pca_reference"],
        "pca_reference_note": prep_summary["pca_reference_note"],
        "used_standard_scaler": bool(prep_summary["used_standard_scaler"]),
        "used_pca": bool(prep_summary["used_pca"]),
        "model_input_dim": int(prep_summary["model_input_dim"]),
        "threshold_log2fc": float(prep_summary.get("threshold_log2fc", 0.15)),
        "selected_dropout_rate": float(selected_dropout),
        "selected_hidden_dims_by_latent": {
            str(int(latent_dim)): [int(hidden[0]), int(hidden[1])] for latent_dim, hidden in selected_hidden.items()
        },
        "tuning_protocol": tuning_selection["protocol"],
        "tune_folds": int(tuning_selection["tune_folds"]),
        "final_validation_fraction": float(a.final_val_fraction),
        "n_drugs_total": int(len(df)),
        "n_known": int((df["Label_Int"].isin([0, 1])).sum()),
        "n_unknown": int((df["Label_Int"] == -1).sum()),
        "ig_steps": int(a.ig_steps),
        "ig_max_samples": int(a.ig_max_samples),
        "ig_batch": int(a.ig_batch),
        "evaluation_protocol": "5-fold CV hyperparameter tuning plus per-run validation reconstruction early stopping for the DeepHeal average-profile ensemble",
        "representation_learning": "unsupervised average log2FC VAE ensemble with thresholded inputs, direct PCA, and no StandardScaler",
    }
    with open(out_dir / "average_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
