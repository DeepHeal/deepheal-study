#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import copy
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import anderson
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import scale


DROPOUT_OPTIONS = [0.0, 0.2, 0.4, 0.6]
INTERMEDIATE_LAYER_OPTIONS = [(50, 5), (100, 25), (250, 50), (250, 100), (300, 150)]
DEFAULT_FINAL_VAL_FRACTION = 0.1


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GMeans:
    """Adapted from DeepProfile's gmeans.py (originally flylo/g-means)."""

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
        self.data = data
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
    return f"{int(round(float(value) * 100)):02d}"


def make_node_names(n_latents: int) -> list[str]:
    return [f"SupCon_{i}" for i in range(n_latents)]


def cache_stem(latent_dim: int, run_i: int, hidden_dims: tuple[int, int], dropout_rate: float) -> str:
    return (
        f"dim{latent_dim:03d}_run{run_i:03d}_"
        f"h{int(hidden_dims[0]):03d}_{int(hidden_dims[1]):03d}_do{dropout_tag(dropout_rate)}"
    )


def map_gene_weights(raw_weights_df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path)
    uid_to_gene = {
        str(uid).strip().replace("\xa0", " "): str(gene).strip().replace("\xa0", " ")
        for uid, gene in zip(mapping_df["UniprotID"], mapping_df["Gene Name"])
    }
    mapped_cols = [uid_to_gene.get(str(col), str(col)) for col in raw_weights_df.columns]
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
    pd.DataFrame(rows).to_csv(out_dir / "supcon_latent_top_genes.csv", index=False)


def save_pathway_results(root: Path, out_dir: Path, pathway_mod, weights_df_mapped: pd.DataFrame, alpha: float) -> None:
    node_names = list(weights_df_mapped.index)
    gene_names = list(weights_df_mapped.columns)
    ensemble_weights = weights_df_mapped.to_numpy(dtype=float)

    analyses = [
        {
            "stem": "supcon",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c2.cp.v6.2.symbols.gmt",
            "uncorr": out_dir / "supcon_uncorrected_pvals_cp.csv",
            "title": "Top enriched canonical pathways across SupCon ensemble latent dimensions",
        },
        {
            "stem": "supcon_c6",
            "gmt": root / "data" / "pathways" / "gmt_cache" / "c6.all.v6.2.symbols.gmt",
            "uncorr": out_dir / "supcon_uncorrected_pvals_c6.csv",
            "title": "Top oncogenic signatures across SupCon ensemble latent dimensions",
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
        if len(freq_df) > 0:
            print(f"[INFO] Top pathways for {analysis['stem']}:")
            print(freq_df.head(15).to_string(index=False))
        else:
            print(f"[WARN] No significant pathways identified for {analysis['stem']} at the current settings.")


def build_stratify_key(df: pd.DataFrame, base_mod, min_count: int) -> np.ndarray:
    labels = df["Label_Int"].to_numpy(dtype=int)
    clusters = df["Cluster"].to_numpy(dtype=int)
    return base_mod.build_stratify(labels, clusters, min_count=min_count)


def compute_supcon_loss_components(model, x1, x2, target, labels, is_real, beta: float, base_mod):
    recon1, mu1, logvar1, proj1 = model(x1)
    recon2, mu2, logvar2, proj2 = model(x2)

    recon1_to_target = torch.nn.functional.mse_loss(recon1, target, reduction="none").mean(dim=1)
    recon1_to_self = torch.nn.functional.mse_loss(recon1, x1, reduction="none").mean(dim=1)
    recon2_to_target = torch.nn.functional.mse_loss(recon2, target, reduction="none").mean(dim=1)
    recon2_to_self = torch.nn.functional.mse_loss(recon2, x2, reduction="none").mean(dim=1)
    loss_recon_1 = base_mod.RECON_TARGET_ALPHA * recon1_to_target + (1.0 - base_mod.RECON_TARGET_ALPHA) * recon1_to_self
    loss_recon_2 = base_mod.RECON_TARGET_ALPHA * recon2_to_target + (1.0 - base_mod.RECON_TARGET_ALPHA) * recon2_to_self
    loss_recon = 0.5 * (loss_recon_1.mean() + loss_recon_2.mean())

    loss_kl = 0.5 * (base_mod.kl(mu1, logvar1) + base_mod.kl(mu2, logvar2))
    cons_weight = torch.where(
        is_real > 0.5,
        torch.ones_like(is_real),
        torch.full_like(is_real, base_mod.PSEUDO_CONSISTENCY_WEIGHT),
    )
    loss_cons_per_sample = ((mu1 - mu2) ** 2).mean(dim=1)
    loss_cons = (loss_cons_per_sample * cons_weight).mean()
    loss_supcon = base_mod.supcon(proj1, proj2, labels)
    loss_total = loss_recon + beta * loss_kl + 0.25 * loss_cons + 0.35 * loss_supcon
    return {
        "loss_total": loss_total,
        "loss_recon": loss_recon,
        "loss_kl": loss_kl,
        "loss_cons": loss_cons,
        "loss_supcon": loss_supcon,
    }


def summarize_epoch_losses(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "loss_total": float("nan"),
            "loss_recon": float("nan"),
            "loss_kl": float("nan"),
            "loss_cons": float("nan"),
            "loss_supcon": float("nan"),
        }
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ["loss_total", "loss_recon", "loss_kl", "loss_cons", "loss_supcon"]
    }


def train_supcon_model_with_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_mod,
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
    label_col: str,
):
    base_mod.set_seed(seed)
    model = base_mod.Model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        proj_dim=32,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    sampler = base_mod.BalancedBatchSampler(
        train_df[label_col].to_numpy(dtype=int),
        batch_size=batch_size,
        seed=seed,
        class_ratios=base_mod.BATCH_CLASS_RATIOS,
    )
    train_loader = torch.utils.data.DataLoader(base_mod.DrugDataset(train_df, label_col=label_col), batch_sampler=sampler)
    val_loader = torch.utils.data.DataLoader(
        base_mod.DrugDataset(val_df, label_col=label_col),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val_recon = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    logs: list[dict[str, float | int]] = []

    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        beta = min(base_mod.KL_MAX_BETA, base_mod.KL_MAX_BETA * ep / max(1, base_mod.KL_WARMUP_EPOCHS))

        model.train()
        train_rows: list[dict[str, float]] = []
        for x1, x2, target, labels, is_real in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)
            labels = labels.to(device)
            is_real = is_real.to(device)

            loss_parts = compute_supcon_loss_components(model, x1, x2, target, labels, is_real, beta, base_mod)
            optimizer.zero_grad()
            loss_parts["loss_total"].backward()
            optimizer.step()
            train_rows.append({key: float(val.item()) for key, val in loss_parts.items()})

        model.eval()
        val_rows: list[dict[str, float]] = []
        with torch.no_grad():
            for x1, x2, target, labels, is_real in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                target = target.to(device)
                labels = labels.to(device)
                is_real = is_real.to(device)
                loss_parts = compute_supcon_loss_components(model, x1, x2, target, labels, is_real, beta, base_mod)
                val_rows.append({key: float(val.item()) for key, val in loss_parts.items()})

        scheduler.step()
        train_mean = summarize_epoch_losses(train_rows)
        val_mean = summarize_epoch_losses(val_rows)
        epoch_log = {
            "epoch": ep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "kl_beta": float(beta),
            "loss_total": float(val_mean["loss_total"]),
            "loss_recon": float(val_mean["loss_recon"]),
            "loss_kl": float(val_mean["loss_kl"]),
            "loss_cons": float(val_mean["loss_cons"]),
            "loss_supcon": float(val_mean["loss_supcon"]),
            "train_loss_total": float(train_mean["loss_total"]),
            "train_loss_recon": float(train_mean["loss_recon"]),
            "train_loss_kl": float(train_mean["loss_kl"]),
            "train_loss_cons": float(train_mean["loss_cons"]),
            "train_loss_supcon": float(train_mean["loss_supcon"]),
            "val_loss_total": float(val_mean["loss_total"]),
            "val_loss_recon": float(val_mean["loss_recon"]),
            "val_loss_kl": float(val_mean["loss_kl"]),
            "val_loss_cons": float(val_mean["loss_cons"]),
            "val_loss_supcon": float(val_mean["loss_supcon"]),
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


def cross_validate_supcon_config(
    df: pd.DataFrame,
    base_mod,
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
    stratify_key = build_stratify_key(df, base_mod, min_count=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    source_col = "target_model" if "target_model" in df.columns else "target_pca"
    input_dim = int(len(df.iloc[0][source_col]))
    rows: list[dict[str, float | int]] = []
    all_idx = np.arange(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_idx, stratify_key), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        fold_seed = seed + latent_dim * 10_000 + fold * 100 + int(round(dropout_rate * 100))
        _, log_df, info = train_supcon_model_with_validation(
            train_df=train_df,
            val_df=val_df,
            base_mod=base_mod,
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
            label_col="Label_Int",
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
                "best_val_supcon": float(best_row["val_loss_supcon"]),
                "best_train_recon": float(best_row["train_loss_recon"]),
            }
        )
    return pd.DataFrame(rows)


def tune_supcon_hyperparameters(
    df: pd.DataFrame,
    out_dir: Path,
    base_mod,
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
):
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
                    fold_df = cross_validate_supcon_config(
                        df=df,
                        base_mod=base_mod,
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
    tuning_df.to_csv(out_dir / "supcon_hyperparameter_tuning_cv.csv", index=False)

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
                "std_best_val_recon": float(best_row["std_best_val_recon"])
                if pd.notna(best_row["std_best_val_recon"])
                else 0.0,
                "n_folds": int(best_row["n_folds"]),
            }
        )

    selection = {
        "protocol": "5-fold cross-validation with validation reconstruction error",
        "dims": [int(x) for x in dims],
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
    with open(out_dir / "supcon_hyperparameter_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    return best_dropout, selected_hidden, tuning_df, selection


def invalidate_stale_supcon_cache(
    out_dir: Path,
    expected_threshold: float,
    expected_dims: list[int],
    expected_dropout_options: list[float],
    expected_intermediate_options: list[tuple[int, int]],
) -> None:
    manifest_path = out_dir / "supcon_cache_manifest.json"
    expected_manifest = {
        "threshold_log2fc": float(expected_threshold),
        "dims": [int(x) for x in expected_dims],
        "dropout_options": [float(x) for x in expected_dropout_options],
        "intermediate_layer_options": [[int(h1), int(h2)] for h1, h2 in expected_intermediate_options],
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
        out_dir / "supcon_hyperparameter_selection.json",
        out_dir / "supcon_hyperparameter_tuning_cv.csv",
        out_dir / "supcon_joined_embeddings_pca_noscale_cv.npy",
        out_dir / "supcon_joined_weights_pca_noscale_cv.npy",
        out_dir / "supcon_joined_node_metadata_pca_noscale_cv.csv",
        out_dir / "supcon_model_bank_summary.csv",
        out_dir / "supcon_training_log_bank.csv",
        out_dir / "model_bank_cache_pca_noscale_cv",
    ]
    for path in stale_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ARTICLE_ROOT))
    ap.add_argument("--results-subdir", default="results_supcon_deepprofile_ensemble")
    ap.add_argument("--dims", default="5,10,25,50,75,100")
    ap.add_argument("--runs-per-dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pca-components", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--dropout-options", default=",".join(str(x) for x in DROPOUT_OPTIONS))
    ap.add_argument(
        "--intermediate-configs",
        default=";".join(f"{int(h1)}:{int(h2)}" for h1, h2 in INTERMEDIATE_LAYER_OPTIONS),
    )
    ap.add_argument("--tune-folds", type=int, default=5)
    ap.add_argument("--tune-epochs", type=int, default=80)
    ap.add_argument("--tune-patience", type=int, default=12)
    ap.add_argument("--tune-min-delta", type=float, default=1e-4)
    ap.add_argument("--final-val-fraction", type=float, default=DEFAULT_FINAL_VAL_FRACTION)
    ap.add_argument("--ig-steps", type=int, default=25)
    ap.add_argument("--ig-max-samples", type=int, default=64)
    ap.add_argument("--ig-batch", type=int, default=16)
    ap.add_argument("--gmeans-strictness", type=int, default=3)
    ap.add_argument("--gmeans-max-depth", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = root / "results" / "supcon"
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = parse_dims(a.dims)
    dropout_options = parse_dropout_options(a.dropout_options)
    intermediate_options = parse_intermediate_configs(a.intermediate_configs)
    total_models = len(dims) * a.runs_per_dim
    if not 0.0 < float(a.final_val_fraction) < 1.0:
        raise ValueError("--final-val-fraction must be between 0 and 1.")

    script_dir = Path(__file__).resolve().parent
    base_mod = load_module(script_dir / 'supcon_vae.py', "supcon_base")
    pathway_mod = load_module(script_dir / 'attribution_pathways.py', "supcon_pathway")

    base_mod.set_seed(a.seed)
    pathway_mod.set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] dims={dims}, runs_per_dim={a.runs_per_dim}")
    print(f"[INFO] dropout_options={dropout_options}")
    print(f"[INFO] intermediate_options={intermediate_options}")

    df, _, pca = base_mod.prepare_drug_dataframe(root, out_dir, seed=a.seed, pca_components=a.pca_components)
    preprocessing_summary_path = out_dir / "supcon_preprocessing_summary.json"
    with open(preprocessing_summary_path, "r", encoding="utf-8") as f:
        preprocessing_summary = json.load(f)
    invalidate_stale_supcon_cache(
        out_dir=out_dir,
        expected_threshold=float(preprocessing_summary.get("threshold_log2fc", 0.15)),
        expected_dims=dims,
        expected_dropout_options=dropout_options,
        expected_intermediate_options=intermediate_options,
    )

    input_dim = int(preprocessing_summary["model_input_dim"])
    target_pca = np.stack(df["target_pca"].to_list()).astype(np.float32)
    feature_ids = [
        str(x).strip().replace("\xa0", " ")
        for x in pd.read_csv(root / "data" / "raw" / "1752_Experiments_Data_With_Replicate_for_VAE.csv", nrows=0).columns.tolist()
        if x != "Sample_ID"
    ]

    selection_path = out_dir / "supcon_hyperparameter_selection.json"
    tuning_df_path = out_dir / "supcon_hyperparameter_tuning_cv.csv"
    selection: dict[str, object] | None = None
    selected_hidden: dict[int, tuple[int, int]] | None = None
    best_dropout: float | None = None

    if selection_path.exists():
        try:
            with open(selection_path, "r", encoding="utf-8") as f:
                cached_selection = json.load(f)
            cached_dropout_options = [float(x) for x in cached_selection.get("dropout_options", [])]
            cached_intermediate_options = [
                (int(item[0]), int(item[1]))
                for item in cached_selection.get("intermediate_layer_options", [])
            ]
            cached_dims = [int(x) for x in cached_selection.get("dims", [])]
            cached_hidden = {
                int(latent_dim): (int(hidden[0]), int(hidden[1]))
                for latent_dim, hidden in cached_selection.get("selected_hidden_dims_by_latent", {}).items()
            }
            dims_match = cached_dims == dims or (
                not cached_dims and sorted(cached_hidden.keys()) == sorted(int(x) for x in dims)
            )
            dropout_match = len(cached_dropout_options) == len(dropout_options) and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(cached_dropout_options, dropout_options)
            )
            intermediate_match = cached_intermediate_options == intermediate_options
            tune_match = (
                int(cached_selection.get("tune_epochs", -1)) == int(a.tune_epochs)
                and int(cached_selection.get("tune_patience", -1)) == int(a.tune_patience)
                and int(cached_selection.get("tune_folds", -1)) == int(a.tune_folds)
                and math.isclose(
                    float(cached_selection.get("tune_min_delta", float("nan"))),
                    float(a.tune_min_delta),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if dims_match and dropout_match and intermediate_match and tune_match and cached_hidden:
                selection = cached_selection
                best_dropout = float(selection["selected_dropout_rate"])
                selected_hidden = cached_hidden
                print("[INFO] Reusing cached SupCon hyperparameter selection")
            else:
                print("[INFO] Cached SupCon hyperparameter selection does not match requested settings; retuning")
        except Exception as exc:
            print(f"[WARN] Failed to read cached SupCon hyperparameter selection: {exc}")

    if selection is None or selected_hidden is None or best_dropout is None:
        best_dropout, selected_hidden, _, selection = tune_supcon_hyperparameters(
            df=df,
            out_dir=out_dir,
            base_mod=base_mod,
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
    elif not tuning_df_path.exists():
        print("[WARN] Hyperparameter selection cache exists but tuning table is missing.")

    joined_embeddings_path = out_dir / "supcon_joined_embeddings_pca_noscale_cv.npy"
    joined_weights_path = out_dir / "supcon_joined_weights_pca_noscale_cv.npy"
    node_meta_path = out_dir / "supcon_joined_node_metadata_pca_noscale_cv.csv"
    model_summary_path = out_dir / "supcon_model_bank_summary.csv"
    training_log_bank_path = out_dir / "supcon_training_log_bank.csv"
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
                stem = cache_stem(latent_dim, run_i, hidden_dims=hidden_dims, dropout_rate=best_dropout)
                emb_cache = bank_cache_dir / f"{stem}_embeddings.npy"
                weight_cache = bank_cache_dir / f"{stem}_weights.npy"
                summary_cache = bank_cache_dir / f"{stem}_summary.json"
                log_cache = bank_cache_dir / f"{stem}_log.csv"
                print(
                    f"[BANK] model {model_idx}/{total_models}: latent_dim={latent_dim}, hidden={hidden_dims}, "
                    f"dropout={best_dropout:.2f}, run={run_i + 1}/{a.runs_per_dim}, seed={model_seed}"
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
                    final_stratify_key = build_stratify_key(df, base_mod, min_count=2)
                    train_df, val_df = train_test_split(
                        df,
                        test_size=a.final_val_fraction,
                        stratify=final_stratify_key,
                        random_state=model_seed,
                    )
                    train_df = train_df.reset_index(drop=True)
                    val_df = val_df.reset_index(drop=True)
                    model, log_df, train_info = train_supcon_model_with_validation(
                        train_df=train_df,
                        val_df=val_df,
                        base_mod=base_mod,
                        device=device,
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        dropout_rate=best_dropout,
                        batch_size=a.batch_size,
                        epochs=a.epochs,
                        patience=a.patience,
                        min_delta=a.min_delta,
                        seed=model_seed,
                        label_col="Label_Int",
                    )
                    log_df["source_latent_dim"] = latent_dim
                    log_df["source_run"] = run_i
                    log_df["hidden_1"] = int(hidden_dims[0])
                    log_df["hidden_2"] = int(hidden_dims[1])
                    log_df["dropout_rate"] = float(best_dropout)
                    log_df["validation_fraction"] = float(a.final_val_fraction)
                    log_tables.append(log_df)
                    log_df.to_csv(log_cache, index=False)

                    embeddings = base_mod.extract_embeddings(
                        model,
                        df,
                        device=device,
                        batch_size=max(128, a.batch_size),
                    )
                    gene_weights = pathway_mod.compute_gene_attributions(
                        model=model,
                        data_pca=target_pca,
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
                        "dropout_rate": float(best_dropout),
                        "validation_fraction": float(a.final_val_fraction),
                        "n_train_drugs": int(len(train_df)),
                        "n_validation_drugs": int(len(val_df)),
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
                            "hidden_1": int(hidden_dims[0]),
                            "hidden_2": int(hidden_dims[1]),
                            "dropout_rate": float(best_dropout),
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
    node_meta_df.to_csv(out_dir / "supcon_ensemble_node_labels.csv", index=False)
    pd.DataFrame(cluster_rows).sort_values("ensemble_label").to_csv(
        out_dir / "supcon_ensemble_cluster_sizes.csv", index=False
    )

    emb_df = base_mod.aggregate_embeddings(
        df=df,
        embedding_runs={idx: [ensemble_embeddings[idx]] for idx in range(len(df))},
        latent_dim=selected_dim,
    )
    emb_df["embedding_source"] = "deepprofile_style_supcon_ensemble_all_labels"
    emb_df["embedding_count"] = 1
    emb_df.to_csv(out_dir / "supcon_consensus_embedding.csv", index=False)
    emb_df.to_csv(out_dir / "supcon_consensus_embedding_full.csv", index=False)

    node_names = make_node_names(selected_dim)
    raw_weights_df = pd.DataFrame(ensemble_weights, index=node_names, columns=feature_ids)
    raw_weights_df.to_csv(out_dir / "supcon_gene_weights_raw_uniprot.csv")
    weights_df_mapped = map_gene_weights(raw_weights_df, root / "data" / "raw" / "Uniprot_ID_to_Gene_name_mapping.csv")
    weights_df_mapped.to_csv(out_dir / "supcon_gene_weights.csv")
    save_top_genes(weights_df_mapped, out_dir)
    save_pathway_results(root, out_dir, pathway_mod, weights_df_mapped, alpha=a.alpha)

    leakage_note = (
        "This DeepProfile-style SupCon ensemble is trained on the full 875-drug cohort, and all 448 labeled drugs "
        "contribute their labels during SupCon representation learning, while each source model now uses an internal "
        "training/validation split and validation reconstruction error for checkpoint selection. Downstream within-cohort "
        "classification results should therefore be treated as transductive/exploratory, not as unbiased generalization estimates."
    )
    (out_dir / "leakage_notice.txt").write_text(leakage_note, encoding="utf-8")

    summary = {
        "device": str(device),
        "dims": dims,
        "runs_per_dim": int(a.runs_per_dim),
        "total_source_models": int(total_models),
        "total_source_nodes": int(joined_embeddings.shape[1]),
        "selected_ensemble_dim": int(selected_dim),
        "gmeans_strictness": int(a.gmeans_strictness),
        "gmeans_max_depth": int(a.gmeans_max_depth),
        "pca_components": int(pca.n_components_),
        "used_standard_scaler": bool(preprocessing_summary.get("used_standard_scaler", False)),
        "used_pca": bool(preprocessing_summary.get("used_pca", False)),
        "model_input_dim": int(preprocessing_summary.get("model_input_dim", input_dim)),
        "threshold_log2fc": float(preprocessing_summary.get("threshold_log2fc", 0.15)),
        "selected_dropout_rate": float(best_dropout),
        "selected_hidden_dims_by_latent": {
            str(int(latent_dim)): [int(hidden[0]), int(hidden[1])]
            for latent_dim, hidden in selected_hidden.items()
        },
        "tuning_protocol": str(selection["protocol"]),
        "tune_folds": int(a.tune_folds),
        "tune_epochs": int(a.tune_epochs),
        "tune_patience": int(a.tune_patience),
        "tune_min_delta": float(a.tune_min_delta),
        "final_validation_fraction": float(a.final_val_fraction),
        "n_drugs_total": int(len(df)),
        "n_known": int((df["Label_Int"].isin([0, 1])).sum()),
        "n_unknown": int((df["Label_Int"] == -1).sum()),
        "ig_steps": int(a.ig_steps),
        "ig_max_samples": int(a.ig_max_samples),
        "ig_batch": int(a.ig_batch),
        "evaluation_protocol": (
            "full-cohort DeepProfile-style SupCon ensemble with 5-fold CV hyperparameter tuning, "
            "validation reconstruction error checkpoint selection, and GMeans-selected ensemble size"
        ),
        "representation_learning": "full-cohort SupCon VAE ensemble with thresholded log2FC inputs, direct PCA, and no StandardScaler",
        "leakage_notice": leakage_note,
    }
    with open(out_dir / "supcon_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
