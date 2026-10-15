#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import copy
import json
import math
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

CLASS_TO_INT = {"Non-Antineoplastic": 0, "Antineoplastic": 1, "Unknown": -1}
POSITIVE_CLASS_INT = 1
MASKED_LABEL_COL = "Label_Int_Masked"
KL_MAX_BETA = 0.1
KL_WARMUP_EPOCHS = 30
RECON_TARGET_ALPHA = 0.7
PSEUDO_CONSISTENCY_WEIGHT = 0.3
BATCH_CLASS_RATIOS = {1: 0.35, 0: 0.35, -1: 0.30}
THRESHOLD_LOG2FC = 0.15


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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_stratify(y: np.ndarray, clusters: np.ndarray, min_count: int = 2) -> np.ndarray:
    combined = np.array([f"{yi}_C{ci}" for yi, ci in zip(y, clusters)])
    counts = Counter(combined)
    if min(counts.values()) >= min_count:
        return combined

    merged = combined.copy()
    for cls in sorted(set(y.tolist())):
        cls_prefix = f"{cls}_C"
        cls_keys = {k: v for k, v in counts.items() if k.startswith(cls_prefix)}
        cls_tiny = {k: v for k, v in cls_keys.items() if v < min_count}
        cls_large = {k: v for k, v in cls_keys.items() if v >= min_count}
        if not cls_tiny:
            continue
        if cls_large:
            largest_key = max(cls_large, key=cls_large.get)
            for key in cls_tiny:
                merged[merged == key] = largest_key
        else:
            for key in cls_tiny:
                merged[merged == key] = str(cls)

    counts_merged = Counter(merged)
    if min(counts_merged.values()) < min_count:
        return y.astype(str)
    return merged


class DrugDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        x1_col = "x1_model" if "x1_model" in row.index else "x1_pca"
        x2_col = "x2_model" if "x2_model" in row.index else "x2_pca"
        target_col = "target_model" if "target_model" in row.index else "target_pca"
        return (
            torch.tensor(row[x1_col], dtype=torch.float32),
            torch.tensor(row[x2_col], dtype=torch.float32),
            torch.tensor(row[target_col], dtype=torch.float32),
            torch.tensor(int(row[self.label_col]), dtype=torch.long),
            torch.tensor(float(row["is_real"]), dtype=torch.float32),
        )


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 64,
        seed: int = 42,
        class_ratios: dict[int, float] | None = None,
    ):
        self.labels = np.asarray(labels, dtype=int)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.class_order = [-1, 0, 1]
        self.class_ratios = class_ratios or BATCH_CLASS_RATIOS
        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in self.class_order}
        if any(len(self.class_indices[cls]) == 0 for cls in self.class_order):
            missing = [cls for cls in self.class_order if len(self.class_indices[cls]) == 0]
            raise ValueError(f"BalancedBatchSampler missing class(es): {missing}")

        ratio_sum = sum(float(self.class_ratios.get(cls, 0.0)) for cls in self.class_order)
        if ratio_sum <= 0:
            raise ValueError("BalancedBatchSampler requires positive class ratios")

        raw_counts = {
            cls: self.batch_size * float(self.class_ratios.get(cls, 0.0)) / ratio_sum
            for cls in self.class_order
        }
        self.per_class = {cls: max(1, int(math.floor(raw_counts[cls]))) for cls in self.class_order}
        total = sum(self.per_class.values())
        if total > self.batch_size:
            for cls in [1, 0, -1]:
                while total > self.batch_size and self.per_class[cls] > 1:
                    self.per_class[cls] -= 1
                    total -= 1
        elif total < self.batch_size:
            ranked = sorted(self.class_order, key=lambda cls: (raw_counts[cls] - math.floor(raw_counts[cls]), cls), reverse=True)
            pos = 0
            while total < self.batch_size:
                cls = ranked[pos % len(ranked)]
                self.per_class[cls] += 1
                total += 1
                pos += 1

        minority = max(1, len(self.class_indices[1]))
        self.n_batches = max(1, int(math.ceil(minority / self.per_class[1])))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            batch: list[int] = []
            for cls in self.class_order:
                idx = self.class_indices[cls]
                size = self.per_class[cls]
                pick = rng.choice(idx, size=size, replace=len(idx) < size)
                batch.extend(pick.tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        proj_dim: int,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, input_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = self.decoder(z)
        proj = F.normalize(self.projector(mu), dim=-1)
        return recon, mu, logvar, proj


def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1))


def supcon(proj1: torch.Tensor, proj2: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    z = torch.cat([proj1, proj2], dim=0)
    y = torch.cat([labels, labels], dim=0)
    n = z.shape[0]
    sim = z @ z.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pair = torch.zeros((n, n), dtype=torch.bool, device=z.device)
    half = n // 2
    idx = torch.arange(half, device=z.device)
    pair[idx, idx + half] = True
    pair[idx + half, idx] = True
    yi, yj = y.unsqueeze(1), y.unsqueeze(0)
    known = (yi >= 0) & (yj >= 0)
    pos = (known & (yi == yj) & (~eye)) | (pair & (yi == -1) & (yj == -1))
    neg = known & (yi != yj) & (~eye)
    valid = pos | neg
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)
    denom = (exp_sim * valid.float()).sum(dim=1, keepdim=True) + 1e-12
    log_prob = sim - torch.log(denom)
    n_pos = pos.float().sum(dim=1)
    per_sample = -(log_prob * pos.float()).sum(dim=1) / n_pos.clamp(min=1.0)
    keep = n_pos > 0
    return per_sample[keep].mean() if keep.any() else torch.tensor(0.0, device=z.device)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
    }


def make_classifier(seed: int):
    try:
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                objective="binary",
                learning_rate=0.05,
                n_estimators=400,
                num_leaves=31,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            ),
            "LightGBM",
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=5,
                random_state=seed,
            ),
            "HistGradientBoosting",
        )


def prepare_drug_dataframe(root: Path, out: Path, seed: int, pca_components: int) -> tuple[pd.DataFrame, None, PCA]:
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
    x_all = data[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    x_all[np.abs(x_all) <= THRESHOLD_LOG2FC] = 0.0
    drugs = data["Drug_Name"].to_numpy()

    grouped: dict[str, list[np.ndarray]] = {}
    for i, drug in enumerate(drugs):
        grouped.setdefault(drug, []).append(x_all[i])

    diffs = [vals[0] - vals[1] for vals in grouped.values() if len(vals) >= 2]
    sigma = np.std(np.vstack(diffs), axis=0) / math.sqrt(2.0) if diffs else np.full(x_all.shape[1], 1e-3)
    sigma = np.clip(sigma, 1e-4, None).astype(np.float32)

    rng = np.random.default_rng(seed)
    rows = []
    for drug, reps in grouped.items():
        if drug not in dmeta.index:
            continue
        drug_class = str(dmeta.at[drug, "Drug_Class"])
        if rep_count_1628.get(drug, 0) >= 2 and len(reps) >= 2:
            x1, x2 = reps[0], reps[1]
            is_real = 1
            target = (x1 + x2) / 2.0
            n_rep = 2
        else:
            x1 = reps[0]
            x2 = x1 + rng.normal(0.0, sigma * math.sqrt(2.0), size=x1.shape).astype(np.float32)
            is_real = 0
            target = x1
            n_rep = 1
        rows.append(
            {
                "Drug_Name": drug,
                "Drug_Class": drug_class,
                "Label_Int": CLASS_TO_INT[drug_class],
                "Cluster": int(dmeta.at[drug, "Cluster"]),
                "n_replicates_raw": int(n_rep),
                "is_real": int(is_real),
                "x1": x1.astype(np.float32),
                "x2": x2.astype(np.float32),
                "target": target.astype(np.float32),
            }
        )

    df = pd.DataFrame(rows).sort_values("Drug_Name").reset_index(drop=True)
    pd.DataFrame(
        {
            "Drug_Name": sorted(grouped.keys()),
            "n_replicates": [len(grouped[k]) for k in sorted(grouped.keys())],
        }
    ).to_csv(out / "replicate_count_from_1752.csv", index=False)

    all_views = np.vstack([np.vstack([row["x1"], row["x2"]]) for _, row in df.iterrows()])
    n_comp = min(pca_components, all_views.shape[0], all_views.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(all_views)

    for col in ["x1", "x2", "target"]:
        df[f"{col}_pca"] = df[col].map(
            lambda v: pca.transform(v.reshape(1, -1)).squeeze(0).astype(np.float32)
        )
        df[f"{col}_model"] = df[f"{col}_pca"]

    with open(out / "supcon_scaler.pkl", "wb") as f:
        pickle.dump(None, f)
    with open(out / "supcon_pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out / "supcon_preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "used_standard_scaler": False,
                "used_pca": True,
                "pca_components_requested": int(pca_components),
                "pca_components_actual": int(pca.n_components_),
                "model_input_dim": int(pca.n_components_),
                "n_samples_supcon": int(len(df)),
                "n_features_proteins": int(len(feats)),
                "threshold_log2fc": float(THRESHOLD_LOG2FC),
                "pca_reference": "replicate-aware direct PCA",
                "pca_reference_note": (
                    "Replicate-aware views were thresholded at |log2FC| > 0.15 and then projected "
                    "with PCA directly on log2FC values without StandardScaler."
                ),
            },
            f,
            indent=2,
        )

    return df, None, pca


def train_supcon_model(
    df: pd.DataFrame,
    device: torch.device,
    input_dim: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    label_col: str,
) -> tuple[Model, pd.DataFrame, dict[str, float | int]]:
    set_seed(seed)
    model = Model(input_dim=input_dim, latent_dim=latent_dim, proj_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    sampler = BalancedBatchSampler(
        df[label_col].to_numpy(dtype=int),
        batch_size=batch_size,
        seed=seed,
        class_ratios=BATCH_CLASS_RATIOS,
    )
    loader = DataLoader(DrugDataset(df, label_col=label_col), batch_sampler=sampler)

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    logs: list[dict[str, float | int]] = []

    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        model.train()
        loss_sum = Counter()
        beta = min(KL_MAX_BETA, KL_MAX_BETA * ep / max(1, KL_WARMUP_EPOCHS))

        for x1, x2, target, labels, is_real in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)
            labels = labels.to(device)
            is_real = is_real.to(device)

            recon1, mu1, logvar1, proj1 = model(x1)
            recon2, mu2, logvar2, proj2 = model(x2)

            recon1_to_target = F.mse_loss(recon1, target, reduction="none").mean(dim=1)
            recon1_to_self = F.mse_loss(recon1, x1, reduction="none").mean(dim=1)
            recon2_to_target = F.mse_loss(recon2, target, reduction="none").mean(dim=1)
            recon2_to_self = F.mse_loss(recon2, x2, reduction="none").mean(dim=1)
            loss_recon_1 = RECON_TARGET_ALPHA * recon1_to_target + (1.0 - RECON_TARGET_ALPHA) * recon1_to_self
            loss_recon_2 = RECON_TARGET_ALPHA * recon2_to_target + (1.0 - RECON_TARGET_ALPHA) * recon2_to_self
            loss_recon = 0.5 * (loss_recon_1.mean() + loss_recon_2.mean())

            loss_kl = 0.5 * (kl(mu1, logvar1) + kl(mu2, logvar2))
            cons_weight = torch.where(
                is_real > 0.5,
                torch.ones_like(is_real),
                torch.full_like(is_real, PSEUDO_CONSISTENCY_WEIGHT),
            )
            loss_cons_per_sample = ((mu1 - mu2) ** 2).mean(dim=1)
            loss_cons = (loss_cons_per_sample * cons_weight).mean()
            loss_supcon = supcon(proj1, proj2, labels)
            loss = loss_recon + beta * loss_kl + 0.25 * loss_cons + 0.35 * loss_supcon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum["n"] += 1
            loss_sum["loss_total"] += float(loss.item())
            loss_sum["loss_recon"] += float(loss_recon.item())
            loss_sum["loss_kl"] += float(loss_kl.item())
            loss_sum["loss_cons"] += float(loss_cons.item())
            loss_sum["loss_supcon"] += float(loss_supcon.item())

        scheduler.step()
        epoch_log = {
            "epoch": ep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "kl_beta": float(beta),
            "loss_total": loss_sum["loss_total"] / loss_sum["n"],
            "loss_recon": loss_sum["loss_recon"] / loss_sum["n"],
            "loss_kl": loss_sum["loss_kl"] / loss_sum["n"],
            "loss_cons": loss_sum["loss_cons"] / loss_sum["n"],
            "loss_supcon": loss_sum["loss_supcon"] / loss_sum["n"],
        }
        logs.append(epoch_log)

        if epoch_log["loss_total"] < best_loss - min_delta:
            best_loss = float(epoch_log["loss_total"])
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TRAIN] early stopping at epoch {ep} (best epoch {best_epoch}, best loss {best_loss:.6f})")
                break

    model.load_state_dict(best_state)
    return (
        model,
        pd.DataFrame(logs),
        {
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "n_epochs_trained": int(len(logs)),
        },
    )


def extract_embeddings(model: Model, df: pd.DataFrame, device: torch.device, batch_size: int = 256) -> np.ndarray:
    x1_col = "x1_model" if "x1_model" in df.columns else "x1_pca"
    x2_col = "x2_model" if "x2_model" in df.columns else "x2_pca"
    x1 = np.stack(df[x1_col].to_list()).astype(np.float32)
    x2 = np.stack(df[x2_col].to_list()).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.FloatTensor(x1), torch.FloatTensor(x2)), batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x1, batch_x2 in loader:
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            mu1, _ = model.encode(batch_x1)
            mu2, _ = model.encode(batch_x2)
            z = ((mu1 + mu2) / 2.0).cpu().numpy()
            embeddings.append(z)

    return np.vstack(embeddings)


def aggregate_embeddings(
    df: pd.DataFrame,
    embedding_runs: dict[int, list[np.ndarray]],
    latent_dim: int,
) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        vals = embedding_runs.get(idx, [])
        if len(vals) == 0:
            raise ValueError(f"No embedding collected for row index {idx} ({row['Drug_Name']})")
        z = np.mean(np.vstack(vals), axis=0)
        source = "oof_known" if int(row["Label_Int"]) in (0, 1) else "avg_unknown"
        rows.append(
            {
                "Drug_Name": row["Drug_Name"],
                "Drug_Class": row["Drug_Class"],
                "Label_Int": int(row["Label_Int"]),
                "Cluster": int(row["Cluster"]),
                "n_replicates_raw": int(row["n_replicates_raw"]),
                "is_real_replicate_pair": int(row["is_real"]),
                "embedding_count": int(len(vals)),
                "embedding_source": source,
                **{f"latent_{i:03d}": float(v) for i, v in enumerate(z[:latent_dim])},
            }
        )
    return pd.DataFrame(rows)


def plot_training(log_df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(6.6, 4.1))
    plt.plot(log_df["epoch"], log_df["loss_total"], label="Total")
    plt.plot(log_df["epoch"], log_df["loss_recon"], label="Recon")
    plt.plot(log_df["epoch"], log_df["loss_cons"], label="Consistency")
    plt.plot(log_df["epoch"], log_df["loss_supcon"], label="SupCon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SupCon-Consensus-VAE training (final deployment model)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out / "figure_supcon_training_loss.png", dpi=220)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, prob: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("SupCon latent ROC (honest OOF)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out / "figure_supcon_roc.png", dpi=220)
    plt.close()


def plot_importance(imp_df: pd.DataFrame, out: Path) -> None:
    top = imp_df.head(20).copy()
    plt.figure(figsize=(6.8, 4.2))
    plt.bar(top["latent_idx"].astype(str), top["importance"], color="#1f78b4")
    plt.xticks(rotation=65)
    plt.xlabel("Latent index")
    plt.ylabel("Mean importance")
    plt.title("Top latent importance across honest outer folds")
    plt.tight_layout()
    plt.savefig(out / "figure_supcon_top_latent_importance.png", dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ARTICLE_ROOT))
    ap.add_argument("--results-subdir", default="results")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--pca-components", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    a = ap.parse_args()

    root = Path(a.root)
    out = root / "results" / "supcon"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df, scaler, pca = prepare_drug_dataframe(root, out, seed=a.seed, pca_components=a.pca_components)
    input_dim = int(pca.n_components_)
    print(
        f"[INFO] drugs={len(df)} real_pairs={int(df['is_real'].sum())} "
        f"pseudo_pairs={int((df['is_real'] == 0).sum())} pca_components={input_dim}"
    )

    known_idx = df.index[df["Label_Int"].isin([0, 1])].to_numpy(dtype=int)
    unknown_idx = df.index[df["Label_Int"] == -1].to_numpy(dtype=int)
    y_known = df.loc[known_idx, "Label_Int"].to_numpy(dtype=int)
    c_known = df.loc[known_idx, "Cluster"].to_numpy(dtype=int)

    embedding_runs: dict[int, list[np.ndarray]] = defaultdict(list)
    known_prob_runs: dict[int, list[float]] = defaultdict(list)
    unknown_prob_runs: dict[int, list[float]] = defaultdict(list)
    importance_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []
    repeat_rows: list[dict[str, float | int | str]] = []
    outer_logs: list[pd.DataFrame] = []
    classifier_name = ""

    for repeat in range(a.n_repeats):
        split_seed = a.seed + repeat
        stratify_key = build_stratify(y_known, c_known, min_count=a.outer_folds)
        skf = StratifiedKFold(n_splits=a.outer_folds, shuffle=True, random_state=split_seed)
        repeat_pred_rows = []

        for fold, (tr_in_known, te_in_known) in enumerate(skf.split(known_idx, stratify_key), start=1):
            train_known_idx = known_idx[tr_in_known]
            test_known_idx = known_idx[te_in_known]

            df_fold = df.copy()
            masked = df_fold["Label_Int"].to_numpy(dtype=int).copy()
            masked[test_known_idx] = -1
            df_fold[MASKED_LABEL_COL] = masked

            fold_seed = a.seed + repeat * 100 + fold
            model_fold, log_fold, train_info = train_supcon_model(
                df=df_fold,
                device=device,
                input_dim=input_dim,
                latent_dim=a.latent_dim,
                batch_size=a.batch_size,
                epochs=a.epochs,
                patience=a.patience,
                min_delta=a.min_delta,
                seed=fold_seed,
                label_col=MASKED_LABEL_COL,
            )
            log_fold["repeat"] = repeat
            log_fold["fold"] = fold
            outer_logs.append(log_fold)

            emb_fold = extract_embeddings(model_fold, df_fold, device=device, batch_size=max(128, a.batch_size))
            for idx in test_known_idx:
                embedding_runs[int(idx)].append(emb_fold[int(idx)])
            for idx in unknown_idx:
                embedding_runs[int(idx)].append(emb_fold[int(idx)])

            clf, classifier_name = make_classifier(fold_seed)
            clf.fit(emb_fold[train_known_idx], df.loc[train_known_idx, "Label_Int"].to_numpy(dtype=int))
            prob_test = clf.predict_proba(emb_fold[test_known_idx])[:, 1]
            pred_test = (prob_test >= 0.5).astype(int)

            test_true = df.loc[test_known_idx, "Label_Int"].to_numpy(dtype=int)
            fold_metric = {
                "repeat": int(repeat),
                "fold": int(fold),
                "n_train_known": int(len(train_known_idx)),
                "n_test_known": int(len(test_known_idx)),
                "best_train_loss": float(train_info["best_loss"]),
                "best_epoch": int(train_info["best_epoch"]),
                "n_epochs_trained": int(train_info["n_epochs_trained"]),
                **binary_metrics(test_true, pred_test, prob_test),
            }
            fold_rows.append(fold_metric)
            print(
                f"[OUTER] repeat={repeat + 1}/{a.n_repeats} fold={fold}/{a.outer_folds} "
                f"acc={fold_metric['accuracy']:.4f} auroc={fold_metric['auroc']:.4f} "
                f"best_epoch={fold_metric['best_epoch']}"
            )

            repeat_pred_rows.append(
                pd.DataFrame(
                    {
                        "row_idx": test_known_idx,
                        "repeat": repeat,
                        "fold": fold,
                        "Drug_Name": df.loc[test_known_idx, "Drug_Name"].to_numpy(),
                        "Drug_Class": df.loc[test_known_idx, "Drug_Class"].to_numpy(),
                        "Cluster": df.loc[test_known_idx, "Cluster"].to_numpy(),
                        "y_true": test_true,
                        "y_pred": pred_test,
                        "prob_antineoplastic": prob_test,
                    }
                )
            )
            for idx, prob_val in zip(test_known_idx, prob_test):
                known_prob_runs[int(idx)].append(float(prob_val))

            if len(unknown_idx) > 0:
                prob_unknown = clf.predict_proba(emb_fold[unknown_idx])[:, 1]
                for idx, prob_val in zip(unknown_idx, prob_unknown):
                    unknown_prob_runs[int(idx)].append(float(prob_val))

            imp = np.asarray(getattr(clf, "feature_importances_", np.zeros(a.latent_dim)), dtype=float)
            importance_rows.append(
                pd.DataFrame(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "latent_idx": np.arange(len(imp), dtype=int),
                        "importance": imp,
                    }
                )
            )

        repeat_pred_df = pd.concat(repeat_pred_rows, ignore_index=True).sort_values("row_idx")
        repeat_metrics = binary_metrics(
            repeat_pred_df["y_true"].to_numpy(dtype=int),
            repeat_pred_df["y_pred"].to_numpy(dtype=int),
            repeat_pred_df["prob_antineoplastic"].to_numpy(dtype=float),
        )
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "model": classifier_name,
                "n_known": int(len(repeat_pred_df)),
                **repeat_metrics,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / "supcon_binary_outer_fold_metrics.csv", index=False)
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out / "supcon_binary_repeated_metrics.csv", index=False)
    if outer_logs:
        pd.concat(outer_logs, ignore_index=True).to_csv(out / "supcon_outer_training_log.csv", index=False)

    known_pred_rows = []
    for idx in known_idx:
        probs = np.array(known_prob_runs[int(idx)], dtype=float)
        prob_mean = float(probs.mean())
        prob_std = float(probs.std(ddof=0))
        y_true = int(df.at[int(idx), "Label_Int"])
        known_pred_rows.append(
            {
                "row_idx": int(idx),
                "Drug_Name": df.at[int(idx), "Drug_Name"],
                "Drug_Class": df.at[int(idx), "Drug_Class"],
                "Cluster": int(df.at[int(idx), "Cluster"]),
                "y_true": y_true,
                "y_pred": int(prob_mean >= 0.5),
                "prob_antineoplastic": prob_mean,
                "prob_antineoplastic_std": prob_std,
                "n_oof_predictions": int(len(probs)),
            }
        )
    known_pred_df = pd.DataFrame(known_pred_rows).sort_values("Drug_Name").reset_index(drop=True)
    known_pred_df.to_csv(out / "supcon_binary_test_predictions.csv", index=False)

    honest_metrics = {
        "model": classifier_name,
        "evaluation_protocol": "outer_cv_oof_label_masked_supcon",
        "outer_folds": int(a.outer_folds),
        "n_repeats": int(a.n_repeats),
        "n_known": int(len(known_pred_df)),
        **binary_metrics(
            known_pred_df["y_true"].to_numpy(dtype=int),
            known_pred_df["y_pred"].to_numpy(dtype=int),
            known_pred_df["prob_antineoplastic"].to_numpy(dtype=float),
        ),
    }
    with open(out / "supcon_binary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(honest_metrics, f, indent=2)

    repeated_summary: dict[str, float | int | str] = {
        "model": classifier_name,
        "evaluation_protocol": "outer_cv_oof_label_masked_supcon",
        "outer_folds": int(a.outer_folds),
        "n_repeats": int(len(repeat_df)),
    }
    for key in ["accuracy", "precision_pos", "recall_pos", "f1_pos", "mcc", "auroc", "auprc"]:
        repeated_summary[f"{key}_mean"] = float(repeat_df[key].mean())
        repeated_summary[f"{key}_std"] = float(repeat_df[key].std(ddof=0))
    with open(out / "supcon_binary_repeated_summary.json", "w", encoding="utf-8") as f:
        json.dump(repeated_summary, f, indent=2)

    if len(unknown_idx) > 0:
        unknown_rows = []
        for idx in unknown_idx:
            probs = np.array(unknown_prob_runs[int(idx)], dtype=float)
            prob_mean = float(probs.mean())
            prob_std = float(probs.std(ddof=0))
            unknown_rows.append(
                {
                    "row_idx": int(idx),
                    "Drug_Name": df.at[int(idx), "Drug_Name"],
                    "Drug_Class": df.at[int(idx), "Drug_Class"],
                    "Cluster": int(df.at[int(idx), "Cluster"]),
                    "prob_antineoplastic": prob_mean,
                    "prob_antineoplastic_std": prob_std,
                    "predicted_class": "Antineoplastic" if prob_mean >= 0.5 else "Non-Antineoplastic",
                    "n_probability_estimates": int(len(probs)),
                }
            )
        unk_df = pd.DataFrame(unknown_rows).sort_values("prob_antineoplastic", ascending=False).reset_index(drop=True)
        unk_df.to_csv(out / "supcon_unknown_predictions.csv", index=False)

    if importance_rows:
        imp_long = pd.concat(importance_rows, ignore_index=True)
        imp_df = (
            imp_long.groupby("latent_idx")["importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "importance", "std": "importance_std", "count": "n_models"})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df["importance_std"] = imp_df["importance_std"].fillna(0.0)
    else:
        imp_df = pd.DataFrame(
            {
                "latent_idx": np.arange(a.latent_dim),
                "importance": np.zeros(a.latent_dim, dtype=float),
                "importance_std": np.zeros(a.latent_dim, dtype=float),
                "n_models": np.zeros(a.latent_dim, dtype=int),
            }
        )
    imp_df.to_csv(out / "supcon_latent_importance.csv", index=False)

    full_model, full_log_df, deploy_info = train_supcon_model(
        df=df,
        device=device,
        input_dim=input_dim,
        latent_dim=a.latent_dim,
        batch_size=a.batch_size,
        epochs=a.epochs,
        patience=a.patience,
        min_delta=a.min_delta,
        seed=a.seed + 9999,
        label_col="Label_Int",
    )
    torch.save(full_model.state_dict(), out / "supcon_consensus_vae.pt")
    full_log_df.to_csv(out / "supcon_training_log.csv", index=False)

    full_embeddings = extract_embeddings(full_model, df, device=device, batch_size=max(128, a.batch_size))
    full_embedding_df = aggregate_embeddings(
        df=df,
        embedding_runs={idx: [full_embeddings[idx]] for idx in range(len(df))},
        latent_dim=a.latent_dim,
    )
    full_embedding_df["embedding_source"] = "full_model"
    full_embedding_df["embedding_count"] = 1
    full_embedding_df.to_csv(out / "supcon_consensus_embedding_full.csv", index=False)

    for idx in known_idx:
        if len(embedding_runs[int(idx)]) == 0:
            embedding_runs[int(idx)].append(full_embeddings[int(idx)])
    for idx in unknown_idx:
        if len(embedding_runs[int(idx)]) == 0:
            embedding_runs[int(idx)].append(full_embeddings[int(idx)])

    emb_df = aggregate_embeddings(df=df, embedding_runs=embedding_runs, latent_dim=a.latent_dim)
    emb_df.to_csv(out / "supcon_consensus_embedding.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(root / 'src/deepheal/models/attribution_pathways.py'),
            "--root",
            str(root),
            "--results-subdir",
            str(a.results_subdir),
            "--seed",
            str(a.seed),
        ],
        check=True,
    )

    plot_training(full_log_df, out)
    plot_roc_curve(
        known_pred_df["y_true"].to_numpy(dtype=int),
        known_pred_df["prob_antineoplastic"].to_numpy(dtype=float),
        out,
    )
    plot_importance(imp_df, out)

    summary = {
        "device": str(device),
        "n_drugs_total": int(len(df)),
        "n_known": int((df["Label_Int"].isin([0, 1])).sum()),
        "n_unknown": int((df["Label_Int"] == -1).sum()),
        "n_real_replicate_pairs": int(df["is_real"].sum()),
        "n_pseudo_replicate_pairs": int((df["is_real"] == 0).sum()),
        "pca_components": int(pca.n_components_),
        "latent_dim": int(a.latent_dim),
        "epochs_max": int(a.epochs),
        "patience": int(a.patience),
        "kl_max_beta": float(KL_MAX_BETA),
        "kl_warmup_epochs": int(KL_WARMUP_EPOCHS),
        "recon_target_alpha": float(RECON_TARGET_ALPHA),
        "pseudo_consistency_weight": float(PSEUDO_CONSISTENCY_WEIGHT),
        "batch_class_ratios": {str(k): float(v) for k, v in BATCH_CLASS_RATIOS.items()},
        "outer_folds": int(a.outer_folds),
        "n_repeats": int(a.n_repeats),
        "evaluation_protocol": "Outer CV with test-drug label masking in SupCon; final full-label model reserved for deployment/pathway analysis",
        "deploy_model_best_epoch": int(deploy_info["best_epoch"]),
        "deploy_model_epochs_trained": int(deploy_info["n_epochs_trained"]),
        "deploy_model_best_loss": float(deploy_info["best_loss"]),
        "metrics_single_split": honest_metrics,
        "metrics_repeated_summary": repeated_summary,
    }
    with open(out / "supcon_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
