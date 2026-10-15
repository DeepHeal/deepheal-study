from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT

from pathlib import Path
from collections import Counter
import pickle
import re
import numpy as np
import pandas as pd
import torch
from deepheal.paths import ARTICLE_ROOT
SOURCE_MODEL_CKPT_DIR = ARTICLE_ROOT / "results/source_model_checkpoints"
SUPCON_DIR = ARTICLE_ROOT / "results/supcon"
AVG_DIR = ARTICLE_ROOT / "results/average"
SUPCON_BASE_SCRIPT = ARTICLE_ROOT / "src/deepheal/models/supcon_vae.py"
AVERAGE_ENSEMBLE_SCRIPT = ARTICLE_ROOT / "src/deepheal/models/average_ensemble.py"

def build_external_branch_frames(aligned_df: pd.DataFrame, supcon_pca, average_pca) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in aligned_df.columns if col not in {"Compound", "compound_norm"}]
    raw = aligned_df[feature_cols].to_numpy(dtype=np.float32)
    supcon_proj = supcon_pca.transform(raw).astype(np.float32)
    average_proj = average_pca.transform(raw).astype(np.float32)

    supcon_rows: list[dict[str, object]] = []
    average_rows: list[dict[str, object]] = []
    for idx, row in aligned_df.iterrows():
        compound = str(row["Compound"])
        compound_norm = str(row["compound_norm"])
        x_raw = raw[idx]
        supcon_rows.append(
            {
                "Compound": compound,
                "compound_norm": compound_norm,
                "drug_norm": compound_norm,
                "Drug_Name": compound,
                "x1": x_raw,
                "x2": x_raw,
                "target": x_raw,
                "x1_model": supcon_proj[idx],
                "x2_model": supcon_proj[idx],
                "target_model": supcon_proj[idx],
                "is_real": 1,
            }
        )
        average_rows.append(
            {
                "Compound": compound,
                "compound_norm": compound_norm,
                "drug_norm": compound_norm,
                "Drug_Name": compound,
                "Sample_ID": compound,
                "x": x_raw,
                "x_model": average_proj[idx],
                "x_pca": average_proj[idx],
            }
        )
    return pd.DataFrame(supcon_rows), pd.DataFrame(average_rows)

def aggregate_from_checkpoints(
    branch: str,
    query_df: pd.DataFrame,
    ckpt_dir: Path,
    node_label_path: Path,
    helper_mod,
    supcon_base,
    average_mod,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    manifest = pd.read_csv(SOURCE_MODEL_CKPT_DIR / "checkpoint_manifest.csv")
    manifest = manifest[manifest["branch"] == branch].sort_values(["source_latent_dim", "source_run"]).reset_index(drop=True)
    node_labels_df = pd.read_csv(node_label_path).sort_values("global_node_idx").reset_index(drop=True)
    ensemble_dim = int(node_labels_df["ensemble_label"].max()) + 1
    latent_cols = [f"latent_{idx:03d}" for idx in range(ensemble_dim)]
    query_sum = np.zeros((len(query_df), ensemble_dim), dtype=np.float32)
    counts = np.zeros(ensemble_dim, dtype=np.float32)

    for idx, row in manifest.iterrows():
        ckpt_path = Path(row["checkpoint_path"])
        if not ckpt_path.exists():
            relocated = ckpt_dir / ckpt_path.name
            if relocated.exists():
                ckpt_path = relocated
            else:
                raise FileNotFoundError(
                    f"Checkpoint listed in manifest is missing and could not be relocated: {row['checkpoint_path']}"
                )
        payload = torch.load(ckpt_path, map_location="cpu")
        summary = payload["source_summary"]
        latent_dim = int(summary["source_latent_dim"])
        run_i = int(summary["source_run"])
        hidden_dims = (int(summary["hidden_1"]), int(summary["hidden_2"]))
        dropout_rate = float(summary["dropout_rate"])
        input_dim = int(payload["input_dim"])

        if branch == "supcon":
            model = supcon_base.Model(
                input_dim=input_dim,
                latent_dim=latent_dim,
                proj_dim=32,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            embeddings = supcon_base.extract_embeddings(model, query_df, device=device, batch_size=max(128, batch_size))
        else:
            model = average_mod.AverageVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            embeddings = average_mod.extract_embeddings(model, query_df, device=device, batch_size=max(128, batch_size))

        node_map = (
            node_labels_df.loc[
                (node_labels_df["source_latent_dim"] == latent_dim) & (node_labels_df["source_run"] == run_i)
            ]
            .sort_values("source_node_idx")["ensemble_label"]
            .to_numpy(dtype=int)
        )
        for source_node_idx, ensemble_label in enumerate(node_map.tolist()):
            query_sum[:, ensemble_label] += embeddings[:, source_node_idx]
            counts[ensemble_label] += 1.0

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (idx + 1) % 25 == 0 or (idx + 1) == len(manifest):
            print(f"[{branch}] checkpoint {idx + 1}/{len(manifest)} projected")

    query_consensus = query_sum / np.where(counts > 0, counts, 1.0)
    out_df = pd.concat(
        [
            query_df[["Compound", "compound_norm", "drug_norm"]].reset_index(drop=True),
            pd.DataFrame(query_consensus, columns=latent_cols),
        ],
        axis=1,
    )
    return out_df
