#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


ROOT = ARTICLE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepheal_paths import CANONICAL_GMT_PATH

OUT_DIR = ROOT / "results" / "pan_drug_pathway"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
GMT_PATH = CANONICAL_GMT_PATH
AVG_DATA_PATH = ROOT / "data" / "raw" / "875_Compounds_Average_Data.csv"
META_PATH = ROOT / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv"
SUPCON_FDR = ROOT / "results" / "supcon" / "supcon_fdr_matrix.csv"
AVERAGE_FDR = ROOT / "results" / "average" / "average_fdr_matrix.csv"

TOP_N = 24
MIN_GENES = 8
MAX_GENES = 350
ALPHA = 0.05
MIN_FIGURE_SIG_CLUSTERS = 2
MAX_JACCARD_WITHIN_GROUP = 0.72

DB_PREFIXES = ("KEGG_", "BIOCARTA_", "REACTOME_")

GROUP_KEYWORDS = [
    ("DNA damage / repair", ["DNA", "REPAIR", "FANCONI", "MISMATCH", "BASE EXCISION", "NUCLEOTIDE EXCISION", "HOMOLOG", "NONHOMOLOG", "DOUBLE STRAND", "ATR", "ATM", "P53", "TELOMERE"]),
    ("Cell cycle / mitosis", ["CELL CYCLE", "MITOT", "MITOSIS", "SPINDLE", "KINETOCHORE", "ANAPHASE", "CYTOKINESIS", "CYCLIN", "G2 M", "G1 S", "M G1", "G2 G2 M", "G1 G1 S", "AURORA", "PLK1", "CENTROSOME", "PROMETAPHASE", "CDC20", "NEK2", "NUMA", "APC/C"]),
    ("Replication stress / checkpoint", ["REPLICATION STRESS", "S PHASE", "ORIGIN", "FORK", "CHK", "CHECKPOINT", "BRCA", "E2F", "CDC6", "CDT1", "ORC", "PRE REPLICATIVE"]),
    ("Apoptosis / death signaling", ["APOPT", "DEATH", "CASP", "TRAIL", "FAS", "TNFR", "BCL", "SURVIV"]),
    ("Proteostasis / ER stress / UPR", ["UNFOLDED", "PROTEASOM", "ENDOPLASMIC", "ER ", "UPR", "AUTOPHAG", "LYSOSOM", "HEAT SHOCK", "CHAPERONE", "XBP1", "GLYCAN", "CALNEXIN", "CALRETICULIN"]),
    ("Metabolism / nucleotide / redox", ["GLYCOL", "TCA", "MITOCH", "OXID", "FOLATE", "PURINE", "PYRIMID", "NUCLEOTIDE", "REDOX", "PENTOSE", "AMINO ACID", "FATTY ACID", "CHOLESTEROL", "PHOSPHOLIPID", "SPHINGOLIPID", "GLUTATHIONE", "LIPID", "METABOL"]),
    ("Inflammatory / cytokine signaling", ["INTERFERON", "INTERLEUKIN", "IL ", "IL-", "CYTOKINE", "CHEMOKINE", "JAK", "STAT", "NFKB", "TGF", "INFLAMM"]),
    ("Chromatin / RNA / translation", ["RNA", "MRNA", "RIBOSOM", "TRANSLAT", "SPLIC", "CHROMAT", "HISTONE", "PRC2", "EZH2", "TRANSCRIPTION", "POST TRANSLATIONAL", "MICRORNA"]),
]

FIGURE_GROUPS = [
    "DNA damage / repair",
    "Cell cycle / mitosis",
    "Replication stress / checkpoint",
    "Apoptosis / death signaling",
    "Proteostasis / ER stress / UPR",
    "Metabolism / nucleotide / redox",
    "Chromatin / RNA / translation",
]

FIGURE_GROUP_KEYWORDS = {
    "DNA damage / repair": ["DNA", "REPAIR", "REPLICATION", "MISMATCH", "FANCONI", "HOMOLOG", "DOUBLE STRAND", "ATR", "P53", "TELOMERE"],
    "Cell cycle / mitosis": ["CELL CYCLE", "MITOT", "MITOSIS", "CENTROSOME", "PROMETAPHASE", "ANAPHASE", "SPINDLE", "G2", "G1", "CYCLIN", "CDC20", "APC/C", "NUMA", "NEK2"],
    "Replication stress / checkpoint": ["S PHASE", "CDC6", "CDT1", "ORC", "E2F", "PRE REPLICATIVE", "REPLICATION STRESS", "ATR", "RB1"],
    "Apoptosis / death signaling": ["APOPT", "DEATH", "CASP", "TRAIL", "FAS", "TNFR", "BCL", "SURVIV"],
    "Proteostasis / ER stress / UPR": ["UNFOLDED", "XBP1", "CHAPERONE", "PROTEAS", "LYSOSOM", "AUTOPHAG", "GLYCAN", "CALNEXIN", "CALRETICULIN", "UPR"],
    "Metabolism / nucleotide / redox": ["GLYCOL", "TCA", "CITRATE", "PURINE", "PYRIMIDINE", "NUCLEOTIDE", "GLUTATHIONE", "FATTY ACID", "PHOSPHOLIPID", "GLYCEROPHOSPHOLIPID", "SPHINGOLIPID", "OXIDATION", "MITOCHONDRIAL", "LIPID"],
    "Chromatin / RNA / translation": ["RNA", "MRNA", "RIBOSOM", "TRANSLAT", "TRANSCRIPTION", "RNA POL", "POST TRANSLATIONAL", "MICRORNA", "REGULATORY RNA"],
}

FIGURE_GROUP_QUOTAS = {
    "DNA damage / repair": 4,
    "Cell cycle / mitosis": 4,
    "Replication stress / checkpoint": 3,
    "Apoptosis / death signaling": 3,
    "Proteostasis / ER stress / UPR": 3,
    "Metabolism / nucleotide / redox": 5,
    "Chromatin / RNA / translation": 2,
}

FIGURE_BLACKLIST = [
    "PLATELET",
    "VIRAL",
    "INFLUENZA",
    "HIV",
    "HOST",
    "NEP_NS2",
    "ANTIVIRAL",
    "RIG_I",
    "MDA5",
    "HELICOBACTER",
    "NEUROTRANSMITTER",
    "GLUTAMATE",
    "DOPAMINE",
    "OLFACTORY",
    "ODORANT",
    "PHOTOTRANSDUCTION",
    "ALLOGRAFT",
    "ASTHMA",
    "LEISHMANIA",
    "MALARIA",
    "TRYPANOSOMA",
    "LIMONENE",
    "PINENE",
    "BILE ACID",
    "DIABETES",
    "NATURAL KILLER",
]

GROUP_COLORS = {
    "DNA damage / repair": "#d73027",
    "Cell cycle / mitosis": "#c51b8a",
    "Replication stress / checkpoint": "#7b3294",
    "Apoptosis / death signaling": "#fc8d59",
    "Proteostasis / ER stress / UPR": "#2b8cbe",
    "Metabolism / nucleotide / redox": "#1b9e77",
    "Inflammatory / cytokine signaling": "#4daf4a",
    "Chromatin / RNA / translation": "#7570b3",
    "Other": "#8c8c8c",
}


def norm_text(value: object) -> str:
    return str(value).strip().replace("\xa0", " ")


def pretty_pathway_name(pathway: str) -> str:
    raw = str(pathway)
    prefix = "Pathway"
    for db in DB_PREFIXES:
        if raw.startswith(db):
            prefix = db[:-1].title()
            raw = raw[len(db):]
            break
    clean = raw.replace("_", " ").strip()
    clean = clean.replace("TCA CYCLE", "TCA cycle")
    clean = clean.replace("DNA ", "DNA ")
    return f"{prefix}: {clean.title()}"


def wrap_label(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def wrap_group_label(text: str) -> str:
    return wrap_label(str(text).replace("/", "/ "), 18)


def set_vertical_colorbar_title(heatmap_ax: plt.Axes, title: str, width: int = 12) -> None:
    if not heatmap_ax.collections:
        return
    cbar = heatmap_ax.collections[0].colorbar
    if cbar is None:
        return
    cbar.set_label("")
    cbar.ax.set_ylabel("")
    cbar.ax.set_title(wrap_label(title, width), fontsize=7.4, pad=6)
    cbar.ax.tick_params(labelsize=7)


def pathway_text(pathway: str) -> str:
    return str(pathway).replace("_", " ").upper()


def pathway_group(pathway: str) -> str:
    text = pathway_text(pathway)
    for group, keywords in GROUP_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return group
    return "Other"


def pathway_blacklisted(pathway: str) -> bool:
    text = pathway_text(pathway)
    return any(keyword in text for keyword in FIGURE_BLACKLIST)


def figure_keyword_match(pathway: str, group: str) -> bool:
    text = pathway_text(pathway)
    keywords = FIGURE_GROUP_KEYWORDS.get(group, [])
    return any(keyword in text for keyword in keywords)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_canonical_pathways() -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    with GMT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            if not name.startswith(DB_PREFIXES):
                continue
            genes = [norm_text(g) for g in parts[2:] if norm_text(g)]
            pathways[name] = genes
    return pathways


def load_gene_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = pd.read_csv(AVG_DATA_PATH, encoding="latin-1")
    avg.columns = [norm_text(c) for c in avg.columns]
    avg["Gene Name"] = avg["Gene Name"].map(norm_text)

    meta = pd.read_csv(META_PATH)
    meta["Sample_ID"] = meta["Sample_ID"].map(norm_text)
    meta["Drug_Name"] = meta["Drug_Name"].map(norm_text)
    meta["Drug_Class"] = meta["Drug_Class"].map(norm_text)

    sample_ids = [c for c in avg.columns[2:] if c in set(meta["Sample_ID"])]
    gene_matrix = avg[["Gene Name", *sample_ids]].copy()
    gene_matrix = gene_matrix.groupby("Gene Name", as_index=True).mean(numeric_only=True)
    gene_matrix = gene_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    meta = meta.set_index("Sample_ID").loc[sample_ids].reset_index()
    return gene_matrix, meta


def build_membership_matrix(pathways: dict[str, list[str]], genes: pd.Index) -> tuple[list[str], np.ndarray, np.ndarray]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    kept_names: list[str] = []
    membership_rows: list[np.ndarray] = []
    sizes: list[int] = []
    for pathway, members in pathways.items():
        idxs = sorted({gene_to_idx[g] for g in members if g in gene_to_idx})
        if not (MIN_GENES <= len(idxs) <= MAX_GENES):
            continue
        row = np.zeros(len(genes), dtype=np.float32)
        row[idxs] = 1.0
        kept_names.append(pathway)
        membership_rows.append(row)
        sizes.append(len(idxs))
    membership = np.vstack(membership_rows)
    return kept_names, membership, np.asarray(sizes, dtype=np.float32)


def compute_pathway_scores(x: np.ndarray, membership: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, ddof=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    z = (x - means) / stds
    scores = z @ membership.T
    return scores / np.sqrt(sizes)[None, :]


def cluster_tests(score_df: pd.DataFrame, sample_meta: pd.DataFrame, drug_class: str) -> pd.DataFrame:
    subset_meta = sample_meta.loc[sample_meta["Drug_Class"].eq(drug_class)].copy()
    rows: list[dict[str, object]] = []
    for cluster in sorted(subset_meta["Cluster"].dropna().unique().tolist()):
        cluster_ids = subset_meta.loc[subset_meta["Cluster"].eq(cluster), "Sample_ID"].tolist()
        cluster_scores = score_df.loc[cluster_ids]
        pvals = []
        means = []
        for pathway in score_df.columns:
            values = cluster_scores[pathway].to_numpy(dtype=float)
            stat = ttest_1samp(values, 0.0, nan_policy="omit")
            pval = float(stat.pvalue) if np.isfinite(stat.pvalue) else 1.0
            pvals.append(max(pval, 1e-300))
            means.append(float(np.nanmean(values)))
        fdr = multipletests(pvals, alpha=ALPHA, method="fdr_bh")[1]
        for pathway, mean_score, pval, qval in zip(score_df.columns.tolist(), means, pvals, fdr.tolist()):
            rows.append(
                {
                    "drug_class": drug_class,
                    "cluster": int(cluster),
                    "n_drugs": int(len(cluster_ids)),
                    "pathway": pathway,
                    "mean_score": float(mean_score),
                    "p_value": float(pval),
                    "fdr": float(qval),
                    "direction": "Activated" if mean_score > 0 else ("Suppressed" if mean_score < 0 else "Neutral"),
                    "signed_log10_p": float(np.sign(mean_score) * -math.log10(max(pval, 1e-300))),
                    "signed_log10_fdr": float(np.sign(mean_score) * -math.log10(max(qval, 1e-300))),
                }
            )
    return pd.DataFrame(rows)


def significance_stars(pval: float) -> str:
    if pval < 1e-4:
        return "****"
    if pval < 1e-3:
        return "***"
    if pval < 1e-2:
        return "**"
    if pval < 5e-2:
        return "*"
    return ""


def summarize_main_pathways(
    anti_cluster_df: pd.DataFrame,
    non_cluster_df: pd.DataFrame,
    anti_score_df: pd.DataFrame,
    non_score_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    anti_clusters = sorted(sample_meta.loc[sample_meta["Drug_Class"].eq("Antineoplastic"), "Cluster"].dropna().unique().tolist())
    non_clusters = sorted(sample_meta.loc[sample_meta["Drug_Class"].eq("Non-Antineoplastic"), "Cluster"].dropna().unique().tolist())
    rows = []
    anti_abs_mean = anti_score_df.abs().mean(axis=0)
    non_abs_mean = non_score_df.abs().mean(axis=0)
    for pathway, group in anti_cluster_df.groupby("pathway"):
        non_group = non_cluster_df.loc[non_cluster_df["pathway"].eq(pathway)].copy()
        sig = group.loc[group["fdr"] < ALPHA].copy()
        sig_non = non_group.loc[non_group["fdr"] < ALPHA].copy()
        n_sig = int(len(sig))
        avg_strength = float((-np.log10(sig["p_value"].clip(lower=1e-300))).mean()) if n_sig else 0.0
        mean_abs_signed = float(sig["signed_log10_p"].abs().mean()) if n_sig else 0.0
        consistency = float(abs(np.sign(sig["mean_score"]).mean())) if n_sig else 0.0
        dominant_direction = "Mixed"
        if n_sig:
            dominant_direction = "Activated" if sig["mean_score"].mean() > 0 else "Suppressed"
        specificity = float(max(0.0, n_sig / max(1, len(anti_clusters)) - len(sig_non) / max(1, len(non_clusters))))
        anti_vs_non_margin = float(max(0.0, anti_abs_mean[pathway] - non_abs_mean.get(pathway, 0.0)))
        rows.append(
            {
                "pathway": pathway,
                "pathway_label": pretty_pathway_name(pathway),
                "functional_group": pathway_group(pathway),
                "n_sig_clusters": n_sig,
                "avg_sig_neglog10_p": avg_strength,
                "avg_sig_abs_signed_log10_p": mean_abs_signed,
                "direction_consistency": consistency,
                "dominant_direction": dominant_direction,
                "n_sig_non_clusters": int(len(sig_non)),
                "specificity": specificity,
                "anti_abs_score_mean": float(anti_abs_mean[pathway]),
                "non_abs_score_mean": float(non_abs_mean.get(pathway, 0.0)),
                "anti_vs_non_margin": anti_vs_non_margin,
            }
        )
    summary = pd.DataFrame(rows)
    for col in ["n_sig_clusters", "avg_sig_neglog10_p", "direction_consistency", "specificity", "anti_vs_non_margin"]:
        summary[f"{col}_pct"] = summary[col].rank(pct=True)
    summary["pan_drug_character_score"] = (
        0.35 * summary["n_sig_clusters_pct"]
        + 0.25 * summary["avg_sig_neglog10_p_pct"]
        + 0.20 * summary["specificity_pct"]
        + 0.20 * summary["anti_vs_non_margin_pct"]
    )
    summary = summary.sort_values(
        ["n_sig_clusters", "avg_sig_neglog10_p", "pan_drug_character_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def cluster_summary(sample_meta: pd.DataFrame) -> pd.DataFrame:
    anti = sample_meta.loc[sample_meta["Drug_Class"].eq("Antineoplastic")].copy()
    rows = []
    for cluster in sorted(anti["Cluster"].dropna().unique().tolist()):
        drugs = anti.loc[anti["Cluster"].eq(cluster), "Drug_Name"].tolist()
        rows.append(
            {
                "cluster": int(cluster),
                "n_drugs": int(len(drugs)),
                "example_drugs": "; ".join(drugs[:6]),
            }
        )
    return pd.DataFrame(rows)


def pathway_order_for_figure(summary_df: pd.DataFrame, pathway_gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    candidates = summary_df.loc[
        summary_df["functional_group"].isin(FIGURE_GROUPS)
        & summary_df["n_sig_clusters"].ge(MIN_FIGURE_SIG_CLUSTERS)
        & ~summary_df["pathway"].map(pathway_blacklisted)
    ].copy()
    candidates = candidates.loc[
        candidates.apply(lambda r: figure_keyword_match(r["pathway"], r["functional_group"]), axis=1)
    ].copy()
    candidates["figure_candidate"] = True

    selected_rows: list[pd.Series] = []
    selected_paths: set[str] = set()
    selected_by_group: dict[str, list[str]] = {group: [] for group in FIGURE_GROUPS}

    for group in FIGURE_GROUPS:
        group_df = candidates.loc[candidates["functional_group"].eq(group)].sort_values(
            ["n_sig_clusters", "avg_sig_neglog10_p", "pan_drug_character_score"],
            ascending=[False, False, False],
        )
        for _, row in group_df.iterrows():
            if len(selected_by_group[group]) >= FIGURE_GROUP_QUOTAS[group]:
                break
            genes = pathway_gene_sets.get(row["pathway"], set())
            if any(
                jaccard_similarity(genes, pathway_gene_sets.get(existing, set())) > MAX_JACCARD_WITHIN_GROUP
                for existing in selected_by_group[group]
            ):
                continue
            selected_rows.append(row)
            selected_paths.add(row["pathway"])
            selected_by_group[group].append(row["pathway"])

    if len(selected_rows) < TOP_N:
        fallback_df = candidates.loc[~candidates["pathway"].isin(selected_paths)].sort_values(
            ["n_sig_clusters", "avg_sig_neglog10_p", "pan_drug_character_score"],
            ascending=[False, False, False],
        )
        for _, row in fallback_df.iterrows():
            genes = pathway_gene_sets.get(row["pathway"], set())
            group = row["functional_group"]
            if any(
                jaccard_similarity(genes, pathway_gene_sets.get(existing, set())) > MAX_JACCARD_WITHIN_GROUP
                for existing in selected_by_group.get(group, [])
            ):
                continue
            selected_rows.append(row)
            selected_paths.add(row["pathway"])
            selected_by_group.setdefault(group, []).append(row["pathway"])
            if len(selected_rows) >= TOP_N:
                break

    top = pd.DataFrame(selected_rows).head(TOP_N).copy()
    if top.empty:
        top = summary_df.head(TOP_N).copy()

    group_rank = (
        top.groupby("functional_group", observed=False)["pan_drug_character_score"].mean().sort_values(ascending=False).reset_index()
    )
    group_order = group_rank["functional_group"].tolist()
    top["functional_group"] = pd.Categorical(top["functional_group"], categories=group_order, ordered=True)
    top = top.sort_values(
        ["functional_group", "pan_drug_character_score", "n_sig_clusters", "avg_sig_neglog10_p"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return top


def plot_main_figure(
    top_df: pd.DataFrame,
    anti_cluster_df: pd.DataFrame,
    cluster_info: pd.DataFrame,
) -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    anti_clusters = sorted(cluster_info["cluster"].tolist())
    cluster_labels = [f"C{c}\n(n={int(cluster_info.loc[cluster_info['cluster'].eq(c), 'n_drugs'].iloc[0])})" for c in anti_clusters]
    heat = (
        anti_cluster_df.loc[anti_cluster_df["pathway"].isin(top_df["pathway"])]
        .pivot(index="pathway", columns="cluster", values="signed_log10_p")
        .reindex(index=top_df["pathway"], columns=anti_clusters)
    )
    stars = (
        anti_cluster_df.loc[anti_cluster_df["pathway"].isin(top_df["pathway"])]
        .assign(stars=lambda d: d["p_value"].map(significance_stars))
        .pivot(index="pathway", columns="cluster", values="stars")
        .reindex(index=top_df["pathway"], columns=anti_clusters)
        .fillna("")
    )

    fig = plt.figure(figsize=(16.2, 9.6))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[4.3, 1.2, 1.3, 4.6, 1.6], wspace=0.08)
    ax_labels = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_labels)
    ax_b = fig.add_subplot(gs[0, 2], sharey=ax_labels)
    ax_c = fig.add_subplot(gs[0, 3], sharey=ax_labels)
    ax_d = fig.add_subplot(gs[0, 4], sharey=ax_labels)

    top_df = top_df.copy().reset_index(drop=True)
    n_rows = len(top_df)
    row_centers = np.arange(n_rows, dtype=float) + 0.5
    top_df["wrapped_label"] = top_df["pathway_label"].map(lambda x: wrap_label(x, 30))

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(n_rows, 0)
    ax_labels.axis("off")

    group_spans = []
    for group, group_df in top_df.groupby("functional_group", sort=False, observed=False):
        idxs = group_df.index.to_list()
        start, end = min(idxs), max(idxs) + 1
        group_spans.append((group, start, end))
        ax_labels.add_patch(
            plt.Rectangle((0.0, start + 0.02), 0.035, end - start - 0.04, color=GROUP_COLORS.get(group, "#8c8c8c"), alpha=0.95)
        )
        ax_labels.text(
            0.06,
            (start + end) / 2,
            wrap_group_label(group),
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            color=GROUP_COLORS.get(group, "#333333"),
        )

    for idx, row in top_df.iterrows():
        ax_labels.text(0.40, row_centers[idx], row["wrapped_label"], va="center", ha="left", fontsize=7.4, color="#1f2933")

    ax_a.barh(row_centers, top_df["n_sig_clusters"], height=0.78, color="#5f6b7a", edgecolor="white", linewidth=0.6)
    ax_a.set_title("i\nSignificant\nclusters", fontweight="bold")
    ax_a.set_xlabel("Count")
    ax_a.set_xlim(0, max(10, int(top_df["n_sig_clusters"].max()) + 1))
    ax_a.set_ylim(n_rows, 0)
    ax_a.set_yticks([])
    ax_a.grid(False)
    ax_a.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax_a.yaxis.grid(False)
    ax_a.set_axisbelow(True)

    ax_b.barh(row_centers, top_df["avg_sig_neglog10_p"], height=0.78, color="#7d8795", edgecolor="white", linewidth=0.6)
    ax_b.set_title("ii\nMean\n-log10(P)", fontweight="bold")
    ax_b.set_xlabel("Strength")
    ax_b.set_ylim(n_rows, 0)
    ax_b.set_yticks([])
    ax_b.grid(False)
    ax_b.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax_b.yaxis.grid(False)
    ax_b.set_axisbelow(True)

    vmax = float(np.nanmax(np.abs(heat.to_numpy(dtype=float)))) if len(heat) else 1.0
    vmax = max(vmax, 2.0)
    heatmap = sns.heatmap(
        heat,
        ax=ax_c,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        cbar=True,
        linewidths=0.35,
        linecolor="white",
        xticklabels=cluster_labels,
        yticklabels=False,
    )
    set_vertical_colorbar_title(heatmap, "Signed -log10(P)", width=10)
    ax_c.set_ylim(n_rows, 0)
    ax_c.grid(False)
    ax_c.set_title("iii\nCluster-level signed response", fontweight="bold")
    ax_c.set_xlabel("Antineoplastic clusters")
    ax_c.set_ylabel("")
    for i in range(stars.shape[0]):
        for j in range(stars.shape[1]):
            star = stars.iat[i, j]
            if star:
                ax_c.text(j + 0.5, i + 0.5, star, ha="center", va="center", fontsize=6.4, color="black")

    bar_colors = [GROUP_COLORS.get(group, "#8c8c8c") for group in top_df["functional_group"]]
    ax_d.barh(row_centers, top_df["pan_drug_character_score"] * 100.0, height=0.78, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax_d.set_title("iv\nPan-drug\ncharacter", fontweight="bold")
    ax_d.set_xlabel("Score")
    ax_d.set_ylim(n_rows, 0)
    ax_d.set_yticks([])
    ax_d.grid(False)
    ax_d.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax_d.yaxis.grid(False)
    ax_d.set_axisbelow(True)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        for _, _, end in group_spans[:-1]:
            ax.axhline(end, color="#c7ced6", lw=0.9, zorder=10)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"deepheal_pan_drug_pathway_response.{ext}", bbox_inches="tight")
    plt.close(fig)


def summarize_latent_capture(fdr_df: pd.DataFrame, selected_pathways: list[str], label: str) -> pd.DataFrame:
    available = [p for p in selected_pathways if p in fdr_df.columns]
    sub = fdr_df[available].copy()
    rows = []
    for pathway in available:
        vals = sub[pathway].clip(lower=1e-300)
        sig = vals[vals < ALPHA]
        rows.append(
            {
                "atlas": label,
                "pathway": pathway,
                "n_sig_latents": int((vals < ALPHA).sum()),
                "mean_sig_neglog10_fdr": float((-np.log10(sig)).mean()) if len(sig) else 0.0,
                "best_neglog10_fdr": float((-np.log10(vals.min()))) if len(vals) else 0.0,
                "best_fdr": float(vals.min()) if len(vals) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def plot_supplement(top_df: pd.DataFrame, latent_summary: pd.DataFrame) -> None:
    top_paths = top_df["pathway"].tolist()
    sup = latent_summary.loc[latent_summary["atlas"].eq("SupCon")].set_index("pathway").reindex(top_paths)
    avg = latent_summary.loc[latent_summary["atlas"].eq("Average")].set_index("pathway").reindex(top_paths)

    heat = pd.DataFrame(
        {
            "SupCon": sup["best_neglog10_fdr"],
            "Average": avg["best_neglog10_fdr"],
        },
        index=top_paths,
    )
    stars = pd.DataFrame(
        {
            "SupCon": sup["best_fdr"].map(significance_stars),
            "Average": avg["best_fdr"].map(significance_stars),
        },
        index=top_paths,
    ).fillna("")

    top_df = top_df.copy().reset_index(drop=True)
    top_df["sup_n_sig_latents"] = sup["n_sig_latents"].to_numpy()
    top_df["avg_n_sig_latents"] = avg["n_sig_latents"].to_numpy()
    top_df["sup_mean_sig_neglog10_fdr"] = sup["mean_sig_neglog10_fdr"].to_numpy()
    top_df["avg_mean_sig_neglog10_fdr"] = avg["mean_sig_neglog10_fdr"].to_numpy()
    top_df["capture_balance"] = top_df["sup_n_sig_latents"] - top_df["avg_n_sig_latents"]
    top_df["wrapped_label"] = top_df["pathway_label"].map(lambda x: wrap_label(x, 30))
    n_rows = len(top_df)
    row_centers = np.arange(n_rows, dtype=float) + 0.5

    sns.set_theme(style="ticks")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8})
    fig = plt.figure(figsize=(16.6, 9.6))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[4.3, 1.5, 1.5, 1.8, 2.0, 1.5], wspace=0.10)
    ax_labels = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax_labels)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax_labels)
    ax3 = fig.add_subplot(gs[0, 3], sharey=ax_labels)
    ax4 = fig.add_subplot(gs[0, 4], sharey=ax_labels)
    ax5 = fig.add_subplot(gs[0, 5], sharey=ax_labels)

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(n_rows, 0)
    ax_labels.axis("off")

    group_spans = []
    for group, group_df in top_df.groupby("functional_group", sort=False, observed=False):
        idxs = group_df.index.to_list()
        start, end = min(idxs), max(idxs) + 1
        group_spans.append((group, start, end))
        ax_labels.add_patch(plt.Rectangle((0.0, start + 0.02), 0.035, end - start - 0.04, color=GROUP_COLORS.get(group, "#8c8c8c"), alpha=0.95))
        ax_labels.text(
            0.06,
            (start + end) / 2,
            wrap_group_label(group),
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            color=GROUP_COLORS.get(group, "#333333"),
        )
    for idx, row in top_df.iterrows():
        ax_labels.text(0.40, row_centers[idx], row["wrapped_label"], va="center", ha="left", fontsize=7.4, color="#1f2933")

    ax1.barh(row_centers, top_df["sup_n_sig_latents"], height=0.78, color="#2b6cb0", edgecolor="white", linewidth=0.6)
    ax1.set_title("i\nSupCon\nsig. latents", fontweight="bold")
    ax1.set_xlabel("Count")
    ax1.set_ylim(n_rows, 0)
    ax1.set_yticks([])
    ax1.grid(False)
    ax1.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax1.yaxis.grid(False)
    ax1.set_axisbelow(True)

    ax2.barh(row_centers, top_df["avg_n_sig_latents"], height=0.78, color="#dd6b20", edgecolor="white", linewidth=0.6)
    ax2.set_title("ii\nAverage\nsig. latents", fontweight="bold")
    ax2.set_xlabel("Count")
    ax2.set_ylim(n_rows, 0)
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax2.yaxis.grid(False)
    ax2.set_axisbelow(True)

    ax3.barh(row_centers - 0.18, top_df["sup_mean_sig_neglog10_fdr"], height=0.34, color="#2b6cb0", edgecolor="white", linewidth=0.5, label="SupCon")
    ax3.barh(row_centers + 0.18, top_df["avg_mean_sig_neglog10_fdr"], height=0.34, color="#dd6b20", edgecolor="white", linewidth=0.5, label="Average")
    ax3.set_title("iii\nMean\n-log10(FDR)", fontweight="bold")
    ax3.set_xlabel("Strength")
    ax3.set_ylim(n_rows, 0)
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax3.yaxis.grid(False)
    ax3.set_axisbelow(True)
    ax3.legend(frameon=False, loc="lower right", fontsize=7)

    heatmap = sns.heatmap(
        heat,
        ax=ax4,
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.35,
        linecolor="white",
        xticklabels=["SupCon", "Average"],
        yticklabels=False,
    )
    set_vertical_colorbar_title(heatmap, "Best -log10(FDR)", width=12)
    ax4.set_ylim(n_rows, 0)
    ax4.grid(False)
    ax4.set_title("iv\nBest latent capture", fontweight="bold")
    ax4.set_xlabel("Atlas")
    for i in range(stars.shape[0]):
        for j in range(stars.shape[1]):
            star = stars.iat[i, j]
            if star:
                ax4.text(j + 0.5, i + 0.5, star, ha="center", va="center", fontsize=6.4, color="black")

    balance_colors = ["#2b6cb0" if v >= 0 else "#dd6b20" for v in top_df["capture_balance"]]
    ax5.barh(row_centers, top_df["capture_balance"], height=0.78, color=balance_colors, edgecolor="white", linewidth=0.6)
    ax5.axvline(0, color="#666666", lw=0.9)
    ax5.set_title("v\nCapture balance", fontweight="bold")
    ax5.set_xlabel("SupCon - Average")
    ax5.set_ylim(n_rows, 0)
    ax5.set_yticks([])
    ax5.grid(False)
    ax5.xaxis.grid(True, color="#d8dee8", linewidth=0.8)
    ax5.yaxis.grid(False)
    ax5.set_axisbelow(True)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for _, _, end in group_spans[:-1]:
            ax.axhline(end, color="#c7ced6", lw=0.9, zorder=10)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"deepheal_pan_drug_pathway_latent_capture_supplement.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pathways = load_canonical_pathways()
    gene_matrix, sample_meta = load_gene_matrix()
    kept_names, membership, sizes = build_membership_matrix(pathways, gene_matrix.index)
    genes_in_data = set(gene_matrix.index.tolist())
    pathway_gene_sets = {name: set(pathways[name]) & genes_in_data for name in kept_names}
    matrix = gene_matrix[sample_meta["Sample_ID"].tolist()].T.to_numpy(dtype=np.float32)
    score_matrix = compute_pathway_scores(matrix, membership, sizes)
    score_df = pd.DataFrame(score_matrix, index=sample_meta["Sample_ID"].tolist(), columns=kept_names)

    anti_score_df = score_df.loc[sample_meta.loc[sample_meta["Drug_Class"].eq("Antineoplastic"), "Sample_ID"]]
    non_score_df = score_df.loc[sample_meta.loc[sample_meta["Drug_Class"].eq("Non-Antineoplastic"), "Sample_ID"]]

    anti_cluster_df = cluster_tests(score_df, sample_meta, "Antineoplastic")
    non_cluster_df = cluster_tests(score_df, sample_meta, "Non-Antineoplastic")
    anti_cluster_df.to_csv(OUT_DIR / "pan_drug_signed_cluster_responses_antineoplastic.csv", index=False)
    non_cluster_df.to_csv(OUT_DIR / "pan_drug_signed_cluster_responses_non_antineoplastic.csv", index=False)

    cluster_info = cluster_summary(sample_meta)
    cluster_info.to_csv(OUT_DIR / "pan_drug_antineoplastic_cluster_summary.csv", index=False)

    summary_df = summarize_main_pathways(anti_cluster_df, non_cluster_df, anti_score_df, non_score_df, sample_meta)
    summary_df["figure_blacklisted"] = summary_df["pathway"].map(pathway_blacklisted)
    summary_df["figure_keyword_match"] = summary_df.apply(
        lambda r: figure_keyword_match(r["pathway"], r["functional_group"]) if r["functional_group"] in FIGURE_GROUPS else False,
        axis=1,
    )
    summary_df["figure_candidate_prequota"] = (
        summary_df["functional_group"].isin(FIGURE_GROUPS)
        & summary_df["n_sig_clusters"].ge(MIN_FIGURE_SIG_CLUSTERS)
        & ~summary_df["figure_blacklisted"]
        & summary_df["figure_keyword_match"]
    )
    summary_df.to_csv(OUT_DIR / "pan_drug_pathway_summary_all.csv", index=False)
    top_df = pathway_order_for_figure(summary_df, pathway_gene_sets)
    top_df.to_csv(OUT_DIR / "pan_drug_pathway_summary_top_for_figure.csv", index=False)

    supcon_fdr = pd.read_csv(SUPCON_FDR, index_col=0)
    average_fdr = pd.read_csv(AVERAGE_FDR, index_col=0)
    latent_summary = pd.concat(
        [
            summarize_latent_capture(supcon_fdr, top_df["pathway"].tolist(), "SupCon"),
            summarize_latent_capture(average_fdr, top_df["pathway"].tolist(), "Average"),
        ],
        ignore_index=True,
    )
    latent_summary.to_csv(OUT_DIR / "pan_drug_pathway_latent_capture_summary.csv", index=False)

    plot_main_figure(top_df, anti_cluster_df, cluster_info)
    plot_supplement(top_df, latent_summary)

    summary_payload = {
        "n_pathways_considered": int(len(kept_names)),
        "n_top_pathways_displayed": int(len(top_df)),
        "anti_clusters": cluster_info.to_dict(orient="records"),
        "main_top_pathways": top_df[
            ["pathway", "functional_group", "n_sig_clusters", "avg_sig_neglog10_p", "dominant_direction", "pan_drug_character_score"]
        ].to_dict(orient="records"),
        "main_figure": str(FIG_DIR / "deepheal_pan_drug_pathway_response.png"),
        "supplement_figure": str(FIG_DIR / "deepheal_pan_drug_pathway_latent_capture_supplement.png"),
    }
    (OUT_DIR / "pan_drug_pathway_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
