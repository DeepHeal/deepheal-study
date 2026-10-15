from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact


WORK_DIR = ARTICLE_ROOT / "results/pathway_comparison"
PACKAGE_ROOT = ARTICLE_ROOT
FIGURE_PACKAGES_ROOT = ARTICLE_ROOT
PROJECT_ROOT = ARTICLE_ROOT
MANUSCRIPT_ROOT = ARTICLE_ROOT
DATA_DIR = WORK_DIR / "data"

MATCHED_BASELINE_DIR = DATA_DIR / "matched_baselines"
SOURCE_VAE_DIR = DATA_DIR / "source_vae_comparator"
DEEPHEAL_TARGET_DIR = DATA_DIR / "deepheal_targets"
REFERENCE_COMPARATOR_DIR = DATA_DIR / "reference_comparator"
DEEPHEAL_RESULTS = PROJECT_ROOT / "results"
PATHWAY_GMT_DIR = PROJECT_ROOT / "data" / "pathways" / "gmt_cache"
LEGACY_FIG2_DIR = ARTICLE_ROOT / "data/pathway_comparators"
FISHER_ALTERNATIVE = "greater"
FDR_METHOD = "benjamini_hochberg_per_pathway"
FDR_ALPHA = 0.05
TOP_N_BY_COLLECTION = {"c2_cp": 31, "c6_oncogenic": 75}
TRIAD_PREFIXES = ("KEGG_", "BIOCARTA_", "REACTOME_")


BASELINE_COLORS = {
    "rp": "#8dd3c7",
    "pca": "#f4cae4",
    "ica": "#bebada",
    "ae": "#80b1d3",
    "dae": "#fdb462",
}


def _baseline_meta(prefix: str, dim: int, display: str) -> dict[str, object]:
    method = f"{prefix}_{dim}L"
    return {
        "display": display,
        "tick": f"{display}\n{dim}L",
        "color": BASELINE_COLORS[prefix],
        "role": "matched_baseline",
        "latent_dim": dim,
        "canonical": MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c2_canonical.csv",
        "oncogenic": MATCHED_BASELINE_DIR / f"{method}_fdr_matrix_c6_oncogenic.csv",
    }


METHOD_META: dict[str, dict[str, object]] = {
    **{f"{prefix}_257L": _baseline_meta(prefix, 257, display) for prefix, display in [
        ("rp", "RP"),
        ("pca", "PCA"),
        ("ica", "ICA"),
        ("ae", "AE"),
        ("dae", "DAE"),
    ]},
    **{f"{prefix}_194L": _baseline_meta(prefix, 194, display) for prefix, display in [
        ("rp", "RP"),
        ("pca", "PCA"),
        ("ica", "ICA"),
        ("ae", "AE"),
        ("dae", "DAE"),
    ]},
    "deepheal_supcon": {
        "display": "DeepHeal Supervised",
        "tick": "DeepHeal\nSupervised\n257L",
        "color": "#fb8072",
        "role": "target",
        "latent_dim": 257,
        "canonical": DEEPHEAL_TARGET_DIR / "deepheal_supcon_fdr_matrix_c2_cp.csv",
        "oncogenic": DEEPHEAL_TARGET_DIR / "deepheal_supcon_fdr_matrix_c6_oncogenic.csv",
    },
    "deepheal_average": {
        "display": "DeepHeal Unsupervised",
        "tick": "DeepHeal\nUnsupervised\n194L",
        "color": "#1f78b4",
        "role": "target",
        "latent_dim": 194,
        "canonical": DEEPHEAL_TARGET_DIR / "deepheal_average_fdr_matrix_c2_cp.csv",
        "oncogenic": DEEPHEAL_TARGET_DIR / "deepheal_average_fdr_matrix_c6_oncogenic.csv",
    },
    "vae_161L": {
        "display": "VAE reference",
        "tick": "VAE reference\n161L",
        "color": "#b3de69",
        "role": "reference",
        "latent_dim": 161,
        "canonical": REFERENCE_COMPARATOR_DIR / "vae_161L_fdr_matrix_c2_canonical.csv",
        "oncogenic": REFERENCE_COMPARATOR_DIR / "vae_161L_fdr_matrix_c6_oncogenic.csv",
    },
    "source_vae_supcon": {
        "display": "VAE source",
        "tick": "VAE source\nSupCon bank",
        "color": "#bc80bd",
        "role": "source_vae_comparator",
        "source_branch": "supcon",
        "latent_dim": None,
        "model_detection": SOURCE_VAE_DIR / "source_vae_supcon_model_detection_c2_canonical.csv",
        "pathway_detection": SOURCE_VAE_DIR / "source_vae_supcon_pathway_detection_c2_canonical.csv",
    },
    "source_vae_average": {
        "display": "VAE source",
        "tick": "VAE source\nDeepHeal Unsupervised bank",
        "color": "#ccebc5",
        "role": "source_vae_comparator",
        "source_branch": "average",
        "latent_dim": None,
        "model_detection": SOURCE_VAE_DIR / "source_vae_average_model_detection_c2_canonical.csv",
        "pathway_detection": SOURCE_VAE_DIR / "source_vae_average_pathway_detection_c2_canonical.csv",
    },
}


COMPARISON_SPECS: dict[str, dict[str, object]] = {
    "primary_deepheal_257L": {
        "label": "Primary DeepHeal Supervised comparison",
        "short_label": "DeepHeal Supervised 257L",
        "matched_dim": 257,
        "target_method": "deepheal_supcon",
        "baseline_methods": ["rp_257L", "pca_257L", "ica_257L", "ae_257L", "dae_257L"],
        "source_vae_method": "source_vae_supcon",
        "reference_method": "vae_161L",
    },
    "sensitivity_average_194L": {
        "label": "Sensitivity / DeepHeal Unsupervised branch comparison",
        "short_label": "DeepHeal Unsupervised 194L",
        "matched_dim": 194,
        "target_method": "deepheal_average",
        "baseline_methods": ["rp_194L", "pca_194L", "ica_194L", "ae_194L", "dae_194L"],
        "source_vae_method": "source_vae_average",
        "reference_method": "vae_161L",
    },
}

METHOD_ORDER = [
    "rp_257L",
    "pca_257L",
    "ica_257L",
    "ae_257L",
    "dae_257L",
    "deepheal_supcon",
    "vae_161L",
    "source_vae_supcon",
    "rp_194L",
    "pca_194L",
    "ica_194L",
    "ae_194L",
    "dae_194L",
    "deepheal_average",
    "source_vae_average",
]

PREFIX_GROUPS = {
    "kegg": ("KEGG_",),
    "biocarta": ("BIOCARTA_",),
    "reactome": ("REACTOME_",),
}

COLLECTION_ORDER = ["kegg", "biocarta", "reactome", "canonical_all", "oncogenic_all", "c2_cp_all_1308"]
COLLECTION_LABELS = {
    "kegg": "KEGG",
    "biocarta": "BioCarta",
    "reactome": "Reactome",
    "canonical_all": "Canonical pathways (shared)",
    "oncogenic_all": "Oncogenic signatures",
    "c2_cp_all_1308": "C2 CP target panel",
}
NODE_THRESHOLDS = list(range(1, 11))
FDR_THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
OVERLAP_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20]
SOURCE_VAE_RECURRENCE_THRESHOLDS = [1, 2, 5, 10, 20, 50]

NATURE_FONT_SIZE = 20.0
NATURE_SMALL_FONT_SIZE = 15.0
NATURE_TITLE_SIZE = 23.0
NATURE_PANEL_LABEL_SIZE = 34.0
NATURE_AXIS_LABEL_SIZE = 22.0
NATURE_BAR_TICK_SIZE = 17.0
NATURE_BAR_YTICK_SIZE = 14.0
NATURE_AXIS_LINEWIDTH = 1.1
NATURE_TICK_LINEWIDTH = 1.0
NATURE_TICK_SIZE = 8.0
NATURE_OUTPUT_DPI = 100
FIGURE_SIZES = {
    "average_pathway_coverages": (15.92, 10.64),
    "pathway_coverage_distributions": (15.92, 10.64),
    "node_level_pathway_annotations": (14.67, 12.0),
    "pathway_detection_overlap": (21.0, 10.0),
    "main_composite": (15.92, 17.4),
}
BAR_LABELS_COMPACT = {
    "kegg": "KEGG",
    "biocarta": "BioCarta",
    "reactome": "Reactome",
    "canonical_all": "Canonical\n(shared)",
    "oncogenic_all": "Oncogenic\nsignatures",
}
SCATTER_POINT_SIZE = 36.0
SCATTER_ALPHA = 1.0
MEDIAN_LINEWIDTH = 2.4
CURVE_LINEWIDTH = 3.0
CURVE_MARKER_SIZE = 10.5
THRESHOLD_BAR_WIDTH = 0.82
THRESHOLD_BAR_ALPHA = 0.58


def all_configured_methods() -> list[str]:
    return METHOD_ORDER.copy()


def comparison_methods(
    comparison_id: str,
    include_reference: bool = True,
    include_source_vae: bool = False,
) -> list[str]:
    spec = COMPARISON_SPECS[comparison_id]
    methods = list(spec["baseline_methods"]) + [str(spec["target_method"])]
    if include_source_vae:
        methods.insert(len(spec["baseline_methods"]), str(spec["source_vae_method"]))
    if include_reference:
        methods.append(str(spec["reference_method"]))
    return methods


def matrix_methods(comparison_id: str) -> list[str]:
    return comparison_methods(comparison_id, include_reference=True, include_source_vae=False)


def unique_matrix_methods() -> list[str]:
    seen: list[str] = []
    for comparison_id in COMPARISON_SPECS:
        for method in matrix_methods(comparison_id):
            if method not in seen:
                seen.append(method)
    return seen


def source_vae_methods() -> list[str]:
    return [str(spec["source_vae_method"]) for spec in COMPARISON_SPECS.values()]


def comparison_label(comparison_id: str) -> str:
    return str(COMPARISON_SPECS[comparison_id]["label"])


def comparison_short_label(comparison_id: str) -> str:
    return str(COMPARISON_SPECS[comparison_id]["short_label"])


def apply_nature_style(font_size: float = NATURE_FONT_SIZE) -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": NATURE_AXIS_LABEL_SIZE,
            "axes.titlesize": NATURE_TITLE_SIZE,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": NATURE_SMALL_FONT_SIZE,
            "axes.linewidth": NATURE_AXIS_LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": NATURE_TICK_LINEWIDTH,
            "ytick.major.width": NATURE_TICK_LINEWIDTH,
            "xtick.major.size": NATURE_TICK_SIZE,
            "ytick.major.size": NATURE_TICK_SIZE,
            "figure.dpi": NATURE_OUTPUT_DPI,
            "savefig.dpi": NATURE_OUTPUT_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_display(method: str) -> str:
    return str(METHOD_META[method]["display"])


def method_tick(method: str) -> str:
    return str(METHOD_META[method]["tick"])


def overlap_panel_title(comparison_id: str, method: str) -> str:
    if comparison_id not in COMPARISON_SPECS:
        raise KeyError(comparison_id)
    return method_tick(method)


def method_color(method: str) -> str:
    return str(METHOD_META[method]["color"])


def method_compact_tick(method: str) -> str:
    if method.startswith("rp_"):
        return "RP"
    if method.startswith("pca_"):
        return "PCA"
    if method.startswith("ica_"):
        return "ICA"
    if method.startswith("ae_"):
        return "AE"
    if method.startswith("dae_"):
        return "DAE"
    if method == "deepheal_supcon":
        return "Deep\nHeal"
    if method == "deepheal_average":
        return "Deep\nHeal"
    if method == "vae_161L":
        return "VAE\nref."
    return method_tick(method)


def method_role(method: str) -> str:
    return str(METHOD_META[method]["role"])


def method_handles(methods: Iterable[str] | None = None) -> list[Line2D]:
    use_methods = list(methods) if methods is not None else unique_matrix_methods()
    return [
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=method_color(method), label=method_tick(method))
        for method in use_methods
    ]


def compact_method_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=BASELINE_COLORS["rp"], label="RP"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=BASELINE_COLORS["pca"], label="PCA"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=BASELINE_COLORS["ica"], label="ICA"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=BASELINE_COLORS["ae"], label="AE"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=BASELINE_COLORS["dae"], label="DAE"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=method_color("deepheal_supcon"), label="DeepHeal supervised"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=method_color("deepheal_average"), label="DeepHeal unsupervised"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10, color=method_color("vae_161L"), label="VAE reference"),
    ]


def add_compact_legend(fig: plt.Figure, y: float = 0.005) -> None:
    fig.legend(
        handles=compact_method_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        frameon=False,
        fontsize=NATURE_SMALL_FONT_SIZE,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=1.3,
        borderaxespad=0.0,
    )


def _relative_to_work_dir(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(WORK_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_indexed_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")


def _copy_reference_matrix(source_name: str, target_name: str, force: bool = False) -> None:
    target = REFERENCE_COMPARATOR_DIR / target_name
    if target.exists() and not force:
        return
    source = LEGACY_FIG2_DIR / source_name
    if not source.exists():
        if target.exists():
            return
        raise FileNotFoundError(f"Missing reference comparator source and package-local copy: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def ensure_reference_comparator_matrices(force: bool = False) -> None:
    _copy_reference_matrix("replicate_161L_fdr_matrix_c2_canonical.csv", "vae_161L_fdr_matrix_c2_canonical.csv", force)
    _copy_reference_matrix("replicate_161L_fdr_matrix_c6_oncogenic.csv", "vae_161L_fdr_matrix_c6_oncogenic.csv", force)


def parse_gmt(path: Path) -> tuple[list[str], list[list[str]]]:
    names: list[str] = []
    genes: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            names.append(parts[0].strip())
            genes.append([gene.strip() for gene in parts[2:] if gene.strip()])
    return names, genes


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def recompute_deepheal_target_matrices(branch: str, method: str, force: bool = False) -> dict[str, Path]:
    outputs = {
        "canonical_pvalue": DEEPHEAL_TARGET_DIR / f"{method}_pvalue_matrix_c2_cp.csv",
        "canonical_fdr": Path(METHOD_META[method]["canonical"]),
        "oncogenic_pvalue": DEEPHEAL_TARGET_DIR / f"{method}_pvalue_matrix_c6_oncogenic.csv",
        "oncogenic_fdr": Path(METHOD_META[method]["oncogenic"]),
    }
    if all(path.exists() for path in outputs.values()) and not force:
        return outputs

    weights_path = DEEPHEAL_RESULTS / branch / f"{branch}_gene_weights.csv"
    if not weights_path.exists():
        if all(path.exists() for path in outputs.values()):
            return outputs
        raise FileNotFoundError(f"Missing DeepHeal target weights for {method}: {weights_path}")

    weights = pd.read_csv(weights_path, index_col=0, encoding="utf-8")
    weights.index = weights.index.astype(str)
    weights.columns = weights.columns.astype(str)
    gene_to_idx = {gene: idx for idx, gene in enumerate(weights.columns.tolist())}
    score_matrix = weights.to_numpy(dtype=float)
    universe_size = len(gene_to_idx)

    for collection_key, gmt_name, pvalue_key, fdr_key in [
        ("c2_cp", "c2.cp.v6.2.symbols.gmt", "canonical_pvalue", "canonical_fdr"),
        ("c6_oncogenic", "c6.all.v6.2.symbols.gmt", "oncogenic_pvalue", "oncogenic_fdr"),
    ]:
        top_n = TOP_N_BY_COLLECTION[collection_key]
        pathway_names, pathway_gene_lists = parse_gmt(PATHWAY_GMT_DIR / gmt_name)
        clean_pathways: list[tuple[str, set[int]]] = []
        for pathway_name, genes in zip(pathway_names, pathway_gene_lists):
            member_indices = {gene_to_idx[gene] for gene in genes if gene in gene_to_idx}
            if member_indices:
                clean_pathways.append((pathway_name, member_indices))

        pvals = np.ones((score_matrix.shape[0], len(clean_pathways)), dtype=float)
        pathway_sizes = [len(members) for _, members in clean_pathways]
        for row_idx, scores in enumerate(score_matrix):
            top_indices = set(np.argsort(scores)[-top_n:].tolist())
            for col_idx, (_, pathway_indices) in enumerate(clean_pathways):
                overlap = len(top_indices & pathway_indices)
                pathway_size = pathway_sizes[col_idx]
                table = np.array(
                    [
                        [overlap, pathway_size - overlap],
                        [top_n - overlap, universe_size - pathway_size - top_n + overlap],
                    ],
                    dtype=int,
                )
                pvals[row_idx, col_idx] = fisher_exact(table, alternative=FISHER_ALTERNATIVE).pvalue

        fdr = np.ones_like(pvals)
        for col_idx in range(pvals.shape[1]):
            fdr[:, col_idx] = bh_adjust(pvals[:, col_idx])

        columns = [name for name, _ in clean_pathways]
        pvalue_df = pd.DataFrame(pvals, index=weights.index, columns=columns)
        fdr_df = pd.DataFrame(fdr, index=weights.index, columns=columns)
        _write_indexed_table(pvalue_df, outputs[pvalue_key])
        _write_indexed_table(fdr_df, outputs[fdr_key])

    return outputs


def ensure_deepheal_target_matrices(force: bool = False) -> None:
    for branch, method in [("supcon", "deepheal_supcon"), ("average", "deepheal_average")]:
        recompute_deepheal_target_matrices(branch=branch, method=method, force=force)


def ensure_package_input_matrices(force_targets: bool = False, force_reference: bool = False) -> None:
    ensure_reference_comparator_matrices(force=force_reference)
    ensure_deepheal_target_matrices(force=force_targets)


def load_fdr_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing FDR matrix: {path}")
    df = pd.read_csv(path, index_col=0, encoding="utf-8")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = df.index.astype(str)
    return df


def load_all_matrices(comparison_id: str | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    methods = matrix_methods(comparison_id) if comparison_id else unique_matrix_methods()
    matrices: dict[str, dict[str, pd.DataFrame]] = {}
    for method in methods:
        matrices[method] = {
            "canonical": load_fdr_matrix(Path(METHOD_META[method]["canonical"])),
            "oncogenic": load_fdr_matrix(Path(METHOD_META[method]["oncogenic"])),
        }
    return matrices


def subset_columns(columns: Iterable[str], prefixes: tuple[str, ...] | None) -> list[str]:
    if prefixes is None:
        return list(columns)
    return [col for col in columns if col.startswith(prefixes)]


def shared_canonical_columns(comparison_id: str, matrices: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    method_columns = []
    for method in matrix_methods(comparison_id):
        cols = {col for col in matrices[method]["canonical"].columns if col.startswith(TRIAD_PREFIXES)}
        method_columns.append(cols)
    shared = set.intersection(*method_columns)
    reference_method = str(COMPARISON_SPECS[comparison_id]["reference_method"])
    reference_order = [col for col in matrices[reference_method]["canonical"].columns if col in shared]
    if len(reference_order) != len(shared):
        reference_order.extend(sorted(shared - set(reference_order)))
    return reference_order


def restrict_to_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Matrix is missing {len(missing)} shared canonical columns; first missing: {missing[0]}")
    return df.loc[:, columns].copy()


def _comparison_fields(comparison_id: str, method: str) -> dict[str, object]:
    spec = COMPARISON_SPECS[comparison_id]
    return {
        "comparison_id": comparison_id,
        "comparison_label": comparison_label(comparison_id),
        "comparison_short_label": comparison_short_label(comparison_id),
        "matched_dim": int(spec["matched_dim"]),
        "method_role": method_role(method),
    }


def summarize_fdr_matrix(
    comparison_id: str,
    method: str,
    df: pd.DataFrame,
    collection_id: str,
    prefixes: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = subset_columns(df.columns, prefixes)
    values = df.loc[:, cols].copy()
    sig_mask = values.lt(0.05)
    common = _comparison_fields(comparison_id, method)
    summary = pd.DataFrame(
        [
            {
                **common,
                "method": method,
                "method_label": method_display(method),
                "method_tick": method_tick(method),
                "collection_id": collection_id,
                "collection_label": COLLECTION_LABELS[collection_id],
                "latent_dims": int(values.shape[0]),
                "n_pathways_tested": int(values.shape[1]),
                "avg_sig_pathways_per_latent": float(sig_mask.sum(axis=1).mean()),
                "median_sig_pathways_per_latent": float(sig_mask.sum(axis=1).median()),
                "annotated_latent_pct": float((sig_mask.sum(axis=1) > 0).mean() * 100.0),
                "unique_sig_pathways": int(sig_mask.any(axis=0).sum()),
                "total_sig_tests": int(sig_mask.to_numpy().sum()),
            }
        ]
    )
    counts = pd.DataFrame(
        {
            **common,
            "method": method,
            "method_label": method_display(method),
            "method_tick": method_tick(method),
            "collection_id": collection_id,
            "collection_label": COLLECTION_LABELS[collection_id],
            "node": values.index.astype(str),
            "n_significant_pathways": sig_mask.sum(axis=1).astype(int).to_numpy(),
        }
    )
    threshold_rows = []
    for threshold in FDR_THRESHOLDS:
        threshold_rows.append(
            {
                **common,
                "method": method,
                "method_label": method_display(method),
                "method_tick": method_tick(method),
                "collection_id": collection_id,
                "collection_label": COLLECTION_LABELS[collection_id],
                "threshold": threshold,
                "neg_log10_threshold": float(-np.log10(threshold)),
                "annotated_latent_pct": float(values.lt(threshold).any(axis=1).mean() * 100.0),
                "mean_sig_pathways_per_latent": float(values.lt(threshold).sum(axis=1).mean()),
            }
        )
    return summary, counts, pd.DataFrame(threshold_rows)


def build_node_level_curve(comparison_id: str, method: str, df: pd.DataFrame, collection_id: str) -> pd.DataFrame:
    values = df.fillna(1.0)
    best = -np.log10(np.clip(values.min(axis=1).to_numpy(dtype=float), 1e-300, 1.0))
    rows = []
    common = _comparison_fields(comparison_id, method)
    for threshold in NODE_THRESHOLDS:
        mask = best > threshold
        rows.append(
            {
                **common,
                "method": method,
                "method_label": method_display(method),
                "method_tick": method_tick(method),
                "collection_id": collection_id,
                "collection_label": COLLECTION_LABELS[collection_id],
                "threshold": threshold,
                "pct_nodes": float(mask.mean() * 100.0),
                "n_nodes": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_overlap_table(
    comparison_id: str,
    matrices: dict[str, dict[str, pd.DataFrame]],
    reference_method: str | None = None,
    canonical_columns: list[str] | None = None,
) -> pd.DataFrame:
    ref_method = reference_method or str(COMPARISON_SPECS[comparison_id]["reference_method"])
    if canonical_columns is None:
        canonical_columns = list(matrices[ref_method]["canonical"].columns)
    ref_sig = restrict_to_columns(matrices[ref_method]["canonical"], canonical_columns).lt(0.05).sum(axis=0)
    ref_paths = set(ref_sig[ref_sig >= 1].index.tolist())
    rows = []
    for method in matrix_methods(comparison_id):
        if method == ref_method:
            continue
        sig = restrict_to_columns(matrices[method]["canonical"], canonical_columns).lt(0.05).sum(axis=0)
        for threshold in OVERLAP_THRESHOLDS:
            method_paths = set(sig[sig >= threshold].index.tolist())
            common = method_paths & ref_paths
            rows.append(
                {
                    **_comparison_fields(comparison_id, method),
                    "method": method,
                    "method_label": method_display(method),
                    "method_tick": method_tick(method),
                    "reference_method": ref_method,
                    "reference_label": method_display(ref_method),
                    "threshold": threshold,
                    "n_detected_pathways": int(len(method_paths)),
                    "n_overlap_with_reference": int(len(common)),
                    "pct_overlap_with_reference": float(len(common) / len(method_paths) * 100.0)
                    if method_paths
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_source_vae_model_detection(
    method: str,
    fdr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    values = fdr_df.copy()
    if "node" in values.columns:
        values = values.set_index("node")
    values = values.apply(pd.to_numeric, errors="coerce")
    if len(values) != len(metadata):
        raise ValueError(f"FDR matrix rows ({len(values)}) do not match metadata rows ({len(metadata)})")

    meta = metadata.reset_index(drop=True).copy()
    meta["source_latent_dim"] = meta["source_latent_dim"].astype(int)
    meta["source_run"] = meta["source_run"].astype(int)
    sig = values.lt(alpha).reset_index(drop=True)

    rows = []
    for (latent_dim, run), frame in meta.groupby(["source_latent_dim", "source_run"], sort=False):
        idx = frame.index.to_numpy()
        detected = sig.iloc[idx].any(axis=0)
        model_key = f"L{int(latent_dim)}_run{int(run):03d}"
        rows.append(
            {
                "method": method,
                "method_label": method_display(method) if method in METHOD_META else method,
                "method_tick": method_tick(method) if method in METHOD_META else method,
                "model_key": model_key,
                "source_latent_dim": int(latent_dim),
                "source_run": int(run),
                "n_source_nodes": int(len(idx)),
                "n_pathways_tested": int(values.shape[1]),
                "n_detected_pathways": int(detected.sum()),
                "pct_pathways_detected": float(detected.mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def load_source_vae_detection_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for comparison_id, spec in COMPARISON_SPECS.items():
        method = str(spec["source_vae_method"])
        path = Path(METHOD_META[method]["model_detection"])
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8")
            df["comparison_id"] = comparison_id
            df["comparison_label"] = comparison_label(comparison_id)
            df["comparison_short_label"] = comparison_short_label(comparison_id)
            tables[method] = df
    return tables


def build_source_vae_percent_table(
    source_vae_detection: pd.DataFrame,
    matrices_by_comparison: dict[str, dict[str, dict[str, pd.DataFrame]]],
    shared_canonical_by_comparison: dict[str, list[str]],
) -> pd.DataFrame:
    if source_vae_detection.empty:
        return pd.DataFrame()
    rows = []
    pathway_tables = {}
    for comparison_id, spec in COMPARISON_SPECS.items():
        method = str(spec["source_vae_method"])
        path = Path(METHOD_META[method]["pathway_detection"])
        if path.exists():
            pathway_tables[comparison_id] = pd.read_csv(path, encoding="utf-8")

    for comparison_id, pathway_df in pathway_tables.items():
        target = str(COMPARISON_SPECS[comparison_id]["target_method"])
        shared_columns = shared_canonical_by_comparison[comparison_id]
        target_sig = restrict_to_columns(matrices_by_comparison[comparison_id][target]["canonical"], shared_columns).lt(0.05).any(axis=0)
        target_paths = set(target_sig[target_sig].index.tolist())
        for threshold in SOURCE_VAE_RECURRENCE_THRESHOLDS:
            detected = pathway_df[pathway_df["pct_source_models_detected"] >= threshold]
            detected_paths = set(detected["pathway"].astype(str)) & set(shared_columns)
            overlap = detected_paths & target_paths
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_label": comparison_label(comparison_id),
                    "comparison_short_label": comparison_short_label(comparison_id),
                    "target_method": target,
                    "target_label": method_display(target),
                    "source_vae_method": str(COMPARISON_SPECS[comparison_id]["source_vae_method"]),
                    "source_vae_threshold_pct": threshold,
                    "n_source_vae_recurrent_pathways": int(len(detected_paths)),
                    "n_overlap_with_target": int(len(overlap)),
                    "pct_overlap_with_target": float(len(overlap) / len(detected_paths) * 100.0) if detected_paths else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_analysis_tables(force: bool = True) -> dict[str, pd.DataFrame]:
    ensure_package_input_matrices(force_targets=force, force_reference=force)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": DATA_DIR / "deepheal_pathway_summary_long.csv",
        "counts": DATA_DIR / "deepheal_pathway_counts_per_latent_long.csv",
        "thresholds": DATA_DIR / "deepheal_pathway_threshold_curves_long.csv",
        "node_curves": DATA_DIR / "deepheal_pathway_node_level_curves_long.csv",
        "overlap": DATA_DIR / "deepheal_pathway_overlap_vs_vae_reference.csv",
        "source_vae_detection": DATA_DIR / "deepheal_source_vae_model_detection_long.csv",
        "source_vae_percent": DATA_DIR / "deepheal_source_vae_percent_overlap_long.csv",
    }
    if not force and all(path.exists() for path in output_paths.values()):
        return {key: pd.read_csv(path, encoding="utf-8") for key, path in output_paths.items()}

    summaries = []
    counts = []
    thresholds = []
    node_curves = []
    overlaps = []
    matrices_by_comparison: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    shared_canonical_by_comparison: dict[str, list[str]] = {}

    for comparison_id in COMPARISON_SPECS:
        matrices = load_all_matrices(comparison_id)
        matrices_by_comparison[comparison_id] = matrices
        shared_columns = shared_canonical_columns(comparison_id, matrices)
        shared_canonical_by_comparison[comparison_id] = shared_columns
        for method in matrix_methods(comparison_id):
            canonical = restrict_to_columns(matrices[method]["canonical"], shared_columns)
            oncogenic = matrices[method]["oncogenic"]
            for collection_id, prefixes in [
                ("kegg", PREFIX_GROUPS["kegg"]),
                ("biocarta", PREFIX_GROUPS["biocarta"]),
                ("reactome", PREFIX_GROUPS["reactome"]),
                ("canonical_all", None),
            ]:
                summary, count, threshold_df = summarize_fdr_matrix(comparison_id, method, canonical, collection_id, prefixes)
                summaries.append(summary)
                counts.append(count)
                thresholds.append(threshold_df)
            summary, count, threshold_df = summarize_fdr_matrix(comparison_id, method, oncogenic, "oncogenic_all", None)
            summaries.append(summary)
            counts.append(count)
            thresholds.append(threshold_df)
            node_curves.append(build_node_level_curve(comparison_id, method, canonical, "canonical_all"))
            node_curves.append(build_node_level_curve(comparison_id, method, oncogenic, "oncogenic_all"))
        target_method = str(COMPARISON_SPECS[comparison_id]["target_method"])
        target_full_c2 = matrices[target_method]["canonical"]
        summary, count, threshold_df = summarize_fdr_matrix(
            comparison_id,
            target_method,
            target_full_c2,
            "c2_cp_all_1308",
            None,
        )
        summaries.append(summary)
        counts.append(count)
        thresholds.append(threshold_df)
        node_curves.append(build_node_level_curve(comparison_id, target_method, target_full_c2, "c2_cp_all_1308"))
        overlaps.append(build_overlap_table(comparison_id, matrices, canonical_columns=shared_columns))

    source_tables = load_source_vae_detection_tables()
    if source_tables:
        source_vae_detection = pd.concat(source_tables.values(), ignore_index=True)
    else:
        source_vae_detection = pd.DataFrame()
    source_vae_percent = build_source_vae_percent_table(
        source_vae_detection,
        matrices_by_comparison,
        shared_canonical_by_comparison,
    )

    tables = {
        "summary": pd.concat(summaries, ignore_index=True),
        "counts": pd.concat(counts, ignore_index=True),
        "thresholds": pd.concat(thresholds, ignore_index=True),
        "node_curves": pd.concat(node_curves, ignore_index=True),
        "overlap": pd.concat(overlaps, ignore_index=True),
        "source_vae_detection": source_vae_detection,
        "source_vae_percent": source_vae_percent,
    }
    for key, df in tables.items():
        df.to_csv(output_paths[key], index=False, encoding="utf-8")

    manifest = {
        "source_note": (
            "DeepHeal Fig2 pathway-enrichment package rebuilt for internal consistency, fair same-universe "
            "baseline comparison, and copied-package reproducibility."
        ),
        "package_work_dir": ".",
        "fisher_test": {
            "alternative": FISHER_ALTERNATIVE,
            "rationale": "One-sided over-representation is used consistently for target, baseline, reference, and source-VAE recurrence analyses.",
        },
        "fdr": {
            "method": FDR_METHOD,
            "alpha": FDR_ALPHA,
        },
        "top_n": TOP_N_BY_COLLECTION,
        "pathway_universe": {
            "headline_c2": {
                comparison_id: {
                    "description": "Shared KEGG/BioCarta/Reactome canonical pathway columns across target, matched baselines, and 161L reference.",
                    "n_pathways": len(columns),
                }
                for comparison_id, columns in shared_canonical_by_comparison.items()
            },
            "target_c2_cp_panel": {
                "description": "Broader DeepHeal target-only C2 CP panel retained as additional biological analysis, not as a same-universe baseline comparison.",
                "n_pathways": int(matrices_by_comparison["primary_deepheal_257L"]["deepheal_supcon"]["canonical"].shape[1]),
            },
        },
        "comparison_specs": COMPARISON_SPECS,
        "methods": {
            method: {
                "label": method_display(method),
                "tick_label": method_tick(method),
                "role": method_role(method),
                "latent_dim": METHOD_META[method].get("latent_dim"),
                "canonical_matrix": _relative_to_work_dir(Path(METHOD_META[method]["canonical"]))
                if METHOD_META[method].get("canonical")
                else "",
                "oncogenic_matrix": _relative_to_work_dir(Path(METHOD_META[method]["oncogenic"]))
                if METHOD_META[method].get("oncogenic")
                else "",
                "source_model_detection": _relative_to_work_dir(Path(METHOD_META[method]["model_detection"]))
                if METHOD_META[method].get("model_detection")
                else "",
                "source_pathway_detection": _relative_to_work_dir(Path(METHOD_META[method]["pathway_detection"]))
                if METHOD_META[method].get("pathway_detection")
                else "",
            }
            for method in METHOD_ORDER
        },
        "tables": {key: _relative_to_work_dir(path) for key, path in output_paths.items()},
        "style": {
            "dpi": 300,
            "font_family": "Arial, Helvetica, DejaVu Sans fallback",
            "pdf.fonttype": 42,
        },
    }
    (DATA_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tables


def load_or_build_tables(force: bool = False) -> dict[str, pd.DataFrame]:
    return build_analysis_tables(force=force)


def save_figure(fig: plt.Figure, stem: str, out_dir: Path = WORK_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=NATURE_OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=NATURE_OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def _ordered_methods_for_summary(summary: pd.DataFrame, comparison_id: str) -> list[str]:
    present = set(summary.loc[summary["comparison_id"] == comparison_id, "method"].astype(str))
    return [method for method in matrix_methods(comparison_id) if method in present]


def _style_tick_params(
    ax: plt.Axes,
    *,
    x_pad: float = 12.0,
    y_pad: float = 10.0,
    x_labelsize: float | None = None,
    y_labelsize: float | None = None,
) -> None:
    ax.tick_params(axis="x", pad=x_pad, labelsize=x_labelsize or NATURE_FONT_SIZE)
    ax.tick_params(axis="y", pad=y_pad, labelsize=y_labelsize or NATURE_FONT_SIZE)


def _branch_panel_title(comparison_id: str, title: str) -> str:
    return f"{comparison_short_label(comparison_id)}\n{title}"


def _plot_average_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    comparison_id: str,
    collection_ids: list[str],
    *,
    compact_labels: bool = False,
    ylabel: str | None = None,
) -> None:
    methods = _ordered_methods_for_summary(summary, comparison_id)
    sub = summary[
        (summary["comparison_id"] == comparison_id) & summary["collection_id"].isin(collection_ids)
    ].copy()
    ordered_labels = [COLLECTION_LABELS[collection_id] for collection_id in collection_ids]
    sub["collection_label"] = pd.Categorical(sub["collection_label"], ordered_labels, ordered=True)
    sns.barplot(
        data=sub,
        x="collection_label",
        y="avg_sig_pathways_per_latent",
        hue="method_tick",
        hue_order=[method_tick(method) for method in methods],
        palette={method_tick(method): method_color(method) for method in methods},
        edgecolor="none",
        errorbar=None,
        ax=ax,
    )
    if compact_labels:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([BAR_LABELS_COMPACT[collection_id] for collection_id in collection_ids], rotation=90, ha="center")
    else:
        ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or "", fontsize=NATURE_BAR_TICK_SIZE, labelpad=14)
    _style_tick_params(ax, x_pad=8, y_pad=8, x_labelsize=NATURE_BAR_TICK_SIZE, y_labelsize=NATURE_BAR_YTICK_SIZE)
    if ax.legend_:
        ax.legend_.remove()


def _plot_distribution_panel(
    ax: plt.Axes,
    counts: pd.DataFrame,
    comparison_id: str,
    *,
    collection_id: str,
    rng: np.random.Generator,
    compact_ticks: bool = False,
    ylabel: str | None = None,
) -> None:
    methods = matrix_methods(comparison_id)
    sub = counts[(counts["comparison_id"] == comparison_id) & (counts["collection_id"] == collection_id)].copy()
    x_positions = np.arange(len(methods))
    for idx, method in enumerate(methods):
        values = sub.loc[sub["method"] == method, "n_significant_pathways"].to_numpy()
        jitter = rng.uniform(-0.38, 0.38, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=SCATTER_POINT_SIZE,
            color=method_color(method),
            alpha=SCATTER_ALPHA,
            linewidths=0,
            rasterized=True,
        )
        median = np.median(values) if len(values) else np.nan
        ax.plot([idx - 0.24, idx + 0.24], [median, median], color="black", linewidth=MEDIAN_LINEWIDTH)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [method_compact_tick(method) if compact_ticks else method_tick(method) for method in methods],
        rotation=35 if not compact_ticks else 0,
        ha="right" if not compact_ticks else "center",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or "", fontsize=24, labelpad=26)
    _style_tick_params(ax, x_pad=14, y_pad=9, x_labelsize=24, y_labelsize=20)


def _plot_threshold_curves(
    ax: plt.Axes,
    node_curves: pd.DataFrame,
    comparison_id: str,
    *,
    collection_id: str,
    ylabel: str | None = None,
) -> None:
    sub = node_curves[
        (node_curves["comparison_id"] == comparison_id) & (node_curves["collection_id"] == collection_id)
    ].copy()
    for method in matrix_methods(comparison_id):
        d = sub[sub["method"] == method].sort_values("threshold")
        ax.bar(
            d["threshold"],
            d["pct_nodes"],
            width=THRESHOLD_BAR_WIDTH,
            color=method_color(method),
            alpha=THRESHOLD_BAR_ALPHA,
            edgecolor="none",
            zorder=1,
        )
    for method in matrix_methods(comparison_id):
        d = sub[sub["method"] == method].sort_values("threshold")
        ax.plot(
            d["threshold"],
            d["pct_nodes"],
            color=method_color(method),
            marker="o",
            markersize=CURVE_MARKER_SIZE,
            linewidth=CURVE_LINEWIDTH,
            label=method_tick(method),
            zorder=3,
        )
    ax.set_xlabel("Significance threshold\n(-log10 minimum FDR)", fontsize=NATURE_AXIS_LABEL_SIZE, labelpad=20)
    ax.set_ylabel(ylabel or "", fontsize=NATURE_AXIS_LABEL_SIZE, labelpad=32)
    ax.set_xticks(NODE_THRESHOLDS)
    ax.set_ylim(-3, 103)
    _style_tick_params(ax, x_pad=12, y_pad=10, x_labelsize=20, y_labelsize=20)


def plot_average_pathway_coverages(summary: pd.DataFrame, out_dir: Path = WORK_DIR) -> list[Path]:
    apply_nature_style(NATURE_FONT_SIZE)
    fig, axes = plt.subplots(
        len(COMPARISON_SPECS),
        2,
        figsize=FIGURE_SIZES["average_pathway_coverages"],
        sharey=False,
    )
    if len(COMPARISON_SPECS) == 1:
        axes = np.array([axes])

    for row, comparison_id in enumerate(COMPARISON_SPECS):
        for col, (collections, title) in enumerate(
            [
                (["kegg", "biocarta", "reactome"], "KEGG, BioCarta, Reactome"),
                (["canonical_all", "oncogenic_all"], "Canonical and oncogenic"),
            ]
        ):
            ax = axes[row, col]
            _plot_average_bars(
                ax,
                summary,
                comparison_id,
                collections,
                compact_labels=True,
                ylabel="Average captured\npathways per latent" if col == 0 else "",
            )
            ax.set_title(_branch_panel_title(comparison_id, title), fontsize=NATURE_TITLE_SIZE, pad=2)

    add_compact_legend(fig, y=0.025)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.26, top=0.91, hspace=0.92, wspace=0.30)
    return save_figure(fig, "deepheal_average_pathway_coverages", out_dir)


def plot_pathway_coverage_distributions(counts: pd.DataFrame, out_dir: Path = WORK_DIR) -> list[Path]:
    apply_nature_style(NATURE_FONT_SIZE)
    rng = np.random.default_rng(20260526)
    fig, axes = plt.subplots(
        len(COMPARISON_SPECS),
        2,
        figsize=FIGURE_SIZES["pathway_coverage_distributions"],
        sharey=False,
    )
    if len(COMPARISON_SPECS) == 1:
        axes = np.array([axes])
    panels = [("canonical_all", "Canonical pathways"), ("oncogenic_all", "Oncogenic signatures")]
    for row, comparison_id in enumerate(COMPARISON_SPECS):
        for col, (collection_id, title) in enumerate(panels):
            ax = axes[row, col]
            _plot_distribution_panel(
                ax,
                counts,
                comparison_id,
                collection_id=collection_id,
                rng=rng,
                compact_ticks=True,
                ylabel="Captured pathways per latent" if col == 0 else "",
            )
            ax.set_title(_branch_panel_title(comparison_id, title), fontsize=NATURE_TITLE_SIZE, pad=2)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.10, top=0.90, hspace=0.55, wspace=0.22)
    return save_figure(fig, "deepheal_pathway_coverage_distributions", out_dir)


def plot_node_level_pathway_annotations(node_curves: pd.DataFrame, out_dir: Path = WORK_DIR) -> list[Path]:
    apply_nature_style(NATURE_FONT_SIZE)
    fig, axes = plt.subplots(
        len(COMPARISON_SPECS),
        2,
        figsize=FIGURE_SIZES["node_level_pathway_annotations"],
        sharey=True,
    )
    if len(COMPARISON_SPECS) == 1:
        axes = np.array([axes])
    panels = [("canonical_all", "Canonical pathways"), ("oncogenic_all", "Oncogenic signatures")]
    for row, comparison_id in enumerate(COMPARISON_SPECS):
        for col, (collection_id, title) in enumerate(panels):
            ax = axes[row, col]
            _plot_threshold_curves(
                ax,
                node_curves,
                comparison_id,
                collection_id=collection_id,
                ylabel="Latents above threshold (%)" if col == 0 else "",
            )
            ax.set_title(_branch_panel_title(comparison_id, title), fontsize=24, pad=-2)
    add_compact_legend(fig, y=0.025)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.90, hspace=0.78, wspace=0.22)
    return save_figure(fig, "deepheal_node_level_pathway_annotations", out_dir)


def plot_detection_overlap_counts(overlap: pd.DataFrame, out_dir: Path = WORK_DIR) -> list[Path]:
    apply_nature_style(NATURE_FONT_SIZE)
    methods_per_row = max(len(matrix_methods(comparison_id)) - 1 for comparison_id in COMPARISON_SPECS)
    fig, axes = plt.subplots(
        len(COMPARISON_SPECS),
        methods_per_row,
        figsize=FIGURE_SIZES["pathway_detection_overlap"],
        sharey=False,
    )
    if len(COMPARISON_SPECS) == 1:
        axes = np.array([axes])
    for row, comparison_id in enumerate(COMPARISON_SPECS):
        methods = [method for method in matrix_methods(comparison_id) if method != COMPARISON_SPECS[comparison_id]["reference_method"]]
        target_method = str(COMPARISON_SPECS[comparison_id]["target_method"])
        target_color = method_color(target_method)
        for col in range(methods_per_row):
            ax = axes[row, col]
            if col >= len(methods):
                ax.axis("off")
                continue
            method = methods[col]
            d = overlap[(overlap["comparison_id"] == comparison_id) & (overlap["method"] == method)].sort_values("threshold")
            x = np.arange(len(d))
            ax.bar(x - 0.18, d["n_detected_pathways"], width=0.34, color="#d9d9d9", edgecolor="none")
            ax.bar(x + 0.18, d["n_overlap_with_reference"], width=0.34, color=target_color, edgecolor="none")
            ax.set_title(overlap_panel_title(comparison_id, method), fontsize=NATURE_SMALL_FONT_SIZE, pad=0)
            ax.set_xticks(x)
            ax.set_xticklabels(d["threshold"].astype(str).tolist())
            ax.set_xlabel("Min latent count", fontsize=NATURE_SMALL_FONT_SIZE, labelpad=12)
            ax.set_ylabel("Pathways", fontsize=NATURE_SMALL_FONT_SIZE, labelpad=10)
            _style_tick_params(ax, x_pad=8, y_pad=8, x_labelsize=NATURE_SMALL_FONT_SIZE, y_labelsize=NATURE_SMALL_FONT_SIZE)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color="#d9d9d9", markersize=10, label="Detected by method"),
        Line2D([0], [0], marker="s", linestyle="", color=method_color("deepheal_supcon"), markersize=10, label="Overlap in DeepHeal Supervised 257L comparison"),
        Line2D([0], [0], marker="s", linestyle="", color=method_color("deepheal_average"), markersize=10, label="Overlap in DeepHeal Unsupervised 194L comparison"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=NATURE_SMALL_FONT_SIZE)
    fig.suptitle("Canonical pathway detection relative to the labeled 161L VAE reference", y=0.995, fontsize=NATURE_TITLE_SIZE)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    return save_figure(fig, "deepheal_pathway_detection_overlap_vs_vae_reference", out_dir)


def plot_main_composite(tables: dict[str, pd.DataFrame], out_dir: Path = WORK_DIR) -> list[Path]:
    apply_nature_style(NATURE_FONT_SIZE)
    fig = plt.figure(figsize=FIGURE_SIZES["main_composite"])
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.25, 1.05],
        hspace=1.05,
        wspace=0.24,
    )
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)])
    summary = tables["summary"]
    counts = tables["counts"]
    node_curves = tables["node_curves"]
    rng = np.random.default_rng(20260526)
    comparison_ids = list(COMPARISON_SPECS)
    panel_titles = [
        ("a", "Average captured pathways"),
        ("b", "Per-latent captured pathways"),
        ("c", "Annotated latent variables"),
    ]

    for col, comparison_id in enumerate(comparison_ids):
        branch_label = comparison_short_label(comparison_id)
        _plot_average_bars(
            axes[0, col],
            summary,
            comparison_id,
            ["kegg", "biocarta", "reactome", "canonical_all", "oncogenic_all"],
            compact_labels=True,
            ylabel="Average captured\npathways per latent" if col == 0 else "",
        )
        _plot_distribution_panel(
            axes[1, col],
            counts,
            comparison_id,
            collection_id="canonical_all",
            rng=rng,
            compact_ticks=True,
            ylabel="Captured pathways\nper latent" if col == 0 else "",
        )
        _plot_threshold_curves(
            axes[2, col],
            node_curves,
            comparison_id,
            collection_id="canonical_all",
            ylabel="Latents annotated\nby pathway (%)" if col == 0 else "",
        )

        for row, (letter, title) in enumerate(panel_titles):
            axes[row, col].set_title(
                branch_label,
                loc="center",
                fontweight="normal",
                fontsize=NATURE_TITLE_SIZE,
                pad=0,
            )
            if col == 0:
                axes[row, col].text(
                    0.0,
                    1.11,
                    f"{letter}  {title}",
                    transform=axes[row, col].transAxes,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    fontsize=NATURE_PANEL_LABEL_SIZE,
                )

    add_compact_legend(fig, y=0.02)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.95, hspace=1.15, wspace=0.24)
    return save_figure(fig, "deepheal_pathway_enrichment_comparison", out_dir)


def make_all_figures(force_tables: bool = False) -> dict[str, list[Path]]:
    tables = load_or_build_tables(force=force_tables)
    outputs = {
        "average_pathway_coverages": plot_average_pathway_coverages(tables["summary"]),
        "pathway_coverage_distributions": plot_pathway_coverage_distributions(tables["counts"]),
        "node_level_pathway_annotations": plot_node_level_pathway_annotations(tables["node_curves"]),
        "pathway_detection_overlap_vs_vae_reference": plot_detection_overlap_counts(tables["overlap"]),
        "main_composite": plot_main_composite(tables),
    }
    manifest = {key: [_relative_to_work_dir(path) for path in paths] for key, paths in outputs.items()}
    (WORK_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    built = make_all_figures(force_tables=True)
    for name, paths in built.items():
        print(f"{name}:")
        for path in paths:
            print(f"  {path}")
