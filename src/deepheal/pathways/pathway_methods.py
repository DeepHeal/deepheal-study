from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import urllib.request

from pathlib import Path

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pandas as pd

from scipy.stats import fisher_exact

TRIAD_PREFIXES = ("KEGG_", "BIOCARTA_", "REACTOME_")

GMT_URLS = {
    "c2.cp.v6.2.symbols.gmt": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/6.2/c2.cp.v6.2.symbols.gmt",
    "c6.all.v6.2.symbols.gmt": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/6.2/c6.all.v6.2.symbols.gmt",
}

def ensure_gmt(cache_dir: Path, filename: str) -> Path:
    out_path = cache_dir / filename
    if out_path.exists():
        return out_path
    url = GMT_URLS[filename]
    urllib.request.urlretrieve(url, out_path)
    return out_path

def normalize_id(value: object) -> str:
    return str(value).replace("\xa0", " ").strip()

def load_universe_gene_names(root: Path) -> List[str]:
    matrix = pd.read_csv(root / "875_Compounds_Average_Data.csv", encoding="latin1")
    gene_names = matrix["Gene Name"].astype(str).map(normalize_id).tolist()
    return gene_names

def parse_gmt(gmt_path: Path, collection: str) -> Dict[str, set]:
    pathways: Dict[str, set] = {}
    with open(gmt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            if collection == "canonical" and not name.startswith(TRIAD_PREFIXES):
                continue
            genes = {gene.strip() for gene in parts[2:] if gene.strip()}
            if genes:
                pathways[name] = genes
    return pathways

def bh_adjust(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out

def run_fisher_enrichment(
    method_name: str,
    node_sets: Dict[str, set],
    pathways: Dict[str, set],
    universe_genes: Iterable[str],
) -> pd.DataFrame:
    universe = {gene for gene in universe_genes if gene and gene.lower() != "nan"}
    use_pathways = {name: (members & universe) for name, members in pathways.items()}
    use_pathways = {name: members for name, members in use_pathways.items() if len(members) >= 5}
    ordered_nodes = list(node_sets.keys())
    ordered_pathways = sorted(use_pathways)

    pval_df = pd.DataFrame(index=ordered_nodes, columns=ordered_pathways, dtype=float)
    universe_size = len(universe)
    for node_name in ordered_nodes:
        top_genes = {g for g in node_sets[node_name] if g in universe}
        n_top = len(top_genes)
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
    for pathway_name in ordered_pathways:
        fdr_df[pathway_name] = bh_adjust(fdr_df[pathway_name].to_numpy(dtype=float))
    return fdr_df.reset_index().rename(columns={"index": "node"})
