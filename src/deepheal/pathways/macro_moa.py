#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
import math
import re
import sys
import textwrap
import time
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


ROOT = ARTICLE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepheal_paths import CANONICAL_GMT_PATH

OUT_DIR = ROOT / "results" / "macro_moa"
FIG_DIR = ROOT / "results" / "figures"
GMT_PATH = CANONICAL_GMT_PATH
AVG_DATA_PATH = ROOT / "data" / "raw" / "875_Compounds_Average_Data.csv"
META_PATH = ROOT / "data" / "raw" / "drug_hct116_classification_including_marginal_with_cluster.csv"
CHEMBL_CACHE = OUT_DIR / "antineoplastic_chembl_lookup.csv"

TOP_N = 24
MIN_GENES = 8
MAX_GENES = 350
ALPHA = 0.05
MIN_FIGURE_SIG_GROUPS = 2
MAX_JACCARD_WITHIN_GROUP = 0.72

DB_PREFIXES = ("KEGG_", "BIOCARTA_", "REACTOME_")
CH_EMBL_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={query}"
CH_EMBL_MECHANISM = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={chembl_id}"

MOA_GROUP_ORDER = [
    "CELL CYCLE / DNA DAMAGE",
    "RNA METABOLISM / CHROMATIN",
    "OXPHOS / METABOLISM",
    "RTK / PI3K / MAPK / ADHESION",
    "APOPTOSIS / PROTEOSTASIS",
    "RECEPTOR / HORMONE / ION",
]

MOA_GROUP_COLORS = {
    "CELL CYCLE / DNA DAMAGE": "#d73027",
    "RNA METABOLISM / CHROMATIN": "#756bb1",
    "OXPHOS / METABOLISM": "#1b9e77",
    "RTK / PI3K / MAPK / ADHESION": "#1f78b4",
    "APOPTOSIS / PROTEOSTASIS": "#ef8a62",
    "RECEPTOR / HORMONE / ION": "#636363",
}

PATHWAY_GROUP_KEYWORDS = [
    ("DNA damage / repair", ["DNA", "REPAIR", "FANCONI", "MISMATCH", "HOMOLOG", "DOUBLE STRAND", "ATR", "ATM", "P53", "TELOMERE"]),
    ("Cell cycle / mitosis", ["CELL CYCLE", "MITOT", "MITOSIS", "SPINDLE", "KINETOCHORE", "ANAPHASE", "CYTOKINESIS", "CYCLIN", "G2 M", "G1 S", "AURORA", "PLK1", "APC", "CENTROSOME", "PROMETAPHASE", "M G1"]),
    ("Replication stress / checkpoint", ["REPLICATION STRESS", "S PHASE", "ORIGIN", "FORK", "CHK", "CHECKPOINT", "CDC6", "CDT1", "ORC", "PRE REPLICATIVE", "E2F"]),
    ("Apoptosis / death signaling", ["APOPT", "DEATH", "CASP", "TRAIL", "FAS", "TNFR", "BCL", "SURVIV"]),
    ("Proteostasis / ER stress / UPR", ["UNFOLDED", "PROTEASOM", "ENDOPLASMIC", "ER ", "UPR", "AUTOPHAG", "LYSOSOM", "HEAT SHOCK", "CHAPERONE", "XBP1", "CALNEXIN", "CALRETICULIN"]),
    ("Metabolism / nucleotide / redox", ["GLYCOL", "TCA", "MITOCH", "OXID", "FOLATE", "PURINE", "PYRIMID", "NUCLEOTIDE", "REDOX", "PENTOSE", "AMINO ACID", "FATTY ACID", "CHOLESTEROL", "PHOSPHOLIPID", "SPHINGOLIPID", "GLUTATHIONE", "LIPID", "METABOL"]),
    ("Inflammatory / cytokine signaling", ["INTERFERON", "INTERLEUKIN", "IL ", "IL-", "CYTOKINE", "CHEMOKINE", "JAK", "STAT", "NFKB", "TGF", "INFLAMM"]),
    ("Chromatin / RNA / translation", ["MRNA", "RIBOSOM", "TRANSLAT", "SPLIC", "CHROMAT", "HISTONE", "PRC2", "EZH2", "TRANSCRIPTION", "RNA POL", "MICRORNA"]),
]

PATHWAY_GROUP_COLORS = {
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

FIGURE_PATHWAY_GROUPS = [
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
    "Cell cycle / mitosis": ["CELL CYCLE", "MITOT", "MITOSIS", "CENTROSOME", "PROMETAPHASE", "ANAPHASE", "SPINDLE", "G2", "G1", "CYCLIN", "CDC20", "APC", "NUMA", "NEK2"],
    "Replication stress / checkpoint": ["S PHASE", "CDC6", "CDT1", "ORC", "E2F", "PRE REPLICATIVE", "REPLICATION STRESS", "ATR", "RB1"],
    "Apoptosis / death signaling": ["APOPT", "DEATH", "CASP", "TRAIL", "FAS", "TNFR", "BCL", "SURVIV"],
    "Proteostasis / ER stress / UPR": ["UNFOLDED", "XBP1", "CHAPERONE", "PROTEAS", "LYSOSOM", "AUTOPHAG", "GLYCAN", "CALNEXIN", "CALRETICULIN", "UPR"],
    "Metabolism / nucleotide / redox": ["GLYCOL", "TCA", "CITRATE", "PURINE", "PYRIMIDINE", "NUCLEOTIDE", "GLUTATHIONE", "FATTY ACID", "PHOSPHOLIPID", "GLYCEROPHOSPHOLIPID", "SPHINGOLIPID", "OXIDATION", "MITOCHONDRIAL", "LIPID"],
    "Chromatin / RNA / translation": ["MRNA", "RIBOSOM", "TRANSLAT", "TRANSCRIPTION", "RNA POL", "POST TRANSLATIONAL", "MICRORNA", "REGULATORY RNA"],
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
    "PLATELET", "VIRAL", "INFLUENZA", "HIV", "HOST", "NEP_NS2", "ANTIVIRAL", "RIG_I", "MDA5",
    "HELICOBACTER", "NEUROTRANSMITTER", "GLUTAMATE", "DOPAMINE", "OLFACTORY", "ODORANT",
    "PHOTOTRANSDUCTION", "ALLOGRAFT", "ASTHMA", "LEISHMANIA", "MALARIA", "TRYPANOSOMA",
    "LIMONENE", "PINENE", "BILE ACID", "DIABETES", "NATURAL KILLER",
]

TARGET_GROUPS = {
    "CELL CYCLE / DNA DAMAGE": {
        "CDK1", "CDK2", "CDK4", "CDK6", "CDK7", "CDK9", "AURKA", "AURKB", "PLK1", "WEE1", "WEE2",
        "CHEK1", "CHEK2", "ATR", "ATM", "TTK", "CENPE", "KIF11", "TOP1", "TOP2A", "PARP1", "PARP2",
        "RAD51", "PRKDC",
    },
    "RNA METABOLISM / CHROMATIN": {
        "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8", "HDAC9", "DNMT1", "DOT1L", "KDM1A", "EHMT1",
        "EHMT2", "EZH2", "BRD4", "SSRP1", "EWSR1", "XPO1", "CLK1", "CLK4", "USP1", "USP7",
    },
    "OXPHOS / METABOLISM": {
        "GLS", "NAMPT", "IMPDH2", "DHODH", "DHFR", "HMGCR", "IDH2", "HK2", "TYMP", "TYMS", "HPRT1",
        "CA9", "CA2", "CA7", "CA12", "CYP1A1", "CYP1A2", "CYP1B1", "CYP2B6", "CYP2C9", "CYP17A1",
        "AKR1C3", "TPH1", "METAP2", "PLA2G7", "CETP", "LSS", "SQLE", "ACACA", "ACACB",
    },
    "RTK / PI3K / MAPK / ADHESION": {
        "EGFR", "ERBB2", "MET", "ALK", "FGFR1", "FGFR2", "FLT1", "FLT3", "FLT4", "KDR", "KIT", "ABL1",
        "SRC", "FYN", "LCK", "PIK3CA", "PIK3CB", "PIK3CD", "PDPK1", "PTK2", "BRAF", "RAF1", "MAP2K1",
        "JAK1", "JAK2", "SYK", "PRKCB", "SGK1", "CSNK2A1", "LIMK1", "PAK4", "FNTA", "IGF1R", "AKT2",
        "GSK3B", "MTOR", "MERTK", "MMP1",
    },
    "APOPTOSIS / PROTEOSTASIS": {
        "BCL2", "BCL2L1", "MCL1", "XIAP", "BIRC2", "BIRC3", "MDM2", "PSMB5", "PSMB8", "NAE1", "HSP90AA1",
        "HSP90B1", "RIPK1",
    },
}

MANUAL_OVERRIDES = {
    "Celastrol": "APOPTOSIS / PROTEOSTASIS",
    "Dihydroartemisinin": "OXPHOS / METABOLISM",
    "Pyrimethamine": "OXPHOS / METABOLISM",
}

MOA_TEXT_RULES = [
    (
        "CELL CYCLE / DNA DAMAGE",
        [
            "DNA TOPOISOMERASE",
            "TOP1",
            "TOP2",
            "PARP",
            "ATR",
            "ATM",
            "CHK1",
            "CHK2",
            "CHEK1",
            "CHEK2",
            "CDK",
            "CYCLIN-DEPENDENT KINASE",
            "AURORA",
            "AURKB",
            "AURKA",
            "PLK1",
            "WEE1",
            "TTK",
            "CENPE",
            "KIF11",
            "KINESIN-LIKE PROTEIN 1",
            "MITOTIC",
            "MITOSIS",
            "SPINDLE",
            "CELL CYCLE",
            "DNA DAMAGE",
            "DNA REPLICATION",
            "REPLICATION",
            "MICROTUBULE",
            "TUBULIN",
        ],
    ),
    (
        "RNA METABOLISM / CHROMATIN",
        [
            "HDAC",
            "HISTONE DEACETYLASE",
            "HISTONE",
            "DNMT",
            "LSD1",
            "KDM",
            "DOT1L",
            "EZH2",
            "BRD",
            "BET",
            "CHROMATIN",
            "SMYD2",
            "EIF4E",
            "RNA",
            "MRNA",
            "RIBOSOME",
            "RIBOSOMAL",
            "TRANSLATION",
            "SPLIC",
            "RNA POLYMERASE",
            "EXPORTIN",
            "XPO1",
            "ERN1",
        ],
    ),
    (
        "OXPHOS / METABOLISM",
        [
            "DHODH",
            "GLUTAMINASE",
            "GLS",
            "NAMPT",
            "IMPDH",
            "DHFR",
            "HPRT1",
            "IDH",
            "AKR1C3",
            "CARBONIC ANHYDRASE",
            "CA9",
            "CA12",
            "CA2",
            "CA7",
            "CYP",
            "PHOSPHOLIPASE",
            "CETP",
            "HMGCR",
            "SQLE",
            "LSS",
            "ACACA",
            "ACACB",
            "METABOL",
            "OXIDATIVE",
            "OXPHOS",
            "MITOCH",
            "RESPIRATORY CHAIN",
            "DIHYDROOROTATE DEHYDROGENASE",
            "NICOTINAMIDE PHOSPHORIBOSYLTRANSFERASE",
        ],
    ),
    (
        "RTK / PI3K / MAPK / ADHESION",
        [
            "TYROSINE-PROTEIN KINASE",
            "TYROSINE KINASE",
            "RECEPTOR TYROSINE KINASE",
            "KINASE RECEPTOR",
            "EGFR",
            "ERBB",
            "FGFR",
            "FLT",
            "FLT3",
            "VEGFR",
            "MET",
            "ALK",
            "ABL",
            "KIT",
            "KDR",
            "SRC",
            "JAK",
            "PI3-KINASE",
            "PI3K",
            "PIK3",
            "AKT",
            "MTOR",
            "RAF",
            "MEK",
            "MAPK",
            "FAK",
            "FOCAL ADHESION KINASE",
            "PTK2",
            "IGF1R",
            "PDGFR",
            "SYK",
            "LCK",
            "MERTK",
            "PKC",
            "PROTEIN KINASE C",
            "ADHESION",
            "INTEGRIN",
        ],
    ),
    (
        "APOPTOSIS / PROTEOSTASIS",
        [
            "BCL2",
            "BCL-2",
            "MCL1",
            "MDM2",
            "XIAP",
            "BIRC",
            "APOPT",
            "DEATH",
            "PROTEASOME",
            "PSMB",
            "HSP90",
            "NEDD8",
            "NAE1",
            "RIPK1",
            "UNFOLDED",
            "UPR",
            "CHAPERONE",
        ],
    ),
    (
        "RECEPTOR / HORMONE / ION",
        [
            "RECEPTOR",
            "AGONIST",
            "ANTAGONIST",
            "CHANNEL",
            "ION",
            "ADRENERGIC",
            "MINERALOCORTICOID",
            "RETINOID",
            "ENDOTHELIN",
            "SEROTONIN",
            "DOPAMINE",
            "MUSCARINIC",
            "HISTAMINE",
            "GABAA",
            "GABA",
            "OPIOID",
            "P2X",
            "SCN",
            "TRP",
            "TRPA1",
        ],
    ),
]


def norm_text(value: object) -> str:
    return str(value).strip().replace("\xa0", " ")


def normalize_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def wrap_label(text: str, width: int = 30) -> str:
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


def pretty_pathway_name(pathway: str) -> str:
    raw = str(pathway)
    prefix = "Pathway"
    for db in DB_PREFIXES:
        if raw.startswith(db):
            prefix = db[:-1].title()
            raw = raw[len(db):]
            break
    return f"{prefix}: {raw.replace('_', ' ').title()}"


def pathway_text(pathway: str) -> str:
    return str(pathway).replace("_", " ").upper()


def pathway_group(pathway: str) -> str:
    text = pathway_text(pathway)
    for group, keywords in PATHWAY_GROUP_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return group
    return "Other"


def pathway_blacklisted(pathway: str) -> bool:
    text = pathway_text(pathway)
    return any(keyword in text for keyword in FIGURE_BLACKLIST)


def figure_keyword_match(pathway: str, group: str) -> bool:
    text = pathway_text(pathway)
    return any(keyword in text for keyword in FIGURE_GROUP_KEYWORDS.get(group, []))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
    meta.columns = [c.replace("\ufeff", "") for c in meta.columns]
    meta["Sample_ID"] = meta["Sample_ID"].map(norm_text)
    meta["Drug_Name"] = meta["Drug_Name"].map(norm_text)
    meta["Drug_Class"] = meta["Drug_Class"].map(norm_text)

    sample_ids = [c for c in avg.columns[2:] if c in set(meta["Sample_ID"])]
    gene_matrix = avg[["Gene Name", *sample_ids]].copy()
    gene_matrix = gene_matrix.groupby("Gene Name", as_index=True).mean(numeric_only=True)
    gene_matrix = gene_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    meta = meta.set_index("Sample_ID").loc[sample_ids].reset_index()
    return gene_matrix, meta


def build_local_target_lookup() -> dict[str, str]:
    avg = pd.read_csv(AVG_DATA_PATH, nrows=0, encoding="latin-1")
    lookup: dict[str, str] = {}
    for col in avg.columns.tolist()[2:]:
        drug, target = col.split("_", 1)
        lookup[norm_text(drug)] = norm_text(target)
    return lookup


def query_chembl(drug_names: list[str]) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict[str, object]] = []
    for idx, drug in enumerate(drug_names, 1):
        row: dict[str, object] = {"Drug_Name": drug}
        try:
            search_resp = session.get(CH_EMBL_SEARCH.format(query=quote(drug)), timeout=25)
            search_resp.raise_for_status()
            molecules = search_resp.json().get("molecules", [])
        except Exception as exc:
            row["status"] = "search_error"
            row["error"] = str(exc)
            rows.append(row)
            continue

        if not molecules:
            row["status"] = "no_hit"
            row["n_hits"] = 0
            rows.append(row)
            continue

        best = molecules[0]
        chembl_id = (best.get("molecule_hierarchy") or {}).get("parent_chembl_id") or best.get("molecule_chembl_id")
        row.update(
            {
                "status": "hit",
                "n_hits": len(molecules),
                "chembl_id": chembl_id,
                "pref_name": best.get("pref_name"),
                "max_phase": best.get("max_phase"),
                "molecule_type": best.get("molecule_type"),
            }
        )
        try:
            mech_resp = session.get(CH_EMBL_MECHANISM.format(chembl_id=quote(chembl_id)), timeout=25)
            mech_resp.raise_for_status()
            mechanisms = mech_resp.json().get("mechanisms", [])
        except Exception as exc:
            mechanisms = []
            row["mechanism_error"] = str(exc)

        row["n_mechanisms"] = len(mechanisms)
        if mechanisms:
            first = mechanisms[0]
            row["mechanism_of_action"] = first.get("mechanism_of_action")
            row["action_type"] = first.get("action_type")
            row["target_pref_name"] = first.get("target_pref_name")
            row["target_chembl_id"] = first.get("target_chembl_id")
            row["mechanism_comment"] = first.get("mechanism_comment")
        rows.append(row)
        if idx % 20 == 0:
            pd.DataFrame(rows).to_csv(CHEMBL_CACHE, index=False)
            time.sleep(0.2)

    chembl_df = pd.DataFrame(rows)
    chembl_df.to_csv(CHEMBL_CACHE, index=False)
    return chembl_df


def load_or_query_chembl(drug_names: list[str], refresh: bool = False) -> pd.DataFrame:
    if CHEMBL_CACHE.exists() and not refresh:
        chembl = pd.read_csv(CHEMBL_CACHE)
    else:
        chembl = query_chembl(drug_names)
    chembl["Drug_Name"] = chembl["Drug_Name"].map(norm_text)
    chembl["chembl_exact_name_match"] = chembl.apply(
        lambda r: normalize_name(r.get("pref_name", "")) == normalize_name(r["Drug_Name"]),
        axis=1,
    )
    return chembl


def parse_targets(target_string: object) -> set[str]:
    raw = str(target_string)
    if raw == "nan":
        return set()
    return {norm_text(x).upper() for x in raw.split(";") if norm_text(x)}


def evidence_summary(row: pd.Series) -> str:
    pieces = [f"Local target: {row['local_target']}"]
    if pd.notna(row.get("mechanism_of_action")):
        pieces.append(f"ChEMBL MoA: {row['mechanism_of_action']}")
    if pd.notna(row.get("action_type")):
        pieces.append(f"Action: {row['action_type']}")
    elif pd.notna(row.get("pref_name")):
        pieces.append(f"ChEMBL name: {row['pref_name']}")
    return " | ".join(pieces)


def has_any_keyword(text: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        pattern = rf"(?<![A-Z0-9]){re.escape(keyword)}(?![A-Z0-9])"
        if re.search(pattern, text):
            return True
    return False


def build_moa_evidence_text(row: pd.Series) -> str:
    fields = [
        row.get("local_target", ""),
        row.get("mechanism_of_action", ""),
        row.get("target_pref_name", ""),
        row.get("pref_name", ""),
        row.get("action_type", ""),
    ]
    return " | ".join(norm_text(x).upper() for x in fields if pd.notna(x))


def classify_macro_group(row: pd.Series) -> tuple[str, str]:
    if row["Drug_Name"] in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[row["Drug_Name"]], "manual_override"

    evidence_text = build_moa_evidence_text(row)
    for group, keywords in MOA_TEXT_RULES:
        if group == "RECEPTOR / HORMONE / ION":
            continue
        if has_any_keyword(evidence_text, keywords):
            return group, "combined_moa_text_rule"

    targets = parse_targets(row["local_target"])
    for group in MOA_GROUP_ORDER:
        if targets & TARGET_GROUPS.get(group, set()):
            return group, "local_target_rule"

    receptor_keywords = next(keywords for group, keywords in MOA_TEXT_RULES if group == "RECEPTOR / HORMONE / ION")
    if has_any_keyword(evidence_text, receptor_keywords):
        return "RECEPTOR / HORMONE / ION", "combined_moa_text_rule"

    return "RECEPTOR / HORMONE / ION", "default_other"


def build_annotation_table(refresh_chembl: bool = False) -> pd.DataFrame:
    meta = pd.read_csv(META_PATH)
    meta.columns = [c.replace("\ufeff", "") for c in meta.columns]
    drug_names = sorted(meta.loc[meta["Drug_Class"].eq("Antineoplastic"), "Drug_Name"].drop_duplicates().map(norm_text).tolist())
    local_lookup = build_local_target_lookup()
    chembl = load_or_query_chembl(drug_names, refresh=refresh_chembl)

    ann = pd.DataFrame({"Drug_Name": drug_names})
    ann["local_target"] = ann["Drug_Name"].map(local_lookup)
    ann = ann.merge(chembl, on="Drug_Name", how="left")

    groups = ann.apply(classify_macro_group, axis=1, result_type="expand")
    ann["macro_moa_group"] = groups[0]
    ann["classification_basis"] = groups[1]
    ann["evidence_summary"] = ann.apply(evidence_summary, axis=1)
    return ann


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


def group_tests(score_df: pd.DataFrame, sample_meta: pd.DataFrame, group_col: str) -> pd.DataFrame:
    subset_meta = sample_meta.loc[sample_meta[group_col].notna()].copy()
    rows: list[dict[str, object]] = []
    present_groups = [g for g in MOA_GROUP_ORDER if g in set(subset_meta[group_col])]
    for group in present_groups:
        group_ids = subset_meta.loc[subset_meta[group_col].eq(group), "Sample_ID"].tolist()
        group_scores = score_df.loc[group_ids]
        pvals = []
        means = []
        for pathway in score_df.columns:
            values = group_scores[pathway].to_numpy(dtype=float)
            stat = ttest_1samp(values, 0.0, nan_policy="omit")
            pval = float(stat.pvalue) if np.isfinite(stat.pvalue) else 1.0
            pvals.append(max(pval, 1e-300))
            means.append(float(np.nanmean(values)))
        fdr = multipletests(pvals, alpha=ALPHA, method="fdr_bh")[1]
        for pathway, mean_score, pval, qval in zip(score_df.columns.tolist(), means, pvals, fdr.tolist()):
            rows.append(
                {
                    "macro_moa_group": group,
                    "n_drugs": int(len(group_ids)),
                    "pathway": pathway,
                    "mean_score": float(mean_score),
                    "p_value": float(pval),
                    "fdr": float(qval),
                    "direction": "Activated" if mean_score > 0 else ("Suppressed" if mean_score < 0 else "Neutral"),
                    "signed_log10_p": float(np.sign(mean_score) * -math.log10(max(pval, 1e-300))),
                }
            )
    return pd.DataFrame(rows)


def summarize_pathways(anti_group_df: pd.DataFrame, anti_score_df: pd.DataFrame, non_score_df: pd.DataFrame) -> pd.DataFrame:
    anti_abs_mean = anti_score_df.abs().mean(axis=0)
    non_abs_mean = non_score_df.abs().mean(axis=0)
    n_groups = anti_group_df["macro_moa_group"].nunique()
    rows = []
    for pathway, group in anti_group_df.groupby("pathway"):
        sig = group.loc[group["fdr"] < ALPHA].copy()
        n_sig = int(len(sig))
        avg_strength = float((-np.log10(sig["p_value"].clip(lower=1e-300))).mean()) if n_sig else 0.0
        consistency = float(abs(np.sign(sig["mean_score"]).mean())) if n_sig else 0.0
        dominant_direction = "Mixed"
        if n_sig:
            dominant_direction = "Activated" if sig["mean_score"].mean() > 0 else "Suppressed"
        anti_margin = float(max(0.0, anti_abs_mean[pathway] - non_abs_mean.get(pathway, 0.0)))
        rows.append(
            {
                "pathway": pathway,
                "pathway_label": pretty_pathway_name(pathway),
                "functional_group": pathway_group(pathway),
                "n_sig_groups": n_sig,
                "frac_sig_groups": float(n_sig / max(1, n_groups)),
                "avg_sig_neglog10_p": avg_strength,
                "direction_consistency": consistency,
                "dominant_direction": dominant_direction,
                "anti_abs_score_mean": float(anti_abs_mean[pathway]),
                "non_abs_score_mean": float(non_abs_mean.get(pathway, 0.0)),
                "anti_vs_non_margin": anti_margin,
            }
        )
    summary = pd.DataFrame(rows)
    for col in ["n_sig_groups", "avg_sig_neglog10_p", "direction_consistency", "anti_vs_non_margin"]:
        summary[f"{col}_pct"] = summary[col].rank(pct=True)
    summary["pan_moa_character_score"] = (
        0.40 * summary["n_sig_groups_pct"]
        + 0.25 * summary["avg_sig_neglog10_p_pct"]
        + 0.20 * summary["direction_consistency_pct"]
        + 0.15 * summary["anti_vs_non_margin_pct"]
    )
    summary["figure_blacklisted"] = summary["pathway"].map(pathway_blacklisted)
    summary["figure_keyword_match"] = summary.apply(
        lambda r: figure_keyword_match(r["pathway"], r["functional_group"]) if r["functional_group"] in FIGURE_PATHWAY_GROUPS else False,
        axis=1,
    )
    summary["figure_candidate_prequota"] = (
        summary["functional_group"].isin(FIGURE_PATHWAY_GROUPS)
        & summary["n_sig_groups"].ge(MIN_FIGURE_SIG_GROUPS)
        & ~summary["figure_blacklisted"]
        & summary["figure_keyword_match"]
    )
    summary = summary.sort_values(
        ["n_sig_groups", "avg_sig_neglog10_p", "pan_moa_character_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def pathway_order_for_figure(summary_df: pd.DataFrame, pathway_gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    candidates = summary_df.loc[summary_df["figure_candidate_prequota"]].copy()
    selected_rows: list[pd.Series] = []
    selected_paths: set[str] = set()
    selected_by_group: dict[str, list[str]] = {group: [] for group in FIGURE_PATHWAY_GROUPS}
    for group in FIGURE_PATHWAY_GROUPS:
        group_df = candidates.loc[candidates["functional_group"].eq(group)].sort_values(
            ["n_sig_groups", "avg_sig_neglog10_p", "pan_moa_character_score"],
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
            ["n_sig_groups", "avg_sig_neglog10_p", "pan_moa_character_score"],
            ascending=[False, False, False],
        )
        for _, row in fallback_df.iterrows():
            group = row["functional_group"]
            genes = pathway_gene_sets.get(row["pathway"], set())
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
        top.groupby("functional_group", observed=False)["pan_moa_character_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    top["functional_group"] = pd.Categorical(top["functional_group"], categories=group_rank["functional_group"].tolist(), ordered=True)
    top = top.sort_values(
        ["functional_group", "pan_moa_character_score", "n_sig_groups", "avg_sig_neglog10_p"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return top


def moa_group_summary(annotation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in MOA_GROUP_ORDER:
        sub = annotation_df.loc[annotation_df["macro_moa_group"].eq(group)].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "macro_moa_group": group,
                "n_drugs": int(len(sub)),
                "example_drugs": "; ".join(sub["Drug_Name"].tolist()[:8]),
                "n_with_chembl_moa": int(sub["mechanism_of_action"].notna().sum()),
                "n_manual_or_target_rule": int(sub["classification_basis"].isin(["manual_override", "local_target_rule"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_group_counts(group_info: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8})
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    plot_df = group_info.copy()
    plot_df["wrapped_group"] = plot_df["macro_moa_group"].map(lambda x: wrap_label(x, 18))
    y = np.arange(len(plot_df))
    colors = [MOA_GROUP_COLORS[g] for g in plot_df["macro_moa_group"]]
    ax.barh(y, plot_df["n_drugs"], color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["wrapped_group"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of antineoplastic drugs")
    ax.set_title("Antineoplastic drugs grouped by macro mechanism of action", loc="left", fontweight="bold")
    for yi, count in zip(y, plot_df["n_drugs"]):
        ax.text(
            count + 0.45,
            yi,
            f"n={int(count)}",
            va="center",
            ha="left",
            fontsize=7.4,
            color="#374151",
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"deepheal_antineoplastic_macro_moa_groups.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_main_figure(top_df: pd.DataFrame, group_df: pd.DataFrame, group_info: pd.DataFrame) -> None:
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

    present_groups = [g for g in MOA_GROUP_ORDER if g in set(group_info["macro_moa_group"])]
    group_labels = [
        f"{wrap_group_label(g)}\n(n={int(group_info.loc[group_info['macro_moa_group'].eq(g), 'n_drugs'].iloc[0])})"
        for g in present_groups
    ]
    heat = (
        group_df.loc[group_df["pathway"].isin(top_df["pathway"])]
        .pivot(index="pathway", columns="macro_moa_group", values="signed_log10_p")
        .reindex(index=top_df["pathway"], columns=present_groups)
    )
    stars = (
        group_df.loc[group_df["pathway"].isin(top_df["pathway"])]
        .assign(stars=lambda d: d["p_value"].map(significance_stars))
        .pivot(index="pathway", columns="macro_moa_group", values="stars")
        .reindex(index=top_df["pathway"], columns=present_groups)
        .fillna("")
    )

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[4.3, 1.2, 1.3, 4.8, 1.5], wspace=0.08)
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
    for group, sub_df in top_df.groupby("functional_group", sort=False, observed=False):
        idxs = sub_df.index.to_list()
        start, end = min(idxs), max(idxs) + 1
        group_spans.append((group, start, end))
        ax_labels.add_patch(
            plt.Rectangle((0.0, start + 0.02), 0.035, end - start - 0.04, color=PATHWAY_GROUP_COLORS.get(group, "#8c8c8c"), alpha=0.95)
        )
        ax_labels.text(
            0.06,
            (start + end) / 2,
            wrap_group_label(group),
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            color=PATHWAY_GROUP_COLORS.get(group, "#333333"),
        )
    for idx, row in top_df.iterrows():
        ax_labels.text(0.40, row_centers[idx], row["wrapped_label"], va="center", ha="left", fontsize=7.4, color="#1f2933")

    ax_a.barh(row_centers, top_df["n_sig_groups"], height=0.78, color="#5f6b7a", edgecolor="white", linewidth=0.6)
    ax_a.set_title("i\nSignificant\nMoA groups", fontweight="bold")
    ax_a.set_xlabel("Count")
    ax_a.set_xlim(0, max(6, int(top_df["n_sig_groups"].max()) + 1))
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
        xticklabels=group_labels,
        yticklabels=False,
    )
    set_vertical_colorbar_title(heatmap, "Signed -log10(P)", width=10)
    ax_c.set_ylim(n_rows, 0)
    ax_c.grid(False)
    ax_c.set_title("iii\nMacro-MoA signed response", fontweight="bold")
    ax_c.set_xlabel("Antineoplastic macro MoA groups")
    ax_c.set_ylabel("")
    for i in range(stars.shape[0]):
        for j in range(stars.shape[1]):
            star = stars.iat[i, j]
            if star:
                ax_c.text(j + 0.5, i + 0.5, star, ha="center", va="center", fontsize=6.3, color="black")

    bar_colors = [PATHWAY_GROUP_COLORS.get(group, "#8c8c8c") for group in top_df["functional_group"]]
    ax_d.barh(row_centers, top_df["pan_moa_character_score"] * 100.0, height=0.78, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax_d.set_title("iv\nPan-MoA\ncharacter", fontweight="bold")
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
        fig.savefig(FIG_DIR / f"deepheal_antineoplastic_macro_moa_pathway_response.{ext}", bbox_inches="tight")
    plt.close(fig)


def main(refresh_chembl: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    annotation_df = build_annotation_table(refresh_chembl=refresh_chembl).sort_values(["macro_moa_group", "Drug_Name"]).reset_index(drop=True)
    annotation_df.to_csv(OUT_DIR / "antineoplastic_macro_moa_annotations.csv", index=False)

    group_info = moa_group_summary(annotation_df)
    group_info.to_csv(OUT_DIR / "antineoplastic_macro_moa_group_summary.csv", index=False)
    plot_group_counts(group_info)

    pathways = load_canonical_pathways()
    gene_matrix, sample_meta = load_gene_matrix()
    sample_meta = sample_meta.merge(annotation_df[["Drug_Name", "macro_moa_group"]], on="Drug_Name", how="left")

    kept_names, membership, sizes = build_membership_matrix(pathways, gene_matrix.index)
    genes_in_data = set(gene_matrix.index.tolist())
    pathway_gene_sets = {name: set(pathways[name]) & genes_in_data for name in kept_names}

    matrix = gene_matrix[sample_meta["Sample_ID"].tolist()].T.to_numpy(dtype=np.float32)
    score_matrix = compute_pathway_scores(matrix, membership, sizes)
    score_df = pd.DataFrame(score_matrix, index=sample_meta["Sample_ID"].tolist(), columns=kept_names)

    anti_ids = sample_meta.loc[sample_meta["Drug_Class"].eq("Antineoplastic"), "Sample_ID"].tolist()
    non_ids = sample_meta.loc[sample_meta["Drug_Class"].eq("Non-Antineoplastic"), "Sample_ID"].tolist()
    anti_score_df = score_df.loc[anti_ids]
    non_score_df = score_df.loc[non_ids]

    group_df = group_tests(score_df, sample_meta, "macro_moa_group")
    group_df.to_csv(OUT_DIR / "antineoplastic_macro_moa_signed_responses.csv", index=False)

    summary_df = summarize_pathways(group_df, anti_score_df, non_score_df)
    summary_df.to_csv(OUT_DIR / "antineoplastic_macro_moa_pathway_summary_all.csv", index=False)

    top_df = pathway_order_for_figure(summary_df, pathway_gene_sets)
    top_df.to_csv(OUT_DIR / "antineoplastic_macro_moa_pathway_summary_top_for_figure.csv", index=False)

    plot_main_figure(top_df, group_df, group_info)

    summary_payload = {
        "n_antineoplastic_drugs": int(len(annotation_df)),
        "n_macro_moa_groups": int(group_info.shape[0]),
        "group_counts": group_info[["macro_moa_group", "n_drugs", "n_with_chembl_moa"]].to_dict(orient="records"),
        "top_pathways_for_figure": top_df[
            ["pathway", "functional_group", "n_sig_groups", "avg_sig_neglog10_p", "dominant_direction", "pan_moa_character_score"]
        ].to_dict(orient="records"),
        "group_count_figure": str(FIG_DIR / "deepheal_antineoplastic_macro_moa_groups.png"),
        "main_figure": str(FIG_DIR / "deepheal_antineoplastic_macro_moa_pathway_response.png"),
    }
    (OUT_DIR / "antineoplastic_macro_moa_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-chembl", action="store_true", help="Refresh ChEMBL lookup cache before grouping.")
    args = parser.parse_args()
    main(refresh_chembl=args.refresh_chembl)
