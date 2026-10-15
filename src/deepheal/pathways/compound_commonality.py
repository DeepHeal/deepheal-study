#!/usr/bin/env python
from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import io
import json
import math
import re
import sys
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact


ROOT = ARTICLE_ROOT
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / 'data/external/deepprofile_commonality'
TABLE_DIR = ROOT / 'tables/deepprofile_commonality'
FIG_DIR = ROOT / 'figures/deepprofile_commonality'
RESULT_DIR = ROOT / 'results/deepprofile_commonality'

ATC_PATH = RAW_DIR / "875_Drugs_Metainformation_ATC_Classfication.xlsx"
ATC_REQUESTED_PATH = RAW_DIR / "875_Drugs_Metainformation_ATC_Classification.xlsx"
CLASS_PATH = RAW_DIR / "drug_hct116_classification_including_marginal_with_cluster.csv"
DECISION_PATH = RAW_DIR / "875_Drug_Antineoplastic_recomputed.csv"
AVG_DATA_PATH = RAW_DIR / "875_Compounds_Average_Data.csv"
SUPCON_WEIGHTS_PATH = ROOT / "results" / "supcon" / "supcon_gene_weights.csv"
GMT_PATH = ROOT / "data" / "pathways" / "gmt_cache" / "c2.cp.v6.2.symbols.gmt"
C6_GMT_PATH = ROOT / "data" / "pathways" / "gmt_cache" / "c6.all.v6.2.symbols.gmt"
STRING_API_PATH = ARTICLE_ROOT / "src/deepheal/string_api.py"

CSPA_URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0121314.s002"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
ENRICHR_LIBRARY_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
ENRICHR_LIBRARIES = {
    "MSigDB_Hallmark_2020": "enrichr_msigdb_hallmark_2020.gmt",
    "GO_Biological_Process_2025": "enrichr_go_biological_process_2025.gmt",
}
SPECIES = 9606
TOP_N = 100
PANEL_A_N = 24
STRING_SCORE = 700
PERTURBATION_THRESHOLD = 0.15
IMMUNE_NODE_COLOR = "#7eb6ca"
STRING_EDGE_COLOR = "#8f98a3"

GROUPS = {
    "antineoplastic": {
        "label": "Antineoplastic",
        "drug_class": "Antineoplastic",
        "color": "#e76f61",
        "light": "#f8c7bd",
    },
    "non_antineoplastic": {
        "label": "Non-antineoplastic",
        "drug_class": "Non-Antineoplastic",
        "color": "#4c9cab",
        "light": "#b9d8df",
    },
}

DB_PREFIXES = ("KEGG_", "BIOCARTA_", "REACTOME_")
TREATMENT_CATEGORY_DEFINITIONS = {
    "antineoplastic": [
        {
            "label": "Cell cycle / mitosis",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": (
                "CELL_CYCLE",
                "CELL CYCLE",
                "MITOT",
                "G2M",
                "G2_M",
                "E2F",
                "SPINDLE",
                "CHROMOSOME_SEGREGATION",
                "CHROMOSOME SEGREGATION",
                "DNA_REPLICATION",
                "DNA REPLICATION",
            ),
        },
        {
            "label": "DNA repair / p53",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": (
                "DNA_REPAIR",
                "DNA REPAIR",
                "DNA_DAMAGE",
                "DNA DAMAGE",
                "P53",
                "TP53",
                "CHECKPOINT",
                "GENOTOXIC",
            ),
        },
        {
            "label": "Apoptosis",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": ("APOPTOSIS", "PROGRAMMED_CELL_DEATH", "PROGRAMMED CELL DEATH", "CASPASE"),
        },
        {
            "label": "Oncogenic signaling",
            "collections": ("MSigDB C6 v6.2", "Enrichr MSigDB Hallmark 2020"),
            "keywords": (
                "KRAS",
                "RAS",
                "MYC",
                "E2F",
                "MTOR",
                "PI3K",
                "AKT",
                "RAF",
                "MEK",
                "EGFR",
                "ERBB",
                "BRAF",
                "WNT",
                "NOTCH",
                "P53",
                "TP53",
            ),
        },
        {
            "label": "Kinase / growth factor",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr GO BP 2025"),
            "keywords": (
                "KINASE",
                "MAPK",
                "ERK",
                "PI3K",
                "AKT",
                "MTOR",
                "EGFR",
                "ERBB",
                "PDGF",
                "FGF",
                "VEGF",
                "GROWTH_FACTOR",
                "GROWTH FACTOR",
            ),
        },
        {
            "label": "Screened cancer targets",
            "target_source": "screened_antineoplastic_targets",
        },
    ],
    "non_antineoplastic": [
        {
            "label": "Immune / cytokine",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": (
                "IMMUNE",
                "IMMUN",
                "CYTOKINE",
                "INTERLEUKIN",
                "INTERFERON",
                "INFLAMM",
                "CHEMOKINE",
                "TNF",
                "TNFA",
                "NF_KB",
                "NF-KB",
                "NFKB",
                "COMPLEMENT",
            ),
        },
        {
            "label": "ECM / adhesion",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": (
                "EXTRACELLULAR_MATRIX",
                "EXTRACELLULAR MATRIX",
                "ECM",
                "COLLAGEN",
                "ADHESION",
                "INTEGRIN",
                "MIGRATION",
                "REMODELING",
                "REMODELLING",
            ),
        },
        {
            "label": "Wound / coagulation",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": ("WOUND", "HEALING", "PLATELET", "COAGULATION", "HEMOSTASIS", "HAEMOSTASIS", "CLOT"),
        },
        {
            "label": "Lipid / cholesterol",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": ("LIPID", "CHOLESTEROL", "STEROL", "LIPOPROTEIN", "FATTY_ACID", "FATTY ACID"),
        },
        {
            "label": "Stress / xenobiotic",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr MSigDB Hallmark 2020", "Enrichr GO BP 2025"),
            "keywords": (
                "XENOBIOTIC",
                "OXIDATIVE",
                "REACTIVE_OXYGEN",
                "REACTIVE OXYGEN",
                "ROS",
                "UNFOLDED_PROTEIN",
                "UNFOLDED PROTEIN",
                "PROTEOTOXIC",
                "HEAT_SHOCK",
                "HEAT SHOCK",
                "STRESS",
                "GLUTATHIONE",
            ),
        },
        {
            "label": "Surface receptor signaling",
            "collections": ("MSigDB C2 CP v6.2", "Enrichr GO BP 2025"),
            "keywords": (
                "CELL_SURFACE",
                "CELL SURFACE",
                "RECEPTOR_SIGNALING",
                "RECEPTOR SIGNALING",
                "RECEPTOR ACTIVITY",
                "TRANSMEMBRANE_SIGNALING",
                "TRANSMEMBRANE SIGNALING",
                "SIGNALING_RECEPTOR",
                "SIGNALING RECEPTOR",
            ),
        },
    ],
}
IMMUNE_KEYWORDS = (
    "IMMUNE",
    "IMMUN",
    "CYTOKINE",
    "INTERLEUKIN",
    "INTERFERON",
    "ANTIGEN",
    "MHC",
    "T_CELL",
    "B_CELL",
    "TOLL",
    "NFKB",
    "INFLAMM",
    "CHEMOKINE",
)


def norm_text(value: object) -> str:
    text = str(value).strip().replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def norm_key(value: object) -> str:
    text = norm_text(value).lower()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    return text


def norm_sample_id(value: object) -> str:
    return re.sub(r"\s*_\s*", "_", norm_text(value))


def safe_log10_p(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        value = 1e-300
    return -math.log10(max(value, 1e-300))


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)
    out = np.empty_like(ranked)
    out[order] = ranked
    return out


def read_tsv_text(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text or text.startswith("Error:"):
        raise RuntimeError(f"STRING API returned an invalid response: {text[:250]}")
    return pd.read_csv(io.StringIO(text), sep="\t")


def load_string_api():
    spec = spec_from_file_location("string_api", STRING_API_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load STRING helper from {STRING_API_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gmt(path: Path) -> dict[str, set[str]]:
    pathways: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = norm_text(parts[0])
            if not name.startswith(DB_PREFIXES):
                continue
            genes = {norm_text(g).upper() for g in parts[2:] if norm_text(g)}
            if genes:
                pathways[name] = genes
    return pathways


def file_source_metadata(path: Path, label: str, source_type: str, url: str | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    meta: dict[str, object] = {
        "label": label,
        "source_type": source_type,
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    if url:
        meta["url"] = url
    return meta


def parse_gene_set_file(path: Path, collection: str, prefixes: tuple[str, ...] | None = None) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = norm_text(parts[0])
            if prefixes and not name.startswith(prefixes):
                continue
            genes = {norm_text(g).upper() for g in parts[2:] if norm_text(g)}
            if genes:
                gene_sets[f"{collection}::{name}"] = genes
    return gene_sets


def download_enrichr_library(library_name: str, cache_name: str) -> dict[str, object]:
    cache_path = EXTERNAL_DIR / cache_name
    if not cache_path.exists():
        response = requests.get(
            ENRICHR_LIBRARY_URL,
            params={"mode": "text", "libraryName": library_name},
            timeout=120,
        )
        response.raise_for_status()
        text = response.text
        if not text.strip() or text.startswith("Error"):
            raise RuntimeError(f"Enrichr returned an invalid response for {library_name}: {text[:250]}")
        cache_path.write_text(text, encoding="utf-8")
    return file_source_metadata(
        cache_path,
        label=f"Enrichr {library_name}",
        source_type="Enrichr geneSetLibrary text export",
        url=f"{ENRICHR_LIBRARY_URL}?mode=text&libraryName={library_name}",
    )


def split_target_symbols(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = norm_text(value).upper()
    if not text or text in {"NAN", "UNKNOWN", "NOT FOUND", "NONE"}:
        return set()
    text = re.sub(r"\([^)]*\)", "", text)
    tokens = re.split(r"[;,/|]+", text)
    genes: set[str] = set()
    for token in tokens:
        token = norm_text(token).upper()
        token = re.sub(r"[^A-Z0-9_.-]", "", token)
        if token and token not in {"NA", "NAN", "UNKNOWN", "NONE"}:
            genes.add(token)
    return genes


def screened_antineoplastic_target_set(decision: pd.DataFrame) -> set[str]:
    anti = decision.loc[decision["Antineoplastic_Including_Marginal"].eq("Yes")].copy()
    genes: set[str] = set()
    for col in ["Primary Target", "Secondary Target"]:
        if col in anti.columns:
            for value in anti[col].dropna():
                genes.update(split_target_symbols(value))
    return genes


def load_treatment_relevance_resources(
    decision: pd.DataFrame,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]], list[dict[str, object]]]:
    sources = [
        file_source_metadata(GMT_PATH, "MSigDB C2 CP v6.2", "local GMT"),
        file_source_metadata(C6_GMT_PATH, "MSigDB C6 v6.2", "local GMT"),
        file_source_metadata(DECISION_PATH, "Screened compound target annotations", "local CSV"),
    ]
    for library_name, cache_name in ENRICHR_LIBRARIES.items():
        sources.append(download_enrichr_library(library_name, cache_name))

    collections = {
        "MSigDB C2 CP v6.2": parse_gene_set_file(GMT_PATH, "MSigDB C2 CP v6.2", prefixes=DB_PREFIXES),
        "MSigDB C6 v6.2": parse_gene_set_file(C6_GMT_PATH, "MSigDB C6 v6.2"),
        "Enrichr MSigDB Hallmark 2020": parse_gene_set_file(
            EXTERNAL_DIR / ENRICHR_LIBRARIES["MSigDB_Hallmark_2020"],
            "Enrichr MSigDB Hallmark 2020",
        ),
        "Enrichr GO BP 2025": parse_gene_set_file(
            EXTERNAL_DIR / ENRICHR_LIBRARIES["GO_Biological_Process_2025"],
            "Enrichr GO BP 2025",
        ),
    }
    special_sets = {
        "screened_antineoplastic_targets": screened_antineoplastic_target_set(decision),
    }
    return collections, special_sets, sources


def treatment_relevance_gene_sets(
    group_key: str,
    collections: dict[str, dict[str, set[str]]],
    special_sets: dict[str, set[str]],
    universe: set[str],
) -> tuple[dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    gene_sets: dict[str, set[str]] = {}
    summary_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []

    for definition in TREATMENT_CATEGORY_DEFINITIONS[group_key]:
        label = definition["label"]
        genes: set[str] = set()
        contributing_terms: list[tuple[str, str, set[str]]] = []

        if "target_source" in definition:
            source_key = str(definition["target_source"])
            target_genes = special_sets.get(source_key, set())
            genes.update(target_genes)
            contributing_terms.append(("local target annotations", source_key, target_genes))
        else:
            keywords = tuple(str(k).upper() for k in definition.get("keywords", ()))
            exclude = tuple(str(k).upper() for k in definition.get("exclude_keywords", ()))
            for collection in definition.get("collections", ()):
                for term, members in collections[str(collection)].items():
                    term_upper = term.upper()
                    if keywords and not any(keyword in term_upper for keyword in keywords):
                        continue
                    if exclude and any(keyword in term_upper for keyword in exclude):
                        continue
                    genes.update(members)
                    contributing_terms.append((str(collection), term.split("::", 1)[-1], members))

        genes_in_universe = genes & universe
        gene_sets[label] = genes_in_universe
        summary_rows.append(
            {
                "group": GROUPS[group_key]["label"],
                "gene_set_label": label,
                "n_contributing_terms": len(contributing_terms),
                "n_genes_total": len(genes),
                "n_genes_in_universe": len(genes_in_universe),
                "genes_in_universe": ";".join(sorted(genes_in_universe)),
            }
        )
        for collection, term, members in contributing_terms:
            term_rows.append(
                {
                    "group": GROUPS[group_key]["label"],
                    "gene_set_label": label,
                    "collection": collection,
                    "source_term": term,
                    "term_n_genes": len(set(members)),
                    "term_n_genes_in_universe": len(set(members) & universe),
                }
            )

    summary = pd.DataFrame(summary_rows)
    terms = pd.DataFrame(term_rows)
    return gene_sets, summary, terms


def pathway_label(pathway: str) -> str:
    label = pathway
    for prefix in DB_PREFIXES:
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    return label.replace("_", " ").title()


def wrap_label(text: object, width: int = 28) -> str:
    words = norm_text(text).split(" ")
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word]).strip()
        if current and len(trial) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def ensure_output_dirs() -> None:
    for path in [EXTERNAL_DIR, TABLE_DIR, FIG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(CLASS_PATH)
    decision = pd.read_csv(DECISION_PATH)
    atc = pd.read_excel(ATC_PATH)
    for df in [meta, decision, atc]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda x: norm_text(x) if pd.notna(x) else x)
    if "Sample_ID" in meta.columns:
        meta["Sample_ID"] = meta["Sample_ID"].map(norm_sample_id)
    if "Drug_name_Target" in atc.columns:
        atc["Drug_name_Target"] = atc["Drug_name_Target"].map(norm_sample_id)
    return meta, decision, atc


def build_counts_and_atc_tables(meta: pd.DataFrame, decision: pd.DataFrame, atc: pd.DataFrame) -> None:
    counts = (
        meta["Drug_Class"]
        .fillna("<NA>")
        .value_counts(dropna=False)
        .rename_axis("Drug_Class")
        .reset_index(name="n_compounds")
    )
    counts.to_csv(TABLE_DIR / "group_counts_from_drug_class.csv", index=False)

    crosstab = pd.crosstab(
        meta["Drug_Class"].fillna("<NA>"),
        meta["Updated_Antineoplastic_at_ScreeningConc"].fillna("<NA>"),
        dropna=False,
    )
    crosstab.to_csv(TABLE_DIR / "drug_class_by_screening_call.csv")

    atc_join = atc.copy()
    atc_join["sample_key"] = atc_join["Drug_name_Target"].map(norm_key)
    atc_join["drug_key"] = atc_join["Drug_name"].map(norm_key)
    atc_sample = atc_join.drop_duplicates("sample_key")
    atc_drug = atc_join.drop_duplicates("drug_key")

    annotated = meta.copy()
    annotated["sample_key"] = annotated["Sample_ID"].map(norm_key)
    annotated["drug_key"] = annotated["Drug_Name"].map(norm_key)
    annotated = annotated.merge(
        atc_sample.add_prefix("atc_"),
        left_on="sample_key",
        right_on="atc_sample_key",
        how="left",
    )
    missing = annotated["atc_Drug_name"].isna()
    fallback = annotated.loc[missing, ["drug_key"]].merge(
        atc_drug.add_prefix("fallback_"),
        left_on="drug_key",
        right_on="fallback_drug_key",
        how="left",
    )
    for col in atc.columns:
        atc_col = f"atc_{col}"
        fb_col = f"fallback_{col}"
        if fb_col in fallback.columns:
            annotated.loc[missing, atc_col] = fallback[fb_col].to_numpy()

    keep_cols = [
        "Drug_Name",
        "Sample_ID",
        "Drug_Class",
        "HCT116_Classification",
        "Updated_Classification",
        "Updated_Antineoplastic_at_ScreeningConc",
        "Antineoplastic_Including_Marginal",
        "atc_match_status",
        "atc_name",
        "atc_drugbank_id",
        "atc_Target",
        "atc_Therapeutic_effect",
        "atc_atc_code",
        "atc_atc_level_1_name",
        "atc_atc_level_2_name",
        "atc_atc_level_3_name",
        "atc_atc_level_4_name",
    ]
    annotated[keep_cols].to_csv(TABLE_DIR / "all_compounds_drug_class_with_atc.csv", index=False)

    anti = annotated.loc[annotated["Drug_Class"].eq("Antineoplastic"), keep_cols].copy()
    anti.to_csv(TABLE_DIR / "antineoplastic_atc_classification.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for level_col, label in [
        ("atc_match_status", "match_status"),
        ("atc_atc_level_1_name", "atc_level_1"),
        ("atc_atc_level_2_name", "atc_level_2"),
        ("atc_atc_level_3_name", "atc_level_3"),
        ("atc_atc_level_4_name", "atc_level_4"),
        ("atc_Therapeutic_effect", "therapeutic_effect"),
    ]:
        vc = anti[level_col].fillna("No ATC annotation").replace("", "No ATC annotation").value_counts()
        for value, n in vc.items():
            summary_rows.append({"summary_level": label, "classification": value, "n_antineoplastic": int(n)})
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "antineoplastic_atc_classification_summary.csv", index=False)

    dose = decision.copy()
    dose["Updated_Dose_IC50_Ratio"] = pd.to_numeric(dose["Updated_Dose_IC50_Ratio"], errors="coerce")
    dose["Updated_Predicted_Inhibition_%_at_ScreeningConc"] = pd.to_numeric(
        dose["Updated_Predicted_Inhibition_%_at_ScreeningConc"], errors="coerce"
    )
    criteria_summary = {
        "requested_atc_path_exists": ATC_REQUESTED_PATH.exists(),
        "actual_atc_path": str(ATC_PATH),
        "drug_class_counts": counts.set_index("Drug_Class")["n_compounds"].to_dict(),
        "screening_call_counts": dose["Updated_Antineoplastic_at_ScreeningConc"].fillna("<NA>").value_counts().to_dict(),
        "classification_counts": dose["Updated_Classification"].fillna("<NA>").value_counts().to_dict(),
        "dose_rule": {
            "formula": "Predicted inhibition at screening concentration = 100 * dose / (dose + reference IC50)",
            "yes_threshold": ">= 50% predicted inhibition, equivalent to screening_concentration / IC50 >= 1",
            "potential_threshold": ">= 20% and < 50% predicted inhibition, equivalent to ratio >= 0.25 and < 1",
            "no_threshold": "< 20% predicted inhibition, equivalent to ratio < 0.25",
        },
        "modeling_label_rule": (
            "DeepHeal uses the curated Drug_Class label. In this file, Antineoplastic consists of "
            "all Antineoplastic_Including_Marginal = Yes compounds plus retained legacy Effective "
            "exceptions; Non-Antineoplastic consists of known inactive/non-dose-positive compounds. "
            "Unknown compounds are excluded from the binary 185/263 analysis."
        ),
        "modeling_label_note": (
            "The 185/263 split used for DeepHeal modeling is the Drug_Class column in "
            "drug_hct116_classification_including_marginal_with_cluster.csv, not ATC level alone."
        ),
        "antineoplastic_label_exceptions": meta.loc[
            meta["Drug_Class"].eq("Antineoplastic")
            & ~meta["Updated_Antineoplastic_at_ScreeningConc"].isin(["Yes", "Potential"]),
            [
                "Drug_Name",
                "Sample_ID",
                "HCT116_Classification",
                "Updated_Classification",
                "Updated_Antineoplastic_at_ScreeningConc",
                "Antineoplastic_Including_Marginal",
            ],
        ].to_dict(orient="records"),
    }
    (TABLE_DIR / "classification_criteria_and_counts.json").write_text(
        json.dumps(criteria_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_gene_matrix(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = pd.read_csv(AVG_DATA_PATH, encoding="latin-1")
    avg.columns = [norm_text(c) if idx < 2 else norm_sample_id(c) for idx, c in enumerate(avg.columns)]
    avg["Gene Name"] = avg["Gene Name"].map(lambda x: norm_text(x).upper())
    sample_ids = [c for c in avg.columns[2:] if c in set(meta["Sample_ID"])]
    gene_matrix = avg[["Gene Name", *sample_ids]].copy()
    gene_matrix = gene_matrix.groupby("Gene Name", as_index=True).mean(numeric_only=True)
    gene_matrix = gene_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    gene_matrix = gene_matrix.loc[[g for g in gene_matrix.index if g and g != "UNK"]]

    sample_meta = meta.set_index("Sample_ID").loc[sample_ids].reset_index()
    return gene_matrix, sample_meta


def load_deepheal_importance(genes: pd.Index) -> pd.DataFrame:
    weights = pd.read_csv(SUPCON_WEIGHTS_PATH, index_col=0)
    weights.columns = [norm_text(c).upper() for c in weights.columns]
    mean_abs = weights.abs().mean(axis=0)
    common = genes.intersection(mean_abs.index)
    importance = pd.DataFrame({"gene": common, "deepheal_importance_mean_abs": mean_abs.loc[common].to_numpy()})
    importance["deepheal_importance_percentile"] = importance["deepheal_importance_mean_abs"].rank(pct=True)
    return importance.set_index("gene")


def build_deepheal_group_ranking(
    group_key: str,
    gene_matrix: pd.DataFrame,
    sample_meta: pd.DataFrame,
    importance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cfg = GROUPS[group_key]
    group_ids = sample_meta.loc[sample_meta["Drug_Class"].eq(cfg["drug_class"]), "Sample_ID"].tolist()
    if not group_ids:
        raise ValueError(f"No samples found for {cfg['drug_class']}")

    common_genes = gene_matrix.index.intersection(importance.index)
    matrix = gene_matrix.loc[common_genes, group_ids].copy()
    abs_matrix = matrix.abs()
    per_compound_pct = abs_matrix.rank(axis=0, pct=True)

    ranking = importance.loc[common_genes].copy()
    ranking["group"] = cfg["label"]
    ranking["n_group_compounds"] = len(group_ids)
    ranking["response_percentile_mean"] = per_compound_pct.mean(axis=1)
    ranking["response_percentile_median"] = per_compound_pct.median(axis=1)
    ranking["response_percentile_q25"] = per_compound_pct.quantile(0.25, axis=1)
    ranking["response_percentile_q75"] = per_compound_pct.quantile(0.75, axis=1)
    ranking["response_abs_log2fc_mean"] = abs_matrix.mean(axis=1)
    ranking["response_abs_log2fc_median"] = abs_matrix.median(axis=1)
    ranking["response_frequency_abs_ge_0_15"] = abs_matrix.ge(PERTURBATION_THRESHOLD).mean(axis=1)
    ranking["response_magnitude_percentile"] = ranking["response_abs_log2fc_mean"].rank(pct=True)
    ranking["response_frequency_percentile"] = ranking["response_frequency_abs_ge_0_15"].rank(pct=True)
    ranking["deepheal_group_commonality_score"] = ranking[
        [
            "deepheal_importance_percentile",
            "response_percentile_mean",
            "response_magnitude_percentile",
            "response_frequency_percentile",
        ]
    ].mean(axis=1)
    ranking = ranking.sort_values(
        ["deepheal_group_commonality_score", "deepheal_importance_percentile", "response_percentile_mean"],
        ascending=[False, False, False],
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking.reset_index().rename(columns={"index": "gene"})

    panel_genes = ranking.head(PANEL_A_N)["gene"].tolist()
    panel_pct = (
        per_compound_pct.loc[panel_genes]
        .reset_index()
        .melt(id_vars="Gene Name" if "Gene Name" in per_compound_pct.reset_index().columns else "index")
    )
    panel_pct.columns = ["gene", "Sample_ID", "compound_response_percentile"]
    panel_pct = panel_pct.merge(sample_meta[["Sample_ID", "Drug_Name", "Drug_Class"]], on="Sample_ID", how="left")

    return ranking, panel_pct, group_ids


def compute_pca_ranking(group_key: str, gene_matrix: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    cfg = GROUPS[group_key]
    group_ids = sample_meta.loc[sample_meta["Drug_Class"].eq(cfg["drug_class"]), "Sample_ID"].tolist()
    genes = gene_matrix.index.to_numpy()
    x = gene_matrix.loc[:, group_ids].T.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    std[std < 1e-8] = 1.0
    x = x / std
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    n_components = min(100, vt.shape[0])
    if len(s) > 1:
        ev = (s**2) / max(1, (x.shape[0] - 1))
        evr = ev / ev.sum()
    else:
        evr = np.ones_like(s)
    weights = evr[:n_components]
    if weights.sum() <= 0:
        weights = np.ones(n_components) / n_components
    else:
        weights = weights / weights.sum()
    score = np.abs(vt[:n_components, :]).T @ weights
    ranking = pd.DataFrame(
        {
            "gene": genes,
            "group": cfg["label"],
            "pca_importance_weighted_abs_loading": score,
            "pca_n_components": n_components,
            "pca_explained_variance_ratio_used": float(evr[:n_components].sum()),
        }
    )
    ranking["pca_importance_percentile"] = ranking["pca_importance_weighted_abs_loading"].rank(pct=True)
    ranking = ranking.sort_values("pca_importance_weighted_abs_loading", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def fisher_enrichment(query_genes: set[str], gene_sets: dict[str, set[str]], universe: set[str]) -> pd.DataFrame:
    query = set(query_genes) & universe
    rows: list[dict[str, object]] = []
    for name, genes in gene_sets.items():
        gs = set(genes) & universe
        if len(gs) < 5:
            continue
        overlap = query & gs
        a = len(overlap)
        b = len(query) - a
        c = len(gs) - a
        d = len(universe) - a - b - c
        _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "pathway": name,
                "pathway_label": pathway_label(name),
                "database": name.split("_", 1)[0].title(),
                "query_n": len(query),
                "universe_n": len(universe),
                "pathway_n_in_universe": len(gs),
                "overlap_n": a,
                "overlap_genes": ";".join(sorted(overlap)),
                "p_value": float(pvalue),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr_bh"] = bh_fdr(df["p_value"])
    df["neglog10_fdr"] = df["fdr_bh"].map(safe_log10_p)
    df["neglog10_p"] = df["p_value"].map(safe_log10_p)
    return df.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False]).reset_index(drop=True)


def pathway_method_comparison(
    group_key: str,
    deepheal_top100: set[str],
    pca_top100: set[str],
    pathways: dict[str, set[str]],
    universe: set[str],
    panel_n: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = {
        "DeepHeal": deepheal_top100,
        "PCA": pca_top100,
    }
    all_results: list[pd.DataFrame] = []
    for method, genes in methods.items():
        enriched = fisher_enrichment(set(genes), pathways, universe)
        enriched.insert(0, "method", method)
        enriched.insert(0, "group", GROUPS[group_key]["label"])
        all_results.append(enriched)

    comparison = pd.concat(all_results, ignore_index=True)
    selection = (
        comparison.groupby(["pathway", "pathway_label", "database"], as_index=False)
        .agg(
            min_fdr=("fdr_bh", "min"),
            max_neglog10_fdr=("neglog10_fdr", "max"),
            max_overlap=("overlap_n", "max"),
        )
        .sort_values(["min_fdr", "max_neglog10_fdr", "max_overlap"], ascending=[True, False, False])
        .head(panel_n)
    )
    selected_pathways = selection["pathway"].tolist()
    panel = comparison.loc[comparison["pathway"].isin(selected_pathways)].copy()
    panel["pathway_order"] = pd.Categorical(panel["pathway"], categories=selected_pathways, ordered=True)
    panel["method_order"] = pd.Categorical(panel["method"], categories=["DeepHeal", "PCA"], ordered=True)
    panel = panel.sort_values(["pathway_order", "method_order"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for method, sub in comparison.groupby("method", sort=False):
        sig = sub.loc[sub["fdr_bh"].lt(0.05)].copy()
        best = sub.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False]).iloc[0]
        summary_rows.append(
            {
                "group": GROUPS[group_key]["label"],
                "method": method,
                "query_n": int(best["query_n"]),
                "universe_n": int(best["universe_n"]),
                "n_pathways_tested": int(len(sub)),
                "n_pathways_fdr_lt_0_05": int(len(sig)),
                "pathway_coherence_score_sum_neglog10_fdr_fdr_lt_0_05": float(sig["neglog10_fdr"].sum()),
                "best_pathway": best["pathway"],
                "best_pathway_label": best["pathway_label"],
                "best_pathway_overlap_n": int(best["overlap_n"]),
                "best_pathway_fdr_bh": float(best["fdr_bh"]),
                "best_pathway_neglog10_fdr": float(best["neglog10_fdr"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return comparison, panel, summary


def treatment_relevance_enrichment(
    group_key: str,
    deepheal_top100: set[str],
    pca_top100: set[str],
    treatment_sets: dict[str, set[str]],
    universe: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = {
        "DeepHeal": deepheal_top100,
        "PCA": pca_top100,
    }
    for method, query_genes in methods.items():
        query = set(query_genes) & universe
        for label, genes in treatment_sets.items():
            gene_set = set(genes) & universe
            overlap = query & gene_set
            a = len(overlap)
            b = len(query) - a
            c = len(gene_set) - a
            d = len(universe) - a - b - c
            odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "group": GROUPS[group_key]["label"],
                    "method": method,
                    "gene_set_label": label,
                    "query_n": len(query),
                    "universe_n": len(universe),
                    "gene_set_n_in_universe": len(gene_set),
                    "overlap_n": a,
                    "overlap_genes": ";".join(sorted(overlap)),
                    "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else "inf",
                    "p_value": float(pvalue),
                    "neglog10_p": safe_log10_p(float(pvalue)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr_bh_panel_d_tests"] = bh_fdr(df["p_value"])
    df["neglog10_fdr"] = df["fdr_bh_panel_d_tests"].map(safe_log10_p)
    df["gene_set_order"] = pd.Categorical(
        df["gene_set_label"],
        categories=[definition["label"] for definition in TREATMENT_CATEGORY_DEFINITIONS[group_key]],
        ordered=True,
    )
    df["method_order"] = pd.Categorical(df["method"], categories=["DeepHeal", "PCA"], ordered=True)
    return df.sort_values(["gene_set_order", "method_order"]).reset_index(drop=True)


def immune_pathway_gene_set(pathways: dict[str, set[str]], universe: set[str]) -> set[str]:
    genes: set[str] = set()
    for name, members in pathways.items():
        upper = name.upper()
        if any(keyword in upper for keyword in IMMUNE_KEYWORDS):
            genes.update(set(members) & universe)
    return genes


def download_file(url: str, dest: Path) -> dict[str, object]:
    if not dest.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        dest.write_bytes(response.content)
    payload = dest.read_bytes()
    return {
        "url": url,
        "path": str(dest),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def parse_cspa_genes(path: Path) -> set[str]:
    table_a = pd.read_excel(path, sheet_name="Table_A")
    table_a.columns = [norm_text(c) for c in table_a.columns]
    human = table_a.loc[table_a["organism"].astype(str).str.lower().eq("human")].copy()
    genes = {norm_text(g).upper() for g in human["ENTREZ gene symbol"].dropna() if norm_text(g)}
    return genes


def fetch_uniprot_tsv(query: str, cache_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_path = EXTERNAL_DIR / cache_name
    source = {
        "url": UNIPROT_URL,
        "query": query,
        "path": str(cache_path),
    }
    if not cache_path.exists():
        chunks: list[str] = []
        next_url: str | None = UNIPROT_URL
        params = {
            "query": query,
            "fields": "accession,gene_primary,keyword,go_p,go_f",
            "format": "tsv",
            "size": "500",
        }
        while next_url:
            if next_url == UNIPROT_URL:
                response = requests.get(next_url, params=params, timeout=120)
            else:
                response = requests.get(next_url, timeout=120)
            response.raise_for_status()
            text = response.text
            if chunks:
                text = "\n".join(text.splitlines()[1:])
            chunks.append(text)
            next_url = None
            link = response.headers.get("Link", "")
            next_match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
            if next_match:
                next_url = next_match.group(1)
        cache_path.write_text("\n".join(chunks), encoding="utf-8")
    text = cache_path.read_text(encoding="utf-8")
    source["bytes"] = len(text.encode("utf-8"))
    source["sha256"] = sha256(text.encode("utf-8")).hexdigest()
    df = pd.read_csv(io.StringIO(text), sep="\t")
    return df, source


def load_external_gene_sets() -> tuple[dict[str, set[str]], pd.DataFrame, list[dict[str, object]]]:
    cspa_meta = download_file(CSPA_URL, EXTERNAL_DIR / "pone.0121314.s002.xlsx")
    cspa_genes = parse_cspa_genes(EXTERNAL_DIR / "pone.0121314.s002.xlsx")

    uniprot_queries = {
        "UniProt cell surface receptors": "(organism_id:9606) AND (reviewed:true) AND (keyword:KW-1003) AND (keyword:KW-0675)",
        "UniProt cytokine receptors": "(organism_id:9606) AND (reviewed:true) AND (go:0004896)",
        "GO immune response-regulating cell surface receptor signaling pathway": "(organism_id:9606) AND (reviewed:true) AND (go:0002768)",
    }
    sources = [cspa_meta]
    gene_sets = {"Human Surface Proteins Atlas": cspa_genes}
    summary_rows = [
        {
            "gene_set": "Human Surface Proteins Atlas",
            "source": "CSPA supplementary Table_A human gene symbols",
            "n_genes": len(cspa_genes),
        }
    ]

    for label, query in uniprot_queries.items():
        cache_name = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() + ".tsv"
        df, source = fetch_uniprot_tsv(query, cache_name)
        sources.append(source)
        gene_col = "Gene Names (primary)"
        genes = {norm_text(g).upper() for g in df[gene_col].dropna() if norm_text(g)}
        gene_sets[label] = genes
        summary_rows.append({"gene_set": label, "source": query, "n_genes": len(genes)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "external_gene_set_summary.csv", index=False)
    return gene_sets, summary, sources


def receptor_enrichment(
    group_key: str,
    deepheal_top100: set[str],
    pca_top100: set[str],
    receptor_sets: dict[str, set[str]],
    universe: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = {
        "DeepHeal": deepheal_top100,
        "PCA": pca_top100,
    }
    for method, query in methods.items():
        query = set(query) & universe
        for label, genes in receptor_sets.items():
            gs = set(genes) & universe
            overlap = query & gs
            a = len(overlap)
            b = len(query) - a
            c = len(gs) - a
            d = len(universe) - a - b - c
            _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "group": GROUPS[group_key]["label"],
                    "method": method,
                    "gene_set": label,
                    "query_n": len(query),
                    "universe_n": len(universe),
                    "gene_set_n_in_universe": len(gs),
                    "overlap_n": a,
                    "overlap_genes": ";".join(sorted(overlap)),
                    "p_value": float(pvalue),
                    "neglog10_p": safe_log10_p(float(pvalue)),
                }
            )
    df = pd.DataFrame(rows)
    df["fdr_bh_all_panel_d_tests"] = bh_fdr(df["p_value"])
    df["neglog10_fdr"] = df["fdr_bh_all_panel_d_tests"].map(safe_log10_p)
    return df


def build_string_network(
    group_key: str,
    top100: pd.DataFrame,
    immune_genes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    prefix = group_key
    string_api = load_string_api()
    genes = top100["gene"].tolist()
    mapped = read_tsv_text(string_api.string_map_ids(genes, species=SPECIES, limit=1))
    mapped = mapped.drop_duplicates(subset=["queryItem"]).copy()
    mapped["queryItem"] = mapped["queryItem"].map(lambda x: norm_text(x).upper())
    mapped["preferredName"] = mapped["preferredName"].map(lambda x: norm_text(x).upper())
    mapped.to_csv(TABLE_DIR / f"{prefix}_string_mapping.tsv", sep="\t", index=False)

    mapped_ids = mapped["stringId"].dropna().astype(str).tolist()
    network = read_tsv_text(string_api.string_network(mapped_ids, species=SPECIES, required_score=STRING_SCORE))
    network["edge_score"] = network["score"]
    network["preferredName_A"] = network["preferredName_A"].map(lambda x: norm_text(x).upper())
    network["preferredName_B"] = network["preferredName_B"].map(lambda x: norm_text(x).upper())

    graph = nx.Graph()
    for row in network.itertuples(index=False):
        graph.add_edge(row.preferredName_A, row.preferredName_B, score=float(row.score))

    degree = dict(graph.degree())
    node_df = mapped.merge(top100, left_on="queryItem", right_on="gene", how="left")
    node_df["hubness_degree"] = node_df["preferredName"].map(degree).fillna(0).astype(int)
    node_df["connected"] = node_df["hubness_degree"].gt(0)
    node_df["immune_response_related"] = node_df["preferredName"].isin(immune_genes) | node_df["gene"].isin(immune_genes)
    node_df = node_df.loc[node_df["connected"]].copy()
    connected = set(node_df["preferredName"])
    edge_df = network.loc[
        network["preferredName_A"].isin(connected) & network["preferredName_B"].isin(connected)
    ].copy()

    node_df = node_df.sort_values(["hubness_degree", "rank"], ascending=[False, True]).reset_index(drop=True)
    edge_df.to_csv(TABLE_DIR / f"{prefix}_string_edges_score{STRING_SCORE}.tsv", sep="\t", index=False)
    node_df.to_csv(TABLE_DIR / f"{prefix}_string_nodes_score{STRING_SCORE}.csv", index=False)
    unmapped = top100.loc[~top100["gene"].isin(set(mapped["queryItem"]))].copy()
    unmapped.to_csv(TABLE_DIR / f"{prefix}_string_unmapped_top100.csv", index=False)

    summary = {
        "group": GROUPS[group_key]["label"],
        "string_score_threshold": STRING_SCORE,
        "top100_n": int(len(top100)),
        "mapped_n": int(len(mapped)),
        "unmapped_n": int(len(unmapped)),
        "connected_node_n": int(len(node_df)),
        "edge_n": int(len(edge_df)),
        "immune_connected_node_n": int(node_df["immune_response_related"].sum()) if len(node_df) else 0,
    }
    return node_df, edge_df, summary


def draw_network(ax: plt.Axes, nodes: pd.DataFrame, edges: pd.DataFrame, group_color: str) -> None:
    ax.set_axis_off()
    if nodes.empty or edges.empty:
        ax.text(0.5, 0.5, f"No connected STRING edges\n(score >= {STRING_SCORE})", ha="center", va="center", fontsize=8)
        return
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.preferredName,
            degree=int(row.hubness_degree),
            rank=int(row.rank),
            immune=bool(row.immune_response_related),
        )
    for row in edges.itertuples(index=False):
        if row.preferredName_A in graph and row.preferredName_B in graph:
            graph.add_edge(row.preferredName_A, row.preferredName_B, weight=float(row.edge_score))

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    anchors = [(0.0, 0.0), (1.45, 0.86), (1.45, -0.86), (-1.45, 0.86), (-1.45, -0.86)]
    pos: dict[str, np.ndarray] = {}
    for i, component in enumerate(components):
        subgraph = graph.subgraph(component).copy()
        scale = 1.0 if i == 0 else 0.16
        if len(subgraph) == 1:
            sub_pos = {next(iter(subgraph.nodes())): np.asarray([0.0, 0.0])}
        else:
            sub_pos = nx.spring_layout(
                subgraph,
                seed=31 + i,
                k=0.78 if i == 0 else 0.35,
                iterations=1300 if i == 0 else 500,
                weight=None,
                scale=scale,
            )
        sub_coords = np.asarray([sub_pos[n] for n in subgraph.nodes()], dtype=float)
        sub_center = sub_coords.mean(axis=0)
        anchor = np.asarray(anchors[i] if i < len(anchors) else (1.45, 0.0), dtype=float)
        for node, xy in sub_pos.items():
            pos[node] = np.asarray(xy, dtype=float) - sub_center + anchor

    degrees = np.asarray([graph.nodes[n]["degree"] for n in graph.nodes()], dtype=float)
    if len(degrees) and degrees.max() > degrees.min():
        sizes = 45 + (degrees - degrees.min()) / (degrees.max() - degrees.min()) * 420
    else:
        sizes = np.full(len(graph.nodes()), 120.0)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=STRING_EDGE_COLOR, width=0.8, alpha=0.62)
    colors = [IMMUNE_NODE_COLOR if graph.nodes[n]["immune"] else group_color for n in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
    )
    coords = np.asarray([pos[n] for n in graph.nodes()], dtype=float)
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    label_nodes = list(graph.nodes())
    node_norm = np.asarray([[(pos[n][0] - x_min) / x_span, (pos[n][1] - y_min) / y_span] for n in label_nodes], dtype=float)
    center_norm = node_norm.mean(axis=0)
    label_norm = np.zeros_like(node_norm)
    for i, point in enumerate(node_norm):
        direction = point - center_norm
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            angle = 2 * math.pi * i / max(len(label_nodes), 1)
            direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        else:
            direction = direction / distance
        label_norm[i] = point + direction * 0.035

    target_norm = label_norm.copy()
    widths = np.asarray([0.014 + 0.0054 * len(node) for node in label_nodes], dtype=float)
    heights = np.full(len(label_nodes), 0.026, dtype=float)
    max_offset = 0.125
    for _ in range(180):
        label_norm += 0.055 * (target_norm - label_norm)
        for i in range(len(label_nodes)):
            for j in range(i + 1, len(label_nodes)):
                dx = label_norm[j, 0] - label_norm[i, 0]
                dy = label_norm[j, 1] - label_norm[i, 1]
                overlap_x = (widths[i] + widths[j]) / 2 - abs(dx)
                overlap_y = (heights[i] + heights[j]) / 2 - abs(dy)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if overlap_x < overlap_y:
                    push = np.asarray([math.copysign(overlap_x / 2 + 0.001, dx if dx else 1.0), 0.0])
                else:
                    push = np.asarray([0.0, math.copysign(overlap_y / 2 + 0.001, dy if dy else 1.0)])
                label_norm[j] += push
                label_norm[i] -= push
        deltas = label_norm - node_norm
        distances = np.linalg.norm(deltas, axis=1)
        far = distances > max_offset
        label_norm[far] = node_norm[far] + deltas[far] / distances[far, None] * max_offset

    label_coords = np.column_stack([x_min + label_norm[:, 0] * x_span, y_min + label_norm[:, 1] * y_span])
    for node, label_xy, node_xy in zip(label_nodes, label_coords, coords):
        direction = label_xy - node_xy
        ax.text(
            label_xy[0],
            label_xy[1],
            node,
            fontsize=4.2,
            color="#111827",
            ha="left" if direction[0] >= 0 else "right",
            va="bottom" if direction[1] >= 0 else "top",
            fontstyle="italic",
            bbox={"boxstyle": "round,pad=0.04", "facecolor": "white", "edgecolor": "none", "alpha": 0.56},
            clip_on=False,
            zorder=5,
        )

    all_x = np.concatenate([coords[:, 0], label_coords[:, 0]])
    all_y = np.concatenate([coords[:, 1], label_coords[:, 1]])
    ax.set_xlim(float(all_x.min()) - 0.08 * x_span, float(all_x.max()) + 0.08 * x_span)
    ax.set_ylim(float(all_y.min()) - 0.08 * y_span, float(all_y.max()) + 0.08 * y_span)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=group_color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color="none",
            markersize=5.5,
            label="STRING node",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=IMMUNE_NODE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color="none",
            markersize=5.5,
            label="Immune-related",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#d1d5db",
            markeredgecolor="white",
            markeredgewidth=0.6,
            color="none",
            markersize=8,
            label="Larger = higher degree",
        ),
        Line2D([0], [0], color=STRING_EDGE_COLOR, linewidth=0.8, alpha=0.62, label=f"STRING score >= {STRING_SCORE}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def plot_group_figure(
    group_key: str,
    ranking: pd.DataFrame,
    panel_pct: pd.DataFrame,
    enrichment_top: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> None:
    cfg = GROUPS[group_key]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    fig = plt.figure(figsize=(15.4, 7.2))
    gs = fig.add_gridspec(1, 7, width_ratios=[1.0, 0.55, 1.0, 0.35, 3.15, 0.55, 1.18], wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 4])
    ax_d = fig.add_subplot(gs[0, 6])

    top_panel = ranking.head(PANEL_A_N).copy()
    top_panel["y"] = np.arange(len(top_panel))
    y_map = dict(zip(top_panel["gene"], top_panel["y"]))
    plot_pct = panel_pct.loc[panel_pct["gene"].isin(y_map)].copy()
    plot_pct["y"] = plot_pct["gene"].map(y_map)
    ax_a.scatter(
        plot_pct["compound_response_percentile"] * 100,
        plot_pct["y"],
        s=8,
        color=cfg["light"],
        alpha=0.22,
        linewidths=0,
    )
    ax_a.scatter(
        top_panel["response_percentile_mean"] * 100,
        top_panel["y"],
        s=38,
        color=cfg["color"],
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax_a.set_yticks(top_panel["y"])
    ax_a.set_yticklabels(top_panel["gene"], fontstyle="italic")
    ax_a.invert_yaxis()
    xmin = max(40, math.floor(float(top_panel["response_percentile_q25"].min() * 100) / 5) * 5 - 5)
    ax_a.set_xlim(xmin, 101)
    ax_a.set_xlabel("Compound response\npercentile")
    ax_a.set_title("a  Top DeepHeal genes", loc="left", fontweight="bold")
    ax_a.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    ax_a.spines[["top", "right"]].set_visible(False)

    if enrichment_top.empty:
        ax_b.text(0.5, 0.5, "No pathway enrichment", ha="center", va="center", transform=ax_b.transAxes)
    else:
        bdf = enrichment_top.head(12).iloc[::-1].copy()
        ax_b.barh(np.arange(len(bdf)), bdf["neglog10_fdr"], color="#7aa6b6")
        ax_b.set_yticks(np.arange(len(bdf)))
        ax_b.set_yticklabels([wrap_label(x, 20) for x in bdf["pathway_label"]])
        ax_b.set_xlabel("-log10(FDR)")
        ax_b.tick_params(axis="y", pad=2)
        ax_b.spines[["top", "right"]].set_visible(False)
        ax_b.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    ax_b.set_title("b  Top pathways", loc="left", fontweight="bold")

    draw_network(ax_c, nodes, edges, cfg["color"])
    ax_c.set_title("c  STRING network of top 100 genes", loc="left", fontweight="bold")

    ddf = panel_d.copy()
    if ddf.empty:
        ax_d.text(0.5, 0.5, "No treatment-relevance comparison", ha="center", va="center", transform=ax_d.transAxes)
    else:
        d_order = [definition["label"] for definition in TREATMENT_CATEGORY_DEFINITIONS[group_key]]
        y_base = np.arange(len(d_order), dtype=float)
        offsets = {"DeepHeal": -0.17, "PCA": 0.17}
        colors = {"DeepHeal": cfg["color"], "PCA": "#e6b949"}
        for method in ["DeepHeal", "PCA"]:
            sub = ddf.loc[ddf["method"].eq(method)].set_index("gene_set_label").reindex(d_order).reset_index()
            ax_d.barh(y_base + offsets[method], sub["neglog10_fdr"], height=0.28, color=colors[method], label=method)
            for y, row in zip(y_base + offsets[method], sub.itertuples(index=False)):
                if pd.notna(row.overlap_n) and row.overlap_n > 0:
                    ax_d.text(
                        float(row.neglog10_fdr) + 0.04,
                        y,
                        str(int(row.overlap_n)),
                        va="center",
                        ha="left",
                        fontsize=6,
                        color="#374151",
                    )
        ax_d.set_yticks(y_base)
        ax_d.set_yticklabels([wrap_label(x, 22) for x in d_order])
        ax_d.tick_params(axis="y", pad=2)
        ax_d.invert_yaxis()
        ax_d.legend(frameon=False, fontsize=7, loc="lower right")
    ax_d.set_xlabel("-log10(FDR)")
    ax_d.set_title("d  Treatment relevance vs PCA", loc="left", fontweight="bold")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.grid(axis="x", color="#e5e7eb", linewidth=0.6)

    fig.savefig(FIG_DIR / f"deepheal_deepprofile_commonality_{group_key}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"deepheal_deepprofile_commonality_{group_key}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-string", action="store_true", help="Skip live STRING network retrieval.")
    args = parser.parse_args()

    ensure_output_dirs()
    meta, decision, atc = load_metadata()
    build_counts_and_atc_tables(meta, decision, atc)

    gene_matrix, sample_meta = load_gene_matrix(meta)
    importance = load_deepheal_importance(gene_matrix.index)
    pathways = parse_gmt(GMT_PATH)
    universe = set(gene_matrix.index.intersection(importance.index))
    immune_genes = immune_pathway_gene_set(pathways, universe)
    treatment_collections, treatment_special_sets, external_sources = load_treatment_relevance_resources(decision)

    summary = {
        "output_tables": str(TABLE_DIR),
        "output_figures": str(FIG_DIR),
        "universe_n_genes": len(universe),
        "top_n": TOP_N,
        "panel_a_n": PANEL_A_N,
        "perturbation_threshold_abs_log2fc": PERTURBATION_THRESHOLD,
        "deepheal_importance_source": str(SUPCON_WEIGHTS_PATH),
        "deepheal_score_definition": (
            "Mean of four percentile components: SupCon DeepHeal latent-weight importance, "
            "mean per-compound absolute-response percentile within group, group mean absolute log2FC percentile, "
            "and group frequency percentile for |log2FC| >= 0.15."
        ),
        "pca_definition": (
            "PCA comparator is fitted separately within each compound group on centered/scaled HCT116 average log2FC "
            "profiles; genes are ranked by explained-variance-weighted absolute loadings across up to 100 PCs."
        ),
        "panel_d_definition": (
            "Treatment-relevance panel d tests prespecified aggregate gene-set categories for each compound group. "
            "Categories are assembled from local MSigDB C2/C6 v6.2 GMT files, cached Enrichr Hallmark 2020 and "
            "GO Biological Process 2025 GMT exports, and local screened-antineoplastic target annotations where "
            "specified. Fisher exact tests compare DeepHeal top-100 genes with PCA top-100 genes against the "
            "same expressed/protein-response universe; Benjamini-Hochberg FDR is applied within the panel-d "
            "testing family for each group."
        ),
        "external_sources": external_sources,
        "groups": {},
    }

    for group_key in GROUPS:
        ranking, panel_pct, group_ids = build_deepheal_group_ranking(group_key, gene_matrix, sample_meta, importance)
        pca = compute_pca_ranking(group_key, gene_matrix.loc[ranking["gene"]], sample_meta)
        prefix = group_key
        ranking.to_csv(TABLE_DIR / f"{prefix}_deepheal_gene_ranking_all.csv", index=False)
        ranking.head(TOP_N).to_csv(TABLE_DIR / f"{prefix}_deepheal_top100_genes.csv", index=False)
        ranking.head(PANEL_A_N).to_csv(TABLE_DIR / f"{prefix}_panel_a_top_genes.csv", index=False)
        panel_pct.to_csv(TABLE_DIR / f"{prefix}_panel_a_gene_compound_percentiles.csv", index=False)
        pca.to_csv(TABLE_DIR / f"{prefix}_pca_gene_ranking_all.csv", index=False)
        pca.head(TOP_N).to_csv(TABLE_DIR / f"{prefix}_pca_top100_genes.csv", index=False)

        top100_genes = set(ranking.head(TOP_N)["gene"])
        pca_top100_genes = set(pca.head(TOP_N)["gene"])
        pathway_enrichment = fisher_enrichment(top100_genes, pathways, universe)
        pathway_enrichment.to_csv(TABLE_DIR / f"{prefix}_pathway_enrichment_all.csv", index=False)
        pathway_top = pathway_enrichment.head(20).copy()
        pathway_top.to_csv(TABLE_DIR / f"{prefix}_pathway_enrichment_top.csv", index=False)
        treatment_sets, treatment_set_summary, treatment_source_terms = treatment_relevance_gene_sets(
            group_key=group_key,
            collections=treatment_collections,
            special_sets=treatment_special_sets,
            universe=universe,
        )
        panel_d = treatment_relevance_enrichment(
            group_key=group_key,
            deepheal_top100=top100_genes,
            pca_top100=pca_top100_genes,
            treatment_sets=treatment_sets,
            universe=universe,
        )
        treatment_set_summary.to_csv(TABLE_DIR / f"{prefix}_panel_d_treatment_gene_set_summary.csv", index=False)
        treatment_source_terms.to_csv(TABLE_DIR / f"{prefix}_panel_d_treatment_source_terms.csv", index=False)
        panel_d.to_csv(TABLE_DIR / f"{prefix}_panel_d_treatment_relevance.csv", index=False)

        if args.skip_string:
            node_df = pd.DataFrame()
            edge_df = pd.DataFrame()
            string_summary = {"skipped": True}
        else:
            node_df, edge_df, string_summary = build_string_network(group_key, ranking.head(TOP_N).copy(), immune_genes)

        plot_group_figure(
            group_key=group_key,
            ranking=ranking,
            panel_pct=panel_pct,
            enrichment_top=pathway_top,
            nodes=node_df,
            edges=edge_df,
            panel_d=panel_d,
        )

        summary["groups"][group_key] = {
            "label": GROUPS[group_key]["label"],
            "n_group_compounds_from_metadata": int((meta["Drug_Class"] == GROUPS[group_key]["drug_class"]).sum()),
            "n_group_compounds_matched_to_average_matrix": int(len(group_ids)),
            "top_deepheal_genes": ranking.head(10)["gene"].tolist(),
            "top_pathways": pathway_top.head(5)[["pathway", "fdr_bh", "overlap_n"]].to_dict(orient="records")
            if not pathway_top.empty
            else [],
            "string_summary": string_summary,
            "panel_d_treatment_gene_sets": treatment_set_summary[
                ["gene_set_label", "n_contributing_terms", "n_genes_in_universe"]
            ].to_dict(orient="records"),
            "panel_d_treatment_relevance": panel_d[
                ["method", "gene_set_label", "overlap_n", "gene_set_n_in_universe", "fdr_bh_panel_d_tests", "neglog10_fdr"]
            ].to_dict(orient="records"),
        }

    (RESULT_DIR / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
