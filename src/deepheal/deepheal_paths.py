from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import os
import sys
import sys
from pathlib import Path


REPO_ROOT = ARTICLE_ROOT
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
CODE_DIR = REPO_ROOT / "src" / "deepheal"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
PATHWAY_GMT_DIR = DATA_DIR / "pathways" / "gmt_cache"

SUPCON_DIR = RESULTS_DIR / "supcon"
AVERAGE_DIR = RESULTS_DIR / "average"
SOURCE_MODEL_CKPT_DIR = RESULTS_DIR / "source_model_checkpoints"
SUPCON_SOURCE_MODEL_CKPT_DIR = SOURCE_MODEL_CKPT_DIR / "supcon"
AVERAGE_SOURCE_MODEL_CKPT_DIR = SOURCE_MODEL_CKPT_DIR / "average"
STRING_SUPCON_DIR = RESULTS_DIR / "stringdb" / "supcon"
STRING_AVERAGE_DIR = RESULTS_DIR / "stringdb" / "average"
PAN_DRUG_PATHWAY_DIR = RESULTS_DIR / "pan_drug_pathway"
MACRO_MOA_DIR = RESULTS_DIR / "macro_moa"
HEPG2_DIR = RESULTS_DIR / "hepg2_projection"
HEPG2_TABLE_DIR = HEPG2_DIR / "tables"
HEPG2_FIG_DIR = HEPG2_DIR / "figures"
HEPG2_AUTOGLUON_DIR = HEPG2_DIR / "autogluon"
HEPG2_CACHE_DIR = HEPG2_DIR / "cache"
HEPG2_MODEL_CACHE_DIR = HEPG2_CACHE_DIR / "source_model_projection"
EXTERNAL_PANEL_DIR = RESULTS_DIR / "external_panel_predictions"
DOMAIN_SHIFT_VALIDATION_DIR = RESULTS_DIR / "external_domain_shift_validation"
DOMAIN_SHIFT_TABLE_DIR = DOMAIN_SHIFT_VALIDATION_DIR / "tables"
DOMAIN_SHIFT_FIG_DIR = DOMAIN_SHIFT_VALIDATION_DIR / "figures"

HCT116_BASELINE_1628 = RAW_DATA_DIR / "1628_Experiments_Data.csv"
HCT116_REPLICATE_DATA = RAW_DATA_DIR / "1752_Experiments_Data_With_Replicate.csv"
HCT116_REPLICATE_VAE_DATA = RAW_DATA_DIR / "1752_Experiments_Data_With_Replicate_for_VAE.csv"
HCT116_AVERAGE_DATA = RAW_DATA_DIR / "875_Compounds_Average_Data.csv"
HCT116_CLASSIFICATION_CSV = RAW_DATA_DIR / "drug_hct116_classification_including_marginal_with_cluster.csv"
HCT116_CLASSIFICATION_XLSX = RAW_DATA_DIR / "drug_hct116_classification_including_marginal_with_cluster.xlsx"
HEPG2_INPUT_XLSX = RAW_DATA_DIR / "HepG2_07_logFC_ave_vs_Con.xlsx"
HEPG2_367_INPUT_XLSX = RAW_DATA_DIR / "HepG2_10_final_l2fc_367_6682.xlsx"
CANONICAL_GMT_PATH = PATHWAY_GMT_DIR / "c2.cp.v6.2.symbols.gmt"
ONCOGENIC_GMT_PATH = PATHWAY_GMT_DIR / "c6.all.v6.2.symbols.gmt"

SUPCON_EMBED_SUMMARY = SUPCON_DIR / "supcon_consensus_embedding_including_marginal.csv"
SUPCON_EMBED_FULL = SUPCON_DIR / "supcon_consensus_embedding_full_including_marginal.csv"
AVERAGE_EMBED_SUMMARY = AVERAGE_DIR / "average_consensus_embedding_including_marginal.csv"
AVERAGE_EMBED_FULL = AVERAGE_DIR / "average_consensus_embedding_full_including_marginal.csv"

PY_BASE = Path(os.environ.get("DEEPHEAL_PY_BASE", sys.executable))
PY_PROTO = Path(os.environ.get("DEEPHEAL_PY_PROTO", sys.executable))
PY_AUTO = Path(os.environ.get("DEEPHEAL_PY_AUTO", sys.executable))

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 200
DEFAULT_FINAL_VAL_FRACTION = 0.10


def ensure_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        METADATA_DIR,
        PATHWAY_GMT_DIR,
        SUPCON_DIR,
        AVERAGE_DIR,
        SOURCE_MODEL_CKPT_DIR,
        STRING_SUPCON_DIR,
        STRING_AVERAGE_DIR,
        PAN_DRUG_PATHWAY_DIR,
        MACRO_MOA_DIR,
        HEPG2_DIR,
        HEPG2_TABLE_DIR,
        HEPG2_FIG_DIR,
        HEPG2_AUTOGLUON_DIR,
        HEPG2_MODEL_CACHE_DIR,
        EXTERNAL_PANEL_DIR,
        DOMAIN_SHIFT_TABLE_DIR,
        DOMAIN_SHIFT_FIG_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        NOTEBOOKS_DIR,
        MANUSCRIPT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    if extra:
        env.update(extra)
    return env
