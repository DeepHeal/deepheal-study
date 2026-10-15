from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT

from pathlib import Path
from collections import Counter
import pickle
import re
import numpy as np
import pandas as pd

def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip()).upper()
    return re.sub(r"\s*_\s*", "_", text)

def load_annotation(proteome_dir: Path) -> pd.DataFrame:
    path = next(proteome_dir.glob("README/Gene_annotation_and_representable_isoform_mapping_table.txt"))
    annotation = pd.read_csv(path, sep="\t", dtype=str)
    annotation["gene"] = annotation["gene"].map(str)
    annotation["gene_symbol"] = annotation["gene_name_BCM_refined"].map(norm)
    annotation["swissprot"] = annotation["SwissProt"].map(norm).replace("NA", "")
    annotation["primary_order"] = annotation["Primary_select"].map(lambda x: 0 if norm(x) == "YES" else 1)
    annotation = annotation.sort_values(["gene", "primary_order"]).drop_duplicates("gene", keep="first")
    return annotation[["gene", "gene_symbol", "swissprot"]].copy()

def proteome_dir(repo_root: Path) -> Path:
    base = repo_root / "data" / "external" / "cptac_pancancer_bcm_v1" / "extracted"
    candidates = [p for p in base.glob("Proteome_BCM_GENCODE_v34_harmonized_v1") if p.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one extracted BCM proteome directory under {base}")
    return candidates[0]
