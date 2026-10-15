"""Download immutable author-processed inputs used by Figure 8.

This script transfers published supplementary/processed files only. It does not
download or process mass-spectrometry RAW files. Binary downloads are written
verbatim; analysis and figure generation remain in the standalone R script.
"""

from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import requests
import urllib3


PACKAGE_ROOT = ARTICLE_ROOT / "data/clinical"
RAW_DIR = PACKAGE_ROOT
ACCESS_DATE = __import__("datetime").date.today().isoformat()

SOURCES = {
    "PXD046566_proteome_quantification_matrix.xlsx":
        "https://download.iprox.org/IPX0002629000/IPX0002629003/Proteome%20quantification%20matrix%20of%20CRC%20samples.xlsx",
    "PXD040451_author_supplement.zip":
        "https://mdpi-res.com/d_attachment/ijms/ijms-24-15412/article_deploy/ijms-24-15412-s001.zip",
    "PXD008436_author_accepted_manuscript.docx":
        "https://livrepository.liverpool.ac.uk/3018923/1/ACPaper_JoP_Revision%20150118.docx",
    "h.all.v6.2.symbols.gmt":
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/6.2/h.all.v6.2.symbols.gmt",
}

PMC_SOURCES = {
    "PXD046566_supplementary_table_S1.xlsx":
        "https://pmc.ncbi.nlm.nih.gov/articles/instance/10772406/bin/mmc2.xlsx",
    "PXD046566_supplementary_table_S8.xlsx":
        "https://pmc.ncbi.nlm.nih.gov/articles/instance/10772406/bin/mmc9.xlsx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_response(response: requests.Response, target: Path) -> None:
    response.raise_for_status()
    target.write_bytes(response.content)


def valid_existing_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() in {".xlsx", ".docx", ".zip"}:
        with path.open("rb") as handle:
            return handle.read(2) == b"PK"
    return True


def download(session: requests.Session, url: str, target: Path, verify: bool = True) -> None:
    if valid_existing_file(target):
        return
    response = session.get(url, timeout=180, verify=verify)
    write_response(response, target)


def download_supplement(session: requests.Session, url: str, target: Path) -> None:
    """Download a public supplement; an access challenge requires manual retrieval."""
    download(session, url, target)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "DeepHeal-article-code/1.0"})

    for filename, url in SOURCES.items():
        download(session, url, RAW_DIR / filename)

    for filename, url in PMC_SOURCES.items():
        download_supplement(session, url, RAW_DIR / filename)

    archive = RAW_DIR / "PXD040451_author_supplement.zip"
    with zipfile.ZipFile(archive) as source_zip:
        extraction_map = {
            "Table S1.xlsx": "PXD040451_Table_S1_author_log2_matrix.xlsx",
            "File S1.xlsx": "PXD040451_File_S1_sample_exclusions.xlsx",
        }
        for member, output_name in extraction_map.items():
            target = RAW_DIR / output_name
            if not target.exists():
                target.write_bytes(source_zip.read(member))

    records = []
    url_lookup = {**SOURCES, **PMC_SOURCES}
    derived_from = {
        "PXD040451_Table_S1_author_log2_matrix.xlsx": "PXD040451_author_supplement.zip::Table S1.xlsx",
        "PXD040451_File_S1_sample_exclusions.xlsx": "PXD040451_author_supplement.zip::File S1.xlsx",
    }
    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_file() or path.name == "download_manifest.json":
            continue
        records.append({
            "file": path.name,
            "source_url": url_lookup.get(path.name, derived_from.get(path.name, "")),
            "access_date": ACCESS_DATE,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (RAW_DIR / "download_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
