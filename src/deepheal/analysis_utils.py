from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REQUIRED_OUT_DIRS = ["tables", "figures", "logs", "reports", "manifests", "checksums"]



def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--reference-root", default=None, type=Path)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    args.out_root = args.out_root.resolve()
    args.reference_root = (args.reference_root or args.repo_root.parent).resolve()
    return args


def ensure_out_dirs(out_root: Path) -> None:
    for name in REQUIRED_OUT_DIRS:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, required: bool = True) -> dict[str, Any]:
    path = path.resolve()
    exists = path.exists()
    record: dict[str, Any] = {
        "role": role,
        "required": bool(required),
        "path": str(path),
        "exists": bool(exists),
    }
    if exists and path.is_file():
        stat = path.stat()
        record.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
            }
        )
    return record


def set_plot_style(font_size: float = 8.5, dpi: int = 300) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="ticks", context="paper")


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def friendly_pathway_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unannotated"
    text = str(value).replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


def compact_pathway_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unannotated"
    text = str(value)
    for prefix in ["BIOCARTA_", "REACTOME_", "PID_", "KEGG_", "ST_", "NABA_", "SIG_"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace(".V1_DN", "").replace(".V1_UP", "")
    text = text.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


def external_group(therapeutic_effect: object) -> str:
    if therapeutic_effect is None or (isinstance(therapeutic_effect, float) and np.isnan(therapeutic_effect)):
        return "No external match"
    return "External antineoplastic" if "Antineoplastic" in str(therapeutic_effect) else "External non-antineoplastic"


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.9g}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)
