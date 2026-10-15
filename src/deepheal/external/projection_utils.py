from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT

from pathlib import Path
from collections import Counter
import pickle
import re
import numpy as np
import pandas as pd

def norm_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\xa0", " ").upper()

def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
