"""Project root and import locations shared by the article entry scripts."""
from pathlib import Path
import sys

ARTICLE_ROOT = Path(__file__).resolve().parents[2]
SOURCE = ARTICLE_ROOT / "src" / "deepheal"
for directory in [SOURCE, *sorted(SOURCE.iterdir())]:
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.append(str(directory))
