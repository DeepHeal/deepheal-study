"""Run the corresponding DeepHeal article workflow from any working directory."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_module("deepheal.pathways.pathway_response", run_name="__main__")
