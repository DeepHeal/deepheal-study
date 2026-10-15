# Validation

- 77 Python source files parsed successfully; no machine-specific absolute paths, dated filenames, missing local Python file references or credential patterns were found.
- 9 retained model/helper classes and 4 core training/loss/attribution functions have unchanged Python syntax trees relative to the research source.
- Average and SupCon posterior means matched the retained implementations exactly on the same small CPU inputs and parameters (maximum absolute difference 0). Both branches produced finite gradients, and frozen-model reconstruction passed batch-consistency checks.
- Integrated-gradient attribution produced finite 16 by 3 output in a small CPU smoke test.
- 38 implementation modules imported successfully across the neural/AutoGluon and general-analysis environments; 13 selected command-line interfaces displayed help successfully. The installed AutoGluon environment lacks openpyxl; experimental workbook modules were checked in the general-analysis environment, and openpyxl is declared in requirements.txt.
- All three standalone R analysis files parsed under R 4.5.1. Full R figure/data execution was not repeated.
- The actual public HepG2 release passed its checksums and structural checks: 18 profiles, 18 by 9,960 features, 194/257-dimensional embeddings and 36 predictions. Recomputed recovery was 14/18 for Average and 15/18 for SupCon.

Neural smoke checks used Python 3.11.13 and PyTorch 2.8.0+cu128; AutoGluon checks used Python 3.11.14 and AutoGluon 1.5.0. A new dependency environment was not solved or installed. Complete VAE model-bank training, predictor fitting and all manuscript figures were not rerun. Exact numerical reproduction additionally requires the input tables and frozen artifacts listed in data/README.md.
