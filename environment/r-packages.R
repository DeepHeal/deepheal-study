# Direct dependencies of the standalone clinical and PRISM workflows.
cran_packages <- c("data.table", "readxl", "dplyr", "tidyr", "stringr", "ggplot2",
                   "circlize", "pROC", "patchwork", "ragg", "digest", "jsonlite",
                   "matrixStats", "officer", "ggplotify", "png")
install.packages(setdiff(cran_packages, rownames(installed.packages())))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) BiocManager::install("ComplexHeatmap")
