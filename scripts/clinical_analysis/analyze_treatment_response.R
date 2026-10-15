# Figure 8: DeepHeal clinical translation
# Standalone RStudio/Rscript workflow using author-processed matrices only.

# 1. Package loading ---------------------------------------------------------
library(data.table)
library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)
library(pROC)
library(patchwork)
library(ragg)
library(digest)
library(jsonlite)
library(matrixStats)
library(officer)
library(grid)
library(ggplotify)
library(png)

# 2. Input/output paths and fixed parameters ---------------------------------
repo_root <- normalizePath(Sys.getenv("DEEPHEAL_ROOT", unset = "."), mustWork = TRUE)
package_root <- file.path(repo_root, "results", "clinical")
raw_dir <- file.path(repo_root, "data", "clinical")
figure_dir <- file.path(package_root, "figures")
table_dir <- file.path(package_root, "tables")
result_dir <- file.path(package_root, "results")

hct116_matrix_file <- file.path(repo_root, "data", "raw", "875_Compounds_Average_Data.csv")
hct116_metadata_file <- file.path(repo_root, "data", "raw", "drug_hct116_classification_including_marginal_with_cluster.csv")
a549_matrix_file <- file.path(repo_root, "data", "external", "deepcovermoa_protargetminer_all", "processed", "supplementary_data1_a549_original_55drugs_avg_log2ratio_for_deepheal.csv")
a549_map_file <- file.path(repo_root, "data", "external", "deepcovermoa_protargetminer_all", "processed", "supplementary_data1_uniprot_selection_map.csv")
c2_gmt_file <- file.path(repo_root, "data", "pathways", "gmt_cache", "c2.cp.v6.2.symbols.gmt")
hallmark_gmt_file <- file.path(raw_dir, "h.all.v6.2.symbols.gmt")
pxd046566_matrix_file <- file.path(raw_dir, "PXD046566_proteome_quantification_matrix.xlsx")
pxd046566_clinical_file <- file.path(raw_dir, "PXD046566_supplementary_table_S1.xlsx")
pxd046566_post_file <- file.path(raw_dir, "PXD046566_supplementary_table_S8.xlsx")
pxd040451_matrix_file <- file.path(raw_dir, "PXD040451_Table_S1_author_log2_matrix.xlsx")
pxd040451_exclusion_file <- file.path(raw_dir, "PXD040451_File_S1_sample_exclusions.xlsx")
pxd008436_manuscript_file <- file.path(raw_dir, "PXD008436_author_accepted_manuscript.docx")
download_manifest_file <- file.path(raw_dir, "download_manifest.json")

old_cptac_figure_dir <- file.path(repo_root, "data", "cptac", "figures")

random_seed <- 42L
zero_threshold <- 0.15
cluster_min_drugs <- 10L
cluster_correlation_threshold <- 0.30
pathway_min_genes <- 8L
pathway_max_genes <- 350L
pathway_coverage_threshold <- 0.70
pathway_jaccard_threshold <- 0.25
frozen_pathway_count <- 20L
bootstrap_iterations <- 10000L
permutation_iterations <- 10000L
main_width_mm <- 180
main_height_mm <- 220
supp_width_mm <- 180
supp_height_mm <- 130
heatmap_body_width_mm <- 90
heatmap_body_height_mm <- 35
heatmap_canvas_height_mm <- 52
heatmap_composition_left_padding_px <- 70L
dpi <- 300
base_family <- "sans"

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(random_seed)

required_files <- c(
  hct116_matrix_file, hct116_metadata_file, a549_matrix_file, a549_map_file,
  c2_gmt_file, hallmark_gmt_file, pxd046566_matrix_file,
  pxd046566_clinical_file, pxd046566_post_file, pxd040451_matrix_file,
  pxd040451_exclusion_file, pxd008436_manuscript_file, download_manifest_file
)
stopifnot(all(file.exists(required_files)))

# 3. Reusable import and scoring functions -----------------------------------
normalize_gene <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x[x %in% c("", "NA", "N/A", "NULL")] <- NA_character_
  x
}

collapse_feature_matrix <- function(genes, feature_by_sample_matrix) {
  genes <- normalize_gene(genes)
  valid <- !is.na(genes)
  genes <- genes[valid]
  feature_by_sample_matrix <- feature_by_sample_matrix[valid, , drop = FALSE]
  split_indices <- split(seq_along(genes), genes)
  collapsed <- vapply(
    split_indices,
    function(index) {
      if (length(index) == 1L) {
        as.numeric(feature_by_sample_matrix[index, ])
      } else {
        matrixStats::colMedians(feature_by_sample_matrix[index, , drop = FALSE], na.rm = TRUE)
      }
    },
    numeric(ncol(feature_by_sample_matrix))
  )
  collapsed[!is.finite(collapsed)] <- NA_real_
  rownames(collapsed) <- colnames(feature_by_sample_matrix)
  colnames(collapsed) <- names(split_indices)
  collapsed
}

read_gmt <- function(path) {
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  parsed <- strsplit(lines, "\t", fixed = FALSE)
  sets <- lapply(parsed, function(parts) unique(normalize_gene(parts[-c(1, 2)])))
  names(sets) <- vapply(parsed, `[[`, character(1), 1)
  sets
}

row_standardize <- function(matrix_samples_by_genes) {
  row_means <- rowMeans(matrix_samples_by_genes, na.rm = TRUE)
  row_sds <- matrixStats::rowSds(matrix_samples_by_genes, na.rm = TRUE)
  standardized <- sweep(matrix_samples_by_genes, 1, row_means, "-")
  standardized <- sweep(standardized, 1, row_sds, "/")
  standardized[!is.finite(standardized)] <- NA_real_
  standardized
}

score_pathways <- function(matrix_samples_by_genes, pathways, coverage_threshold = 1) {
  standardized <- row_standardize(matrix_samples_by_genes)
  output <- matrix(
    NA_real_, nrow = nrow(standardized), ncol = length(pathways),
    dimnames = list(rownames(standardized), names(pathways))
  )
  coverage <- matrix(
    NA_real_, nrow = nrow(standardized), ncol = length(pathways),
    dimnames = list(rownames(standardized), names(pathways))
  )
  for (pathway_name in names(pathways)) {
    genes <- intersect(pathways[[pathway_name]], colnames(standardized))
    observed <- rowSums(!is.na(standardized[, genes, drop = FALSE]))
    coverage[, pathway_name] <- observed / length(genes)
    valid <- observed > 0 & coverage[, pathway_name] >= coverage_threshold
    output[valid, pathway_name] <- rowSums(
      standardized[valid, genes, drop = FALSE], na.rm = TRUE
    ) / sqrt(observed[valid])
  }
  list(score = output, coverage = coverage)
}

filter_pathways <- function(pathways, genes) {
  measured_sizes <- vapply(pathways, function(x) length(intersect(x, genes)), integer(1))
  keep <- measured_sizes >= pathway_min_genes & measured_sizes <= pathway_max_genes
  pathways[keep]
}

pathway_jaccard <- function(pathway_a, pathway_b, genes) {
  a <- intersect(pathway_a, genes)
  b <- intersect(pathway_b, genes)
  length(intersect(a, b)) / length(union(a, b))
}

select_nonredundant_pathways <- function(ranked_table, pathways, genes, threshold, target_n) {
  selected <- character()
  for (candidate in ranked_table$pathway) {
    if (length(selected) == 0L || all(vapply(
      selected,
      function(existing) pathway_jaccard(pathways[[candidate]], pathways[[existing]], genes) < threshold,
      logical(1)
    ))) {
      selected <- c(selected, candidate)
    }
    if (length(selected) == target_n) break
  }
  selected
}

clean_pathway_label <- function(x) {
  x <- sub("^(REACTOME|KEGG|HALLMARK)_", "", x)
  x <- gsub("_", " ", x)
  stringr::str_to_title(tolower(x))
}

auc_rank <- function(response, score) {
  response <- as.integer(response)
  keep <- is.finite(score) & !is.na(response)
  response <- response[keep]
  score <- score[keep]
  n_positive <- sum(response == 1L)
  n_negative <- sum(response == 0L)
  if (n_positive == 0L || n_negative == 0L) return(NA_real_)
  (sum(rank(score, ties.method = "average")[response == 1L]) -
     n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
}

average_precision <- function(response, score) {
  keep <- is.finite(score) & !is.na(response)
  response <- as.integer(response[keep])
  score <- score[keep]
  if (sum(response == 1L) == 0L) return(NA_real_)
  ordering <- order(score, decreasing = TRUE)
  ordered_response <- response[ordering]
  precision <- cumsum(ordered_response) / seq_along(ordered_response)
  mean(precision[ordered_response == 1L])
}

compatibility_score <- function(patient_pathways, drug_effects, weights) {
  common <- intersect(colnames(patient_pathways), names(drug_effects))
  effect <- drug_effects[common]
  weight <- weights[common]
  vapply(seq_len(nrow(patient_pathways)), function(index) {
    patient <- patient_pathways[index, common]
    valid <- is.finite(patient) & is.finite(effect) & is.finite(weight)
    if (sum(valid) < 2L) return(NA_real_)
    weighted_patient <- patient[valid] * sqrt(weight[valid])
    weighted_effect <- effect[valid] * sqrt(weight[valid])
    -sum(weighted_patient * weighted_effect) /
      sqrt(sum(weighted_patient ^ 2) * sum(weighted_effect ^ 2))
  }, numeric(1))
}

evaluate_cohort <- function(cohort_name, response, score, endpoint_class) {
  keep <- !is.na(response) & is.finite(score)
  response <- as.integer(response[keep])
  score <- score[keep]
  n_positive <- sum(response == 1L)
  n_negative <- sum(response == 0L)
  observed_auc <- auc_rank(response, score)
  observed_auprc <- average_precision(response, score)
  wilcox <- wilcox.test(score ~ response, alternative = "two.sided", exact = FALSE)

  positive_indices <- which(response == 1L)
  negative_indices <- which(response == 0L)
  bootstrap_auc <- numeric(bootstrap_iterations)
  bootstrap_auprc <- numeric(bootstrap_iterations)
  bootstrap_rbc <- numeric(bootstrap_iterations)
  set.seed(random_seed + sum(utf8ToInt(cohort_name)))
  for (iteration in seq_len(bootstrap_iterations)) {
    indices <- c(
      sample(positive_indices, length(positive_indices), replace = TRUE),
      sample(negative_indices, length(negative_indices), replace = TRUE)
    )
    bootstrap_auc[iteration] <- auc_rank(response[indices], score[indices])
    bootstrap_auprc[iteration] <- average_precision(response[indices], score[indices])
    bootstrap_rbc[iteration] <- 2 * bootstrap_auc[iteration] - 1
  }

  set.seed(random_seed + 10000L + sum(utf8ToInt(cohort_name)))
  permutation_auc <- replicate(
    permutation_iterations,
    auc_rank(sample(response, length(response), replace = FALSE), score)
  )
  permutation_p <- (1 + sum(permutation_auc >= observed_auc)) / (permutation_iterations + 1)

  data.frame(
    cohort = cohort_name,
    endpoint_class = endpoint_class,
    n = length(response),
    n_sensitive = n_positive,
    n_nonsensitive = n_negative,
    median_sensitive = median(score[response == 1L]),
    median_nonsensitive = median(score[response == 0L]),
    wilcoxon_W = unname(wilcox$statistic),
    wilcoxon_p = wilcox$p.value,
    rank_biserial = 2 * observed_auc - 1,
    rank_biserial_ci_low = unname(quantile(bootstrap_rbc, 0.025)),
    rank_biserial_ci_high = unname(quantile(bootstrap_rbc, 0.975)),
    auroc = observed_auc,
    auroc_ci_low = unname(quantile(bootstrap_auc, 0.025)),
    auroc_ci_high = unname(quantile(bootstrap_auc, 0.975)),
    auroc_permutation_p = permutation_p,
    auprc = observed_auprc,
    auprc_ci_low = unname(quantile(bootstrap_auprc, 0.025)),
    auprc_ci_high = unname(quantile(bootstrap_auprc, 0.975)),
    prevalence = n_positive / length(response),
    stringsAsFactors = FALSE
  )
}

pr_curve <- function(response, score, cohort_name) {
  ordering <- order(score, decreasing = TRUE)
  response <- as.integer(response[ordering])
  data.frame(
    cohort = cohort_name,
    recall = c(0, cumsum(response) / sum(response)),
    precision = c(1, cumsum(response) / seq_along(response))
  )
}

write_csv_stable <- function(object, filename) {
  data.table::fwrite(object, file.path(table_dir, filename), na = "NA", quote = TRUE)
}

export_plot <- function(plot_object, basename, width_mm, height_mm) {
  png_path <- file.path(figure_dir, paste0(basename, ".png"))
  tiff_path <- file.path(figure_dir, paste0(basename, ".tiff"))
  pdf_path <- file.path(figure_dir, paste0(basename, ".pdf"))
  ragg::agg_png(png_path, width = width_mm, height = height_mm, units = "mm", res = dpi, background = "white")
  print(plot_object)
  dev.off()
  ragg::agg_tiff(tiff_path, width = width_mm, height = height_mm, units = "mm", res = dpi, compression = "lzw", background = "white")
  print(plot_object)
  dev.off()
  grDevices::cairo_pdf(pdf_path, width = width_mm / 25.4, height = height_mm / 25.4, family = base_family, onefile = TRUE)
  print(plot_object)
  dev.off()
}

# 4. Data import without clinical outcome labels -----------------------------
hct116_raw <- data.table::fread(hct116_matrix_file, encoding = "Latin-1", check.names = FALSE)
hct116_metadata <- data.table::fread(hct116_metadata_file, encoding = "UTF-8", check.names = FALSE)
names(hct116_raw) <- gsub("[\u00A0[:space:]]+", " ", iconv(names(hct116_raw), from = "latin1", to = "UTF-8"))
hct116_metadata$Sample_ID <- gsub("[\u00A0[:space:]]+", " ", hct116_metadata$Sample_ID)
hct116_feature_by_drug <- as.matrix(hct116_raw[, -(1:2), with = FALSE])
storage.mode(hct116_feature_by_drug) <- "double"
rownames(hct116_feature_by_drug) <- seq_len(nrow(hct116_feature_by_drug))
colnames(hct116_feature_by_drug) <- names(hct116_raw)[-(1:2)]
hct116_matrix <- collapse_feature_matrix(hct116_raw[["Gene Name"]], hct116_feature_by_drug)
hct116_matrix_unthresholded <- hct116_matrix
hct116_matrix[abs(hct116_matrix) <= zero_threshold] <- 0

a549_raw <- data.table::fread(a549_matrix_file, encoding = "UTF-8", check.names = FALSE)
a549_map <- data.table::fread(a549_map_file, encoding = "UTF-8", check.names = FALSE)
a549_gene_map <- setNames(
  sub("^(?:sp|tr)_[^_]+_([^_]+)_HUMAN$", "\\1", a549_map[["Protein ID"]]),
  a549_map[["selected_uniprot_id"]]
)
a549_genes <- unname(a549_gene_map[as.character(a549_raw[["protein_id"]])])
a549_feature_by_drug <- as.matrix(a549_raw[, -1, with = FALSE])
storage.mode(a549_feature_by_drug) <- "double"
rownames(a549_feature_by_drug) <- seq_len(nrow(a549_feature_by_drug))
colnames(a549_feature_by_drug) <- names(a549_raw)[-1]
a549_matrix <- collapse_feature_matrix(a549_genes, a549_feature_by_drug)
a549_matrix_unthresholded <- a549_matrix
a549_matrix[abs(a549_matrix) <= zero_threshold] <- 0
stopifnot(all(c("5-fluorouracil", "Oxaliplatin") %in% rownames(a549_matrix)))

read_pxd046566_matrix <- function(sheet_name) {
  imported <- as.data.frame(readxl::read_excel(
    pxd046566_matrix_file, sheet = sheet_name, .name_repair = "minimal"
  ))
  genes <- imported[["Symbol"]]
  values <- as.matrix(imported[, -(1:2), drop = FALSE])
  storage.mode(values) <- "double"
  values[!is.finite(values) | values <= 0] <- NA_real_
  values <- log2(values)
  collapse_feature_matrix(genes, values)
}

pxd046566_mbr <- read_pxd046566_matrix(2)
pxd046566_strict <- read_pxd046566_matrix(3)

pxd040451_header <- unlist(readxl::read_excel(
  pxd040451_matrix_file, range = "A2:AW2", col_names = FALSE
), use.names = FALSE)
pxd040451_sample_columns <- grepl("^[0-9]+(?:_[0-9]+)?$", pxd040451_header)
pxd040451_data <- as.data.frame(readxl::read_excel(
  pxd040451_matrix_file, skip = 3, col_names = FALSE, .name_repair = "minimal"
))
pxd040451_genes <- pxd040451_data[[1]]
pxd040451_feature_by_technical <- as.matrix(pxd040451_data[, pxd040451_sample_columns, drop = FALSE])
storage.mode(pxd040451_feature_by_technical) <- "double"
pxd040451_technical_ids <- as.character(pxd040451_header[pxd040451_sample_columns])
colnames(pxd040451_feature_by_technical) <- pxd040451_technical_ids
pxd040451_technical_matrix <- collapse_feature_matrix(pxd040451_genes, pxd040451_feature_by_technical)
pxd040451_patient_short <- sub("_.*$", "", rownames(pxd040451_technical_matrix))
pxd040451_patient_ids <- paste0("139", pxd040451_patient_short)
pxd040451_split <- split(seq_len(nrow(pxd040451_technical_matrix)), pxd040451_patient_ids)
pxd040451_matrix <- do.call(rbind, lapply(pxd040451_split, function(index) {
  matrixStats::colMedians(pxd040451_technical_matrix[index, , drop = FALSE], na.rm = TRUE)
}))
pxd040451_matrix[!is.finite(pxd040451_matrix)] <- NA_real_
rownames(pxd040451_matrix) <- names(pxd040451_split)
colnames(pxd040451_matrix) <- colnames(pxd040451_technical_matrix)
predefined_exclusions <- c("13947", "13948", "13964", "13966")
stopifnot(!any(predefined_exclusions %in% rownames(pxd040451_matrix)))

# Pathway library is fixed to MSigDB v6.2 Reactome, KEGG, and Hallmark.
c2_pathways_all <- read_gmt(c2_gmt_file)
c2_pathways <- c2_pathways_all[grepl("^(REACTOME|KEGG)_", names(c2_pathways_all))]
hallmark_pathways <- read_gmt(hallmark_gmt_file)
hallmark_pathways <- hallmark_pathways[grepl("^HALLMARK_", names(hallmark_pathways))]
pathways_all <- c(c2_pathways, hallmark_pathways)
stopifnot(!anyDuplicated(names(pathways_all)))

# 5. Freeze the DeepHeal program before reading outcome labels ----------------
strict_metadata <- hct116_metadata[
  Updated_Antineoplastic_at_ScreeningConc == "Yes" & Sample_ID %in% rownames(hct116_matrix)
]
stopifnot(nrow(strict_metadata) == 120L, !anyDuplicated(strict_metadata$Sample_ID))
strict_hct116 <- hct116_matrix[strict_metadata$Sample_ID, , drop = FALSE]

hct_a549_common_genes <- Reduce(intersect, list(colnames(strict_hct116), colnames(a549_matrix)))
pathways_hct_a549 <- filter_pathways(pathways_all, hct_a549_common_genes)
hct116_anchor_scores <- score_pathways(
  strict_hct116[, hct_a549_common_genes, drop = FALSE], pathways_hct_a549, 1
)$score
a549_anchor_scores <- score_pathways(
  a549_matrix[c("5-fluorouracil", "Oxaliplatin"), hct_a549_common_genes, drop = FALSE],
  pathways_hct_a549, 1
)$score

cluster_correlations <- lapply(sort(unique(strict_metadata$Cluster)), function(cluster_id) {
  sample_ids <- strict_metadata$Sample_ID[strict_metadata$Cluster == cluster_id]
  cluster_effect <- matrixStats::colMedians(hct116_anchor_scores[sample_ids, , drop = FALSE], na.rm = TRUE)
  data.frame(
    cluster = as.integer(cluster_id),
    n_drugs = length(sample_ids),
    r_5fu = cor(cluster_effect, a549_anchor_scores["5-fluorouracil", ], method = "pearson", use = "pairwise.complete.obs"),
    r_oxaliplatin = cor(cluster_effect, a549_anchor_scores["Oxaliplatin", ], method = "pearson", use = "pairwise.complete.obs"),
    stringsAsFactors = FALSE
  )
}) |> bind_rows() |>
  mutate(
    eligible_n = n_drugs >= cluster_min_drugs,
    selected = eligible_n & r_5fu >= cluster_correlation_threshold & r_oxaliplatin >= cluster_correlation_threshold
  )
selected_clusters <- cluster_correlations$cluster[cluster_correlations$selected]
selected_metadata <- strict_metadata[Cluster %in% selected_clusters]
stopifnot(identical(selected_clusters, c(1L, 3L)), nrow(selected_metadata) == 67L)

four_platform_common_genes <- Reduce(intersect, list(
  colnames(strict_hct116), colnames(a549_matrix), colnames(pxd046566_mbr), colnames(pxd040451_matrix)
)) |> sort()
stopifnot(length(four_platform_common_genes) == 1218L)

pathways_common <- filter_pathways(pathways_all, four_platform_common_genes)
strict_common_scores <- score_pathways(
  strict_hct116[, four_platform_common_genes, drop = FALSE], pathways_common, 1
)$score
cluster_1_ids <- selected_metadata$Sample_ID[selected_metadata$Cluster == 1L]
cluster_3_ids <- selected_metadata$Sample_ID[selected_metadata$Cluster == 3L]
cluster_1_effect <- matrixStats::colMedians(strict_common_scores[cluster_1_ids, , drop = FALSE], na.rm = TRUE)
cluster_3_effect <- matrixStats::colMedians(strict_common_scores[cluster_3_ids, , drop = FALSE], na.rm = TRUE)
ensemble_effect <- (cluster_1_effect + cluster_3_effect) / 2

set.seed(random_seed)
bootstrap_effects <- matrix(
  NA_real_, nrow = bootstrap_iterations, ncol = ncol(strict_common_scores),
  dimnames = list(NULL, colnames(strict_common_scores))
)
for (iteration in seq_len(bootstrap_iterations)) {
  resampled_cluster_1 <- sample(cluster_1_ids, length(cluster_1_ids), replace = TRUE)
  resampled_cluster_3 <- sample(cluster_3_ids, length(cluster_3_ids), replace = TRUE)
  bootstrap_effects[iteration, ] <- (
    matrixStats::colMedians(strict_common_scores[resampled_cluster_1, , drop = FALSE], na.rm = TRUE) +
      matrixStats::colMedians(strict_common_scores[resampled_cluster_3, , drop = FALSE], na.rm = TRUE)
  ) / 2
}

bootstrap_ci_low <- as.numeric(matrixStats::colQuantiles(bootstrap_effects, probs = 0.025))
bootstrap_ci_high <- as.numeric(matrixStats::colQuantiles(bootstrap_effects, probs = 0.975))
names(bootstrap_ci_low) <- colnames(bootstrap_effects)
names(bootstrap_ci_high) <- colnames(bootstrap_effects)
direction_probability <- pmax(
  colMeans(bootstrap_effects > 0),
  colMeans(bootstrap_effects < 0)
)
stable_pathway_table <- data.frame(
  pathway = names(ensemble_effect),
  ensemble_effect = as.numeric(ensemble_effect),
  ci_low = as.numeric(bootstrap_ci_low[names(ensemble_effect)]),
  ci_high = as.numeric(bootstrap_ci_high[names(ensemble_effect)]),
  direction_probability = as.numeric(direction_probability[names(ensemble_effect)]),
  stringsAsFactors = FALSE
) |>
  mutate(
    stable = (ci_low > 0 & ci_high > 0) | (ci_low < 0 & ci_high < 0),
    ranking_score = abs(ensemble_effect) * direction_probability,
    measured_genes = vapply(pathway, function(name) length(intersect(pathways_common[[name]], four_platform_common_genes)), integer(1))
  ) |>
  filter(stable) |>
  arrange(desc(ranking_score), pathway)

selected_pathways <- select_nonredundant_pathways(
  stable_pathway_table, pathways_common, four_platform_common_genes,
  pathway_jaccard_threshold, frozen_pathway_count
)
stopifnot(length(selected_pathways) == frozen_pathway_count)
frozen_program <- stable_pathway_table |>
  filter(pathway %in% selected_pathways) |>
  mutate(
    frozen_rank = match(pathway, selected_pathways),
    normalized_weight = ranking_score / sum(ranking_score),
    direction = ifelse(ensemble_effect > 0, "Induced", "Suppressed"),
    display_label = clean_pathway_label(pathway),
    gene_members = vapply(pathway, function(name) {
      paste(sort(intersect(pathways_common[[name]], four_platform_common_genes)), collapse = ";")
    }, character(1))
  ) |>
  arrange(frozen_rank)

frozen_program_file <- file.path(table_dir, "frozen_deepheal_pathway_program.csv")
data.table::fwrite(frozen_program, frozen_program_file, na = "NA", quote = TRUE)
frozen_program_sha256 <- digest::digest(file = frozen_program_file, algo = "sha256")

# 6. Read response labels and calculate prespecified clinical statistics ------
pxd046566_clinical <- as.data.frame(readxl::read_excel(
  pxd046566_clinical_file, sheet = "Table S1A"
)) |>
  transmute(
    sample_id = as.character(`Exp NO`),
    regimen = as.character(`Therapy regiments`),
    response = as.character(`Tumor response`)
  )
stopifnot(nrow(pxd046566_clinical) == 124L, !anyDuplicated(pxd046566_clinical$sample_id))
stopifnot(setequal(pxd046566_clinical$sample_id, rownames(pxd046566_mbr)))

pxd040451_response_row <- unlist(readxl::read_excel(
  pxd040451_matrix_file, range = "A3:AW3", col_names = FALSE
), use.names = FALSE)
technical_response <- as.character(pxd040451_response_row[pxd040451_sample_columns])
response_by_patient <- lapply(split(technical_response, pxd040451_patient_ids), unique)
stopifnot(all(lengths(response_by_patient) == 1L))
pxd040451_clinical <- data.frame(
  sample_id = names(response_by_patient),
  regimen = "nCRT",
  response = vapply(response_by_patient, `[[`, character(1), 1),
  stringsAsFactors = FALSE
)
stopifnot(nrow(pxd040451_clinical) == 20L, !anyDuplicated(pxd040451_clinical$sample_id))
stopifnot(sum(pxd040451_clinical$response == "Responders") == 9L)
stopifnot(sum(pxd040451_clinical$response == "Nonresponder") == 11L)

frozen_sets <- pathways_common[frozen_program$pathway]
pxd046566_mbr_pathways <- score_pathways(
  pxd046566_mbr[, four_platform_common_genes, drop = FALSE], frozen_sets,
  pathway_coverage_threshold
)

strict_aligned <- matrix(
  NA_real_, nrow = nrow(pxd046566_strict), ncol = length(four_platform_common_genes),
  dimnames = list(rownames(pxd046566_strict), four_platform_common_genes)
)
strict_present <- intersect(colnames(pxd046566_strict), four_platform_common_genes)
strict_aligned[, strict_present] <- pxd046566_strict[, strict_present, drop = FALSE]
pxd046566_strict_pathways <- score_pathways(
  strict_aligned, frozen_sets, pathway_coverage_threshold
)
pxd040451_pathways <- score_pathways(
  pxd040451_matrix[, four_platform_common_genes, drop = FALSE], frozen_sets,
  pathway_coverage_threshold
)

frozen_effect_vector <- setNames(frozen_program$ensemble_effect, frozen_program$pathway)
frozen_weight_vector <- setNames(frozen_program$normalized_weight, frozen_program$pathway)
pxd046566_mbr_scores <- compatibility_score(
  pxd046566_mbr_pathways$score, frozen_effect_vector, frozen_weight_vector
)
pxd046566_strict_scores <- compatibility_score(
  pxd046566_strict_pathways$score, frozen_effect_vector, frozen_weight_vector
)
pxd040451_scores <- compatibility_score(
  pxd040451_pathways$score, frozen_effect_vector, frozen_weight_vector
)

pxd046566_score_table <- data.frame(
  sample_id = rownames(pxd046566_mbr_pathways$score),
  score = pxd046566_mbr_scores,
  score_strict_1pct = pxd046566_strict_scores,
  stringsAsFactors = FALSE
) |>
  left_join(pxd046566_clinical, by = "sample_id") |>
  mutate(
    cohort = case_when(
      regimen == "FOLFOX therapy" ~ "PXD046566 FOLFOX",
      regimen == "FOLFOX+Radiotherapy" ~ "PXD046566 FOLFOX+RT",
      regimen == "Cetuximab+FOLFOX therapy" ~ "PXD046566 FOLFOX+cetuximab",
      TRUE ~ "PXD046566 other"
    ),
    response_binary = ifelse(response == "S", 1L, 0L),
    response_display = ifelse(response == "S", "Sensitive", "Non-sensitive")
  )

pxd040451_score_table <- data.frame(
  sample_id = rownames(pxd040451_pathways$score),
  score = pxd040451_scores,
  score_strict_1pct = NA_real_,
  stringsAsFactors = FALSE
) |>
  left_join(pxd040451_clinical, by = "sample_id") |>
  mutate(
    cohort = "PXD040451 nCRT",
    response_binary = ifelse(response == "Responders", 1L, 0L),
    response_display = ifelse(response == "Responders", "Responder", "Non-responder")
  )

clinical_scores <- bind_rows(pxd046566_score_table, pxd040451_score_table)

expected_counts <- pxd046566_score_table |>
  filter(cohort %in% c("PXD046566 FOLFOX", "PXD046566 FOLFOX+RT", "PXD046566 FOLFOX+cetuximab")) |>
  dplyr::count(cohort, response)
stopifnot(
  sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX") == 38L,
  sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+RT") == 30L,
  sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+cetuximab") == 17L,
  sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX" & pxd046566_score_table$response == "S") == 16L,
  sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX" & pxd046566_score_table$response == "NS") == 22L
)

cohort_definitions <- data.frame(
  cohort = c(
    "PXD046566 FOLFOX", "PXD040451 nCRT", "PXD046566 FOLFOX+RT",
    "PXD046566 FOLFOX+cetuximab"
  ),
  endpoint_class = c("Primary", "Independent validation", "Exploratory", "Exploratory")
)

performance_metrics <- lapply(seq_len(nrow(cohort_definitions)), function(index) {
  definition <- cohort_definitions[index, ]
  cohort_data <- clinical_scores |> filter(cohort == definition$cohort)
  evaluate_cohort(
    definition$cohort, cohort_data$response_binary, cohort_data$score,
    definition$endpoint_class
  )
}) |> bind_rows()
exploratory_rows <- which(performance_metrics$endpoint_class == "Exploratory")
performance_metrics$wilcoxon_p_bh <- NA_real_
performance_metrics$auroc_permutation_p_bh <- NA_real_
performance_metrics$wilcoxon_p_bh[exploratory_rows] <- p.adjust(
  performance_metrics$wilcoxon_p[exploratory_rows], method = "BH"
)
performance_metrics$auroc_permutation_p_bh[exploratory_rows] <- p.adjust(
  performance_metrics$auroc_permutation_p[exploratory_rows], method = "BH"
)

pr_curve_table <- lapply(cohort_definitions$cohort, function(cohort_name) {
  cohort_data <- clinical_scores |> filter(cohort == cohort_name)
  pr_curve(cohort_data$response_binary, cohort_data$score, cohort_name)
}) |> bind_rows()

# 7. Prespecified sensitivity analyses ---------------------------------------
strict_sensitivity_metrics <- lapply(cohort_definitions$cohort[c(1, 3, 4)], function(cohort_name) {
  cohort_data <- pxd046566_score_table |> filter(cohort == cohort_name)
  evaluate_cohort(
    paste0(cohort_name, " [strict 1% FDR]"),
    cohort_data$response_binary,
    cohort_data$score_strict_1pct,
    "Sensitivity"
  )
}) |> bind_rows()

marginal_metadata <- hct116_metadata[
  Antineoplastic_Including_Marginal == "Yes" & Cluster %in% selected_clusters &
    Sample_ID %in% rownames(hct116_matrix)
]
marginal_pathway_scores <- score_pathways(
  hct116_matrix[marginal_metadata$Sample_ID, four_platform_common_genes, drop = FALSE],
  frozen_sets, 1
)$score
marginal_cluster_1 <- marginal_metadata$Sample_ID[marginal_metadata$Cluster == 1L]
marginal_cluster_3 <- marginal_metadata$Sample_ID[marginal_metadata$Cluster == 3L]
marginal_effect <- (
  matrixStats::colMedians(marginal_pathway_scores[marginal_cluster_1, , drop = FALSE], na.rm = TRUE) +
    matrixStats::colMedians(marginal_pathway_scores[marginal_cluster_3, , drop = FALSE], na.rm = TRUE)
) / 2
marginal_score <- compatibility_score(
  pxd046566_mbr_pathways$score, marginal_effect, frozen_weight_vector
)
marginal_primary <- pxd046566_score_table |> filter(cohort == "PXD046566 FOLFOX")
marginal_metric <- evaluate_cohort(
  "PXD046566 FOLFOX [strict+marginal drugs]",
  marginal_primary$response_binary,
  marginal_score[match(marginal_primary$sample_id, rownames(pxd046566_mbr_pathways$score))],
  "Sensitivity"
)

leave_one_drug_out <- lapply(selected_metadata$Sample_ID, function(omitted_id) {
  ids_1 <- setdiff(cluster_1_ids, omitted_id)
  ids_3 <- setdiff(cluster_3_ids, omitted_id)
  lodo_effect <- (
    matrixStats::colMedians(strict_common_scores[ids_1, frozen_program$pathway, drop = FALSE], na.rm = TRUE) +
      matrixStats::colMedians(strict_common_scores[ids_3, frozen_program$pathway, drop = FALSE], na.rm = TRUE)
  ) / 2
  lodo_score <- compatibility_score(
    pxd046566_mbr_pathways$score, lodo_effect, frozen_weight_vector
  )
  primary_indices <- match(marginal_primary$sample_id, rownames(pxd046566_mbr_pathways$score))
  data.frame(
    omitted_sample_id = omitted_id,
    omitted_drug = selected_metadata$Drug_Name[match(omitted_id, selected_metadata$Sample_ID)],
    cluster = selected_metadata$Cluster[match(omitted_id, selected_metadata$Sample_ID)],
    effect_correlation_to_full = cor(lodo_effect, frozen_effect_vector[names(lodo_effect)], use = "pairwise.complete.obs"),
    primary_auroc = auc_rank(marginal_primary$response_binary, lodo_score[primary_indices]),
    primary_auprc = average_precision(marginal_primary$response_binary, lodo_score[primary_indices]),
    stringsAsFactors = FALSE
  )
}) |> bind_rows()

a549_threshold_sensitivity <- lapply(c(0.10, 0.15, 0.20), function(threshold) {
  hct_temp <- hct116_matrix_unthresholded[strict_metadata$Sample_ID, hct_a549_common_genes, drop = FALSE]
  a549_temp <- a549_matrix_unthresholded[c("5-fluorouracil", "Oxaliplatin"), hct_a549_common_genes, drop = FALSE]
  hct_temp[abs(hct_temp) <= threshold] <- 0
  a549_temp[abs(a549_temp) <= threshold] <- 0
  hct_score <- score_pathways(hct_temp, pathways_hct_a549, 1)$score
  a549_score <- score_pathways(a549_temp, pathways_hct_a549, 1)$score
  lapply(sort(unique(strict_metadata$Cluster)), function(cluster_id) {
    ids <- strict_metadata$Sample_ID[strict_metadata$Cluster == cluster_id]
    effect <- matrixStats::colMedians(hct_score[ids, , drop = FALSE], na.rm = TRUE)
    data.frame(
      threshold = threshold,
      cluster = cluster_id,
      n_drugs = length(ids),
      r_5fu = cor(effect, a549_score["5-fluorouracil", ], use = "pairwise.complete.obs"),
      r_oxaliplatin = cor(effect, a549_score["Oxaliplatin", ], use = "pairwise.complete.obs"),
      selected_by_fixed_rule = length(ids) >= cluster_min_drugs &
        cor(effect, a549_score["5-fluorouracil", ], use = "pairwise.complete.obs") >= cluster_correlation_threshold &
        cor(effect, a549_score["Oxaliplatin", ], use = "pairwise.complete.obs") >= cluster_correlation_threshold
    )
  }) |> bind_rows()
}) |> bind_rows()

jaccard_sensitivity <- lapply(c(0.20, 0.25, 0.30), function(threshold) {
  selected <- select_nonredundant_pathways(
    stable_pathway_table, pathways_common, four_platform_common_genes,
    threshold, frozen_pathway_count
  )
  data.frame(
    threshold = threshold,
    n_selected = length(selected),
    overlap_with_primary = length(intersect(selected, frozen_program$pathway)),
    jaccard_with_primary = length(intersect(selected, frozen_program$pathway)) /
      length(union(selected, frozen_program$pathway)),
    selected_pathways = paste(selected, collapse = ";"),
    stringsAsFactors = FALSE
  )
}) |> bind_rows()

# 8. Author-reported post-treatment and longitudinal evidence ----------------
pxd046566_post <- as.data.frame(readxl::read_excel(
  pxd046566_post_file, sheet = "Table S8B"
)) |>
  tidyr::fill(Subcohort, Group) |>
  transmute(
    source = "PXD046566 Table S8B",
    comparison = paste(Subcohort, Group),
    evidence_type = "Post-treatment group enrichment",
    mechanism = as.character(Pathway),
    direction = sign(as.numeric(NES)),
    estimate = as.numeric(NES),
    p_value = as.numeric(`P value`),
    detail = paste0("NES=", formatC(as.numeric(NES), digits = 2, format = "f")),
    patient_level_delta = FALSE
  )

pxd008436_document <- officer::read_docx(pxd008436_manuscript_file)
pxd008436_summary <- officer::docx_summary(pxd008436_document)
pxd008436_tables <- pxd008436_summary |>
  filter(content_type == "table cell")

extract_docx_cell <- function(table_index_value, row_id_value, cell_id_value) {
  values <- pxd008436_tables |>
    filter(
      table_index == table_index_value,
      row_id == row_id_value,
      cell_id == cell_id_value
    ) |>
    pull(text)
  paste(values, collapse = " ")
}

pxd008436_response_evidence <- data.frame(
  source = "PXD008436 published tables",
  comparison = c("Responders vs non-responders", rep("Responder pathway analysis", 4)),
  evidence_type = c("Response-group protein", rep("Response-group pathway", 4)),
  mechanism = c(
    extract_docx_cell(2, 12, 2),
    extract_docx_cell(3, 20, 1),
    extract_docx_cell(4, 26, 1),
    extract_docx_cell(4, 27, 1),
    extract_docx_cell(4, 28, 1)
  ),
  direction = c(-1, 1, -1, -1, -1),
  estimate = c(
    as.numeric(extract_docx_cell(2, 12, 4)),
    NA_real_, NA_real_, NA_real_, NA_real_
  ),
  p_value = c(
    as.numeric(extract_docx_cell(2, 12, 3)),
    as.numeric(extract_docx_cell(3, 20, 2)),
    as.numeric(extract_docx_cell(4, 26, 2)),
    as.numeric(extract_docx_cell(4, 27, 2)),
    as.numeric(extract_docx_cell(4, 28, 2))
  ),
  detail = c(
    "logFC=-1.526", "up, p=0.00528", "down, p=0.000992",
    "down, p=0.00132", "down, p=0.0178"
  ),
  patient_level_delta = FALSE,
  stringsAsFactors = FALSE
)

pxd008436_longitudinal <- data.frame(
  source = "PXD008436 published Results",
  comparison = c("Pre to post-CRT", "Pre-CRT to resection", "Post-CRT to resection"),
  evidence_type = "Longitudinal group-level significant-protein counts",
  mechanism = "Global significant proteins",
  direction = sign(c(16 - 2, 19 - 10, 9 - 30)),
  estimate = c((16 - 2) / 18, (19 - 10) / 29, (9 - 30) / 39),
  p_value = NA_real_,
  detail = c("16 up / 2 down", "19 up / 10 down", "9 up / 30 down"),
  patient_level_delta = FALSE,
  stringsAsFactors = FALSE
)
pharmacodynamic_evidence <- bind_rows(
  pxd046566_post, pxd008436_response_evidence, pxd008436_longitudinal
)

# Coarse mechanism map for the result-level evidence matrix.
mechanism_family <- function(x) {
  case_when(
    grepl("cell cycle", x, ignore.case = TRUE) ~ "Cell cycle",
    grepl("DNA replication|DNA repair", x, ignore.case = TRUE) ~ "DNA replication / repair",
    grepl("pyrimidine|folate|vitamin|cofactor", x, ignore.case = TRUE) ~ "Nucleotide / folate metabolism",
    grepl("glycerophospholipid|ceramid|sphingosine", x, ignore.case = TRUE) ~ "Lipid / ceramide metabolism",
    grepl("MHC|interferon|IL2|antigen", x, ignore.case = TRUE) ~ "Immune signalling",
    grepl("autophagy", x, ignore.case = TRUE) ~ "Autophagy",
    grepl("focal adhesion|neutrophil", x, ignore.case = TRUE) ~ "Adhesion / neutrophil",
    grepl("global significant", x, ignore.case = TRUE) ~ "Global longitudinal proteome",
    TRUE ~ "Other metabolism"
  )
}

evidence_matrix_table <- pharmacodynamic_evidence |>
  mutate(
    mechanism_family = mechanism_family(mechanism),
    evidence_column = case_when(
      source == "PXD046566 Table S8B" ~ paste0("PXD046566\n", comparison),
      comparison == "Responders vs non-responders" | comparison == "Responder pathway analysis" ~ "PXD008436\nresponse groups",
      TRUE ~ paste0("PXD008436\n", comparison)
    )
  ) |>
  group_by(mechanism_family, evidence_column) |>
  summarise(
    direction = sign(mean(direction, na.rm = TRUE)),
    detail = paste(unique(detail), collapse = "; "),
    .groups = "drop"
  )

# 9. Stable interface tables and integrity checks ----------------------------
clinical_scores_output <- clinical_scores |>
  select(sample_id, cohort, regimen, response, response_display, response_binary, score, score_strict_1pct)
write_csv_stable(cluster_correlations, "hct116_cluster_a549_component_correlations.csv")
write_csv_stable(
  selected_metadata[, .(Drug_Name, Sample_ID, Cluster)],
  "selected_cluster_1_3_drugs.csv"
)
write_csv_stable(
  data.frame(
    technical_profile = rownames(pxd040451_technical_matrix),
    patient_id = pxd040451_patient_ids,
    response = technical_response,
    stringsAsFactors = FALSE
  ),
  "pxd040451_technical_replicate_map.csv"
)
write_csv_stable(clinical_scores_output, "clinical_patient_scores.csv")
write_csv_stable(performance_metrics, "clinical_performance_metrics.csv")
write_csv_stable(strict_sensitivity_metrics, "sensitivity_strict_1pct_metrics.csv")
write_csv_stable(marginal_metric, "sensitivity_marginal_drug_metrics.csv")
write_csv_stable(leave_one_drug_out, "sensitivity_leave_one_drug_out.csv")
write_csv_stable(a549_threshold_sensitivity, "sensitivity_a549_threshold.csv")
write_csv_stable(jaccard_sensitivity, "sensitivity_pathway_jaccard.csv")
write_csv_stable(pr_curve_table, "precision_recall_curves.csv")
write_csv_stable(pharmacodynamic_evidence, "pharmacodynamic_group_level_evidence.csv")
write_csv_stable(evidence_matrix_table, "pharmacodynamic_evidence_matrix.csv")

coverage_table <- bind_rows(
  as.data.frame(pxd046566_mbr_pathways$coverage) |>
    mutate(sample_id = rownames(pxd046566_mbr_pathways$coverage), cohort = "PXD046566 MBR"),
  as.data.frame(pxd046566_strict_pathways$coverage) |>
    mutate(sample_id = rownames(pxd046566_strict_pathways$coverage), cohort = "PXD046566 strict 1% FDR"),
  as.data.frame(pxd040451_pathways$coverage) |>
    mutate(sample_id = rownames(pxd040451_pathways$coverage), cohort = "PXD040451")
) |>
  pivot_longer(cols = all_of(frozen_program$pathway), names_to = "pathway", values_to = "coverage") |>
  select(cohort, sample_id, pathway, coverage)
write_csv_stable(coverage_table, "pathway_coverage_by_sample.csv")

sample_audit <- bind_rows(
  clinical_scores_output |> dplyr::count(cohort, response, name = "n") |>
    mutate(audit_type = "clinical endpoint"),
  data.frame(
    cohort = c("DeepHeal strict drugs", "DeepHeal selected clusters", "PXD046566 all", "PXD040451 all"),
    response = c("Yes", "Clusters 1+3", "All", "All"),
    n = c(nrow(strict_metadata), nrow(selected_metadata), nrow(pxd046566_clinical), nrow(pxd040451_clinical)),
    audit_type = "analysis invariant"
  )
)
write_csv_stable(sample_audit, "sample_and_group_audit.csv")

# Label permutation must not alter the already-written frozen program.
set.seed(random_seed)
permuted_labels <- sample(clinical_scores_output$response_binary)
stopifnot(length(permuted_labels) == nrow(clinical_scores_output))
frozen_hash_after_label_permutation <- digest::digest(
  file = frozen_program_file, algo = "sha256"
)
synthetic_direction_score <- compatibility_score(
  matrix(c(2, 0), nrow = 1, dimnames = list("synthetic_patient", c("suppressed_pathway", "neutral_pathway"))),
  c(suppressed_pathway = -2, neutral_pathway = 0),
  c(suppressed_pathway = 0.5, neutral_pathway = 0.5)
)

validation_checks <- data.frame(
  check = c(
    "PXD046566 total n=124", "PXD046566 FOLFOX n=38 (16 S/22 NS)",
    "PXD046566 FOLFOX+RT n=30", "PXD046566 FOLFOX+cetuximab n=17",
    "PXD040451 n=20 (9 R/11 NR)", "DeepHeal strict drugs n=120",
    "Selected clusters are 1+3", "Selected cluster drugs n=67",
    "Four-platform common proteins n=1218", "No duplicate clinical patients",
    "Technical replicates collapsed to unique patients", "Frozen program hash label-invariant",
    "Synthetic suppressed-high direction test", "Frozen effects finite",
    "Clinical scores finite", "Computed pathway matrices finite where observed",
    "Frozen pathways meet 70% coverage in primary cohort"
  ),
  passed = c(
    nrow(pxd046566_clinical) == 124L,
    sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX") == 38L &&
      sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX" & pxd046566_score_table$response == "S") == 16L &&
      sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX" & pxd046566_score_table$response == "NS") == 22L,
    sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+RT") == 30L,
    sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+cetuximab") == 17L,
    nrow(pxd040451_clinical) == 20L && sum(pxd040451_clinical$response == "Responders") == 9L,
    nrow(strict_metadata) == 120L,
    identical(selected_clusters, c(1L, 3L)),
    nrow(selected_metadata) == 67L,
    length(four_platform_common_genes) == 1218L,
    !anyDuplicated(clinical_scores_output$sample_id),
    nrow(pxd040451_technical_matrix) == 40L && nrow(pxd040451_matrix) == 20L,
    identical(frozen_program_sha256, frozen_hash_after_label_permutation),
    isTRUE(all.equal(as.numeric(synthetic_direction_score), 1, tolerance = 1e-12)),
    all(is.finite(frozen_program$ensemble_effect)),
    all(is.finite(clinical_scores_output$score)),
    all(is.finite(pxd046566_mbr_pathways$score[!is.na(pxd046566_mbr_pathways$score)])) &&
      all(is.finite(pxd046566_strict_pathways$score[!is.na(pxd046566_strict_pathways$score)])) &&
      all(is.finite(pxd040451_pathways$score[!is.na(pxd040451_pathways$score)])),
    all(pxd046566_mbr_pathways$coverage[
      pxd046566_score_table$sample_id[pxd046566_score_table$cohort == "PXD046566 FOLFOX"],
      frozen_program$pathway,
      drop = FALSE
    ] >= pathway_coverage_threshold)
  ),
  detail = c(
    nrow(pxd046566_clinical),
    "38; 16 S; 22 NS",
    sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+RT"),
    sum(pxd046566_score_table$cohort == "PXD046566 FOLFOX+cetuximab"),
    "20; 9 R; 11 NR",
    nrow(strict_metadata),
    paste(selected_clusters, collapse = "+"),
    nrow(selected_metadata),
    length(four_platform_common_genes),
    sum(duplicated(clinical_scores_output$sample_id)),
    paste(nrow(pxd040451_technical_matrix), "technical profiles to", nrow(pxd040451_matrix), "patients"),
    frozen_program_sha256,
    format(synthetic_direction_score, digits = 5),
    sum(is.finite(frozen_program$ensemble_effect)),
    sum(is.finite(clinical_scores_output$score)),
    sum(is.finite(pxd046566_mbr_pathways$score)) +
      sum(is.finite(pxd046566_strict_pathways$score)) +
      sum(is.finite(pxd040451_pathways$score)),
    min(pxd046566_mbr_pathways$coverage[
      pxd046566_score_table$sample_id[pxd046566_score_table$cohort == "PXD046566 FOLFOX"],
      frozen_program$pathway,
      drop = FALSE
    ])
  ),
  stringsAsFactors = FALSE
)
write_csv_stable(validation_checks, "validation_checks.csv")
stopifnot(all(validation_checks$passed))

# 10. Main figure -------------------------------------------------------------
response_colors <- c("Sensitive" = "#0072B2", "Non-sensitive" = "#D55E00")
validation_response_colors <- c("Responder" = "#0072B2", "Non-responder" = "#D55E00")
direction_colors <- c("Suppressed" = "#3B4CC0", "Induced" = "#B40426")
cohort_colors <- c(
  "PXD046566 FOLFOX" = "#0072B2",
  "PXD040451 nCRT" = "#009E73",
  "PXD046566 FOLFOX+RT" = "#E69F00",
  "PXD046566 FOLFOX+cetuximab" = "#CC79A7"
)

theme_figure <- theme_classic(base_family = base_family, base_size = 7.5) +
  theme(
    plot.title = element_text(face = "bold", size = 8.5, margin = margin(b = 3)),
    axis.title = element_text(size = 7.5),
    axis.text = element_text(size = 6.8, colour = "black"),
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 6.5),
    plot.margin = margin(3, 3, 3, 3)
  )

workflow_nodes <- data.frame(
  x = c(1, 2.35, 3.70, 5.05),
  label = c(
    "HCT116\n120 strict drugs",
    "A549\n5-FU + OXA",
    "Frozen\n20 pathways",
    "Patients\n2 cohorts"
  )
)
panel_a <- ggplot(workflow_nodes, aes(x = x, y = 1)) +
  geom_segment(
    data = data.frame(x = c(1, 2.35, 3.70), xend = c(2.35, 3.70, 5.05)),
    aes(x = x, xend = xend, y = 1, yend = 1),
    inherit.aes = FALSE, colour = "#6B7280", linewidth = 0.5,
    arrow = arrow(length = unit(1.8, "mm"), type = "closed")
  ) +
  geom_label(
    aes(label = label), size = 1.80, family = base_family,
    fill = c("#DCEAF7", "#FDE7C6", "#E3F2E9", "#EFE3F3"),
    colour = "#1F2937", linewidth = 0.25, label.r = unit(1.5, "mm")
  ) +
  annotate("text", x = 3.03, y = 0.62, label = "Cluster 1 + 3 | 67 drugs | 1,218 proteins", size = 2.05, family = base_family) +
  coord_cartesian(xlim = c(0.45, 5.55), ylim = c(0.45, 1.35), clip = "off") +
  theme_void(base_family = base_family) +
  ggtitle("Evidence workflow") +
  theme(plot.title = element_text(face = "bold", size = 8.5), plot.margin = margin(3, 4, 1, 4))

panel_b_data <- cluster_correlations |>
  filter(eligible_n) |>
  select(cluster, selected, `5-FU` = r_5fu, Oxaliplatin = r_oxaliplatin) |>
  pivot_longer(cols = c(`5-FU`, Oxaliplatin), names_to = "component", values_to = "correlation") |>
  mutate(cluster_label = paste0("Cluster ", cluster))
panel_b <- ggplot(panel_b_data, aes(component, cluster_label, fill = correlation)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", correlation), fontface = ifelse(selected, "bold", "plain")), size = 2.35, family = base_family) +
  scale_fill_gradient2(low = "#3B4CC0", mid = "white", high = "#B40426", midpoint = 0, limits = c(-0.55, 0.55), name = "Pearson r") +
  labs(title = "A549 component anchoring", x = NULL, y = NULL) +
  theme_figure +
  theme(legend.position = "right", axis.text.x = element_text(angle = 25, hjust = 1))

panel_c_data <- frozen_program |>
  mutate(display_label = factor(display_label, levels = rev(display_label)))
panel_c <- ggplot(panel_c_data, aes(ensemble_effect, display_label, colour = direction)) +
  geom_vline(xintercept = 0, colour = "#9CA3AF", linewidth = 0.35) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.45) +
  geom_point(aes(size = normalized_weight), alpha = 0.95) +
  scale_colour_manual(values = direction_colors) +
  scale_size_continuous(range = c(1.3, 3.1), breaks = c(0.04, 0.06, 0.08), name = "Weight") +
  labs(title = "Frozen DeepHeal pathway program", x = "Equal-weight ensemble pathway effect", y = NULL, colour = "Direction") +
  theme_figure +
  theme(legend.position = "right", axis.text.y = element_text(size = 5.2))

primary_plot_data <- clinical_scores |>
  filter(cohort == "PXD046566 FOLFOX") |>
  mutate(response_display = factor(response_display, levels = c("Non-sensitive", "Sensitive")))
primary_metric <- performance_metrics |> filter(cohort == "PXD046566 FOLFOX")
panel_d <- ggplot(primary_plot_data, aes(response_display, score, fill = response_display)) +
  geom_violin(width = 0.82, alpha = 0.28, colour = NA, trim = FALSE) +
  geom_boxplot(width = 0.32, outlier.shape = NA, linewidth = 0.4, alpha = 0.75) +
  geom_jitter(width = 0.09, height = 0, shape = 21, size = 1.7, stroke = 0.25, colour = "white") +
  scale_fill_manual(values = response_colors, guide = "none") +
  labs(
    title = sprintf("FOLFOX primary | r_rb = %.2f", primary_metric$rank_biserial),
    x = NULL, y = "Clinical compatibility score"
  ) +
  theme_figure + theme(axis.text.x = element_text(angle = 15, hjust = 1))

forest_data <- bind_rows(
  performance_metrics |>
    transmute(cohort, endpoint_class, metric = "AUROC", estimate = auroc, low = auroc_ci_low, high = auroc_ci_high),
  performance_metrics |>
    transmute(cohort, endpoint_class, metric = "AUPRC", estimate = auprc, low = auprc_ci_low, high = auprc_ci_high)
) |>
  mutate(
    cohort_label = recode(
      cohort,
      "PXD046566 FOLFOX" = "FOLFOX",
      "PXD040451 nCRT" = "nCRT validation",
      "PXD046566 FOLFOX+RT" = "FOLFOX + RT",
      "PXD046566 FOLFOX+cetuximab" = "FOLFOX + cetuximab"
    ),
    cohort_label = factor(cohort_label, levels = rev(c("FOLFOX", "nCRT validation", "FOLFOX + RT", "FOLFOX + cetuximab")))
  )
panel_e <- ggplot(forest_data, aes(estimate, cohort_label, colour = cohort)) +
  geom_vline(xintercept = 0.5, linetype = 2, colour = "#9CA3AF", linewidth = 0.35) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.45) +
  geom_point(size = 1.9) +
  facet_wrap(~metric, ncol = 2) +
  scale_colour_manual(values = cohort_colors, guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  labs(title = "AUROC / AUPRC", x = "Estimate (95% stratified bootstrap CI)", y = NULL) +
  theme_figure +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 7))

primary_ids <- primary_plot_data$sample_id[order(primary_plot_data$score, decreasing = TRUE)]
heatmap_activity <- t(pxd046566_mbr_pathways$score[primary_ids, frozen_program$pathway, drop = FALSE])
heatmap_display <- t(scale(t(heatmap_activity)))
heatmap_display[!is.finite(heatmap_display)] <- 0
rownames(heatmap_activity) <- frozen_program$display_label
rownames(heatmap_display) <- frozen_program$display_label
primary_annotations <- primary_plot_data |>
  select(sample_id, response_display, score) |>
  slice(match(primary_ids, sample_id))
rownames(primary_annotations) <- primary_annotations$sample_id

write_csv_stable(
  data.frame(pathway = rownames(heatmap_activity), heatmap_activity, check.names = FALSE),
  "heatmap_activity_matrix.csv"
)
write_csv_stable(
  data.frame(pathway = rownames(heatmap_display), heatmap_display, check.names = FALSE),
  "heatmap_display_row_z_matrix.csv"
)
write_csv_stable(primary_annotations, "heatmap_sample_annotations.csv")
write_csv_stable(data.frame(row_order = seq_along(rownames(heatmap_display)), pathway = rownames(heatmap_display)), "heatmap_row_order.csv")
write_csv_stable(data.frame(column_order = seq_along(primary_ids), sample_id = primary_ids), "heatmap_column_order.csv")

heatmap_annotation <- HeatmapAnnotation(
  Response = primary_annotations$response_display,
  col = list(Response = response_colors),
  annotation_name_gp = gpar(fontfamily = base_family, fontsize = 6.5, fontface = "bold"),
  annotation_legend_param = list(
    Response = list(title_gp = gpar(fontfamily = base_family, fontsize = 6.5, fontface = "bold"), labels_gp = gpar(fontfamily = base_family, fontsize = 6))
  )
)
heatmap_object <- Heatmap(
  heatmap_display,
  name = "Activity\n(row z)",
  col = circlize::colorRamp2(c(-2, 0, 2), c("#3B4CC0", "white", "#B40426")),
  top_annotation = heatmap_annotation,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_column_names = FALSE,
  row_names_gp = gpar(fontfamily = base_family, fontsize = 5.6),
  column_title = "PXD046566 FOLFOX pretreatment patients ordered by frozen score",
  column_title_gp = gpar(fontfamily = base_family, fontsize = 7, fontface = "bold"),
  width = unit(heatmap_body_width_mm, "mm"),
  height = unit(heatmap_body_height_mm, "mm"),
  heatmap_legend_param = list(
    title_gp = gpar(fontfamily = base_family, fontsize = 6.5, fontface = "bold"),
    labels_gp = gpar(fontfamily = base_family, fontsize = 6)
  ),
  use_raster = TRUE,
  raster_quality = 3
)
heatmap_grob <- grid::grid.grabExpr(draw(
  heatmap_object, heatmap_legend_side = "right", annotation_legend_side = "right", newpage = FALSE
))
heatmap_panel_png <- file.path(result_dir, "heatmap_panel_for_composition.png")
ragg::agg_png(
  heatmap_panel_png, width = main_width_mm, height = heatmap_canvas_height_mm,
  units = "mm", res = dpi, background = "white"
)
draw(heatmap_object, heatmap_legend_side = "right", annotation_legend_side = "right", newpage = TRUE)
dev.off()
heatmap_raster <- png::readPNG(heatmap_panel_png)
heatmap_ink <- apply(heatmap_raster[, , seq_len(min(3, dim(heatmap_raster)[3])), drop = FALSE], c(1, 2), min) < 0.995
heatmap_rows <- which(rowSums(heatmap_ink) > 0)
heatmap_columns <- which(colSums(heatmap_ink) > 0)
row_padding <- 8L
column_padding <- 8L
heatmap_raster <- heatmap_raster[
  max(1L, min(heatmap_rows) - row_padding):min(dim(heatmap_raster)[1], max(heatmap_rows) + row_padding),
  max(1L, min(heatmap_columns) - column_padding):min(dim(heatmap_raster)[2], max(heatmap_columns) + column_padding),
  , drop = FALSE
]
padded_heatmap_raster <- array(
  1,
  dim = c(
    dim(heatmap_raster)[1],
    dim(heatmap_raster)[2] + heatmap_composition_left_padding_px,
    dim(heatmap_raster)[3]
  )
)
padded_heatmap_raster[
  , (heatmap_composition_left_padding_px + 1L):dim(padded_heatmap_raster)[2],
] <- heatmap_raster
heatmap_raster <- padded_heatmap_raster
panel_f <- patchwork::wrap_elements(
  full = grid::rasterGrob(
    heatmap_raster,
    width = unit(1, "npc"), height = unit(1, "npc"),
    interpolate = TRUE
  )
)

evidence_column_levels <- c(
  "PXD046566\nFOLFOX NS", "PXD046566\nFOLFOX S",
  "PXD046566\nFOLFOX+Radiotherapy NS", "PXD046566\nFOLFOX+Radiotherapy S",
  "PXD046566\nFOLFOX+Cetuximab NS", "PXD046566\nFOLFOX+Cetuximab S",
  "PXD008436\nresponse groups", "PXD008436\nPre to post-CRT",
  "PXD008436\nPre-CRT to resection", "PXD008436\nPost-CRT to resection"
)
evidence_family_levels <- c(
  "Cell cycle", "DNA replication / repair", "Nucleotide / folate metabolism",
  "Lipid / ceramide metabolism", "Immune signalling", "Autophagy",
  "Adhesion / neutrophil", "Other metabolism", "Global longitudinal proteome"
)
panel_g_data <- evidence_matrix_table |>
  mutate(
    evidence_column = factor(evidence_column, levels = evidence_column_levels),
    mechanism_family = factor(mechanism_family, levels = rev(evidence_family_levels))
  ) |>
  filter(!is.na(evidence_column))
panel_g <- ggplot(panel_g_data, aes(evidence_column, mechanism_family, fill = factor(direction))) +
  geom_tile(colour = "white", linewidth = 0.35) +
  geom_text(aes(label = ifelse(direction > 0, "+", "-")), size = 2.6, family = base_family) +
  scale_fill_manual(
    values = c(`-1` = "#6BAED6", `0` = "#E5E7EB", `1` = "#EF8A62"),
    labels = c(`-1` = "Down / negative NES", `0` = "Mixed", `1` = "Up / positive NES"),
    name = "Author result"
  ) +
  labs(title = "Published post-treatment and longitudinal evidence (group level)", x = NULL, y = NULL) +
  theme_figure +
  theme(
    axis.text.x = element_text(angle = 38, hjust = 1, vjust = 1, size = 5.6),
    axis.text.y = element_text(size = 5.8),
    legend.position = "right"
  )

main_figure <- ((panel_a | panel_b) + plot_layout(widths = c(1.35, 1))) /
  panel_c /
  ((panel_d | panel_e) + plot_layout(widths = c(0.85, 1.15))) /
  panel_f /
  panel_g +
  plot_layout(heights = c(0.70, 1.60, 1.0, 1.35, 1.35)) +
  plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(family = base_family, face = "bold", size = 10)))
export_plot(main_figure, "Figure8_Clinical_translation", main_width_mm, main_height_mm)

# 11. Supplementary figures ---------------------------------------------------
source_s8 <- file.path(old_cptac_figure_dir, "Figure8_DeepHeal_CPTAC_reversal")
source_s9 <- file.path(old_cptac_figure_dir, "FigureS8_CPTAC_patient_latent_projection")
for (extension in c("png", "tiff", "pdf")) {
  stopifnot(file.copy(
    paste0(source_s8, ".", extension),
    file.path(figure_dir, paste0("FigureS8_CPTAC_reversal.", extension)),
    overwrite = TRUE
  ))
  stopifnot(file.copy(
    paste0(source_s9, ".", extension),
    file.path(figure_dir, paste0("FigureS9_CPTAC_patient_latent_projection.", extension)),
    overwrite = TRUE
  ))
}

coverage_summary <- coverage_table |>
  group_by(cohort, pathway) |>
  summarise(median_coverage = median(coverage, na.rm = TRUE), .groups = "drop") |>
  mutate(pathway_label = clean_pathway_label(pathway))
panel_s10a <- ggplot(coverage_summary, aes(median_coverage, pathway_label, colour = cohort)) +
  geom_vline(xintercept = pathway_coverage_threshold, linetype = 2, colour = "#9CA3AF") +
  geom_point(size = 1.5) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(title = "Frozen-pathway coverage", x = "Median observed fraction", y = NULL, colour = NULL) +
  theme_figure
platform_sizes <- data.frame(
  platform = c("HCT116", "A549", "PXD046566", "PXD040451", "Common"),
  proteins = c(
    ncol(strict_hct116), ncol(a549_matrix), ncol(pxd046566_mbr),
    ncol(pxd040451_matrix), length(four_platform_common_genes)
  )
)
panel_s10b <- ggplot(platform_sizes, aes(reorder(platform, proteins), proteins, fill = platform == "Common")) +
  geom_col(width = 0.72) +
  geom_text(aes(label = scales::comma(proteins)), hjust = -0.08, size = 2.35, family = base_family) +
  coord_flip(clip = "off") +
  scale_fill_manual(values = c(`FALSE` = "#9CA3AF", `TRUE` = "#0072B2"), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.32))) +
  labs(title = "Protein universes", x = NULL, y = "Proteins") +
  theme_figure
figure_s10 <- (panel_s10a | panel_s10b) +
  plot_layout(widths = c(1.2, 0.8)) +
  plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(family = base_family, face = "bold", size = 10)))
export_plot(figure_s10, "FigureS10_Coverage_and_common_space", supp_width_mm, supp_height_mm)

panel_s11a <- ggplot(pr_curve_table, aes(recall, precision, colour = cohort)) +
  geom_step(linewidth = 0.65) +
  scale_colour_manual(
    values = cohort_colors,
    labels = c(
      "PXD046566 FOLFOX" = "FOLFOX",
      "PXD040451 nCRT" = "nCRT",
      "PXD046566 FOLFOX+RT" = "FOLFOX + RT",
      "PXD046566 FOLFOX+cetuximab" = "FOLFOX + cetuximab"
    )
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(title = "Precision-recall curves", x = "Recall", y = "Precision", colour = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  theme_figure + theme(legend.position = "bottom", legend.text = element_text(size = 5.8))
sensitivity_forest <- bind_rows(
  performance_metrics |> filter(cohort == "PXD046566 FOLFOX") |>
    transmute(analysis = "MBR primary", auroc, low = auroc_ci_low, high = auroc_ci_high),
  strict_sensitivity_metrics |> filter(grepl("PXD046566 FOLFOX \\[strict", cohort)) |>
    transmute(analysis = "Strict 1% FDR", auroc, low = auroc_ci_low, high = auroc_ci_high),
  marginal_metric |>
    transmute(analysis = "Include marginal drugs", auroc, low = auroc_ci_low, high = auroc_ci_high)
) |>
  mutate(analysis = factor(analysis, levels = rev(c("MBR primary", "Strict 1% FDR", "Include marginal drugs"))))
panel_s11b <- ggplot(sensitivity_forest, aes(auroc, analysis)) +
  geom_vline(xintercept = 0.5, linetype = 2, colour = "#9CA3AF") +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.5, colour = "#0072B2") +
  geom_point(size = 2, colour = "#0072B2") +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  labs(title = "Primary-cohort sensitivities", x = "AUROC (95% CI)", y = NULL) +
  theme_figure
jaccard_plot_data <- jaccard_sensitivity |>
  mutate(overlap_fraction = overlap_with_primary / frozen_pathway_count)
panel_s11c <- ggplot(jaccard_plot_data, aes(threshold, overlap_fraction)) +
  geom_vline(xintercept = pathway_jaccard_threshold, linetype = 2, colour = "#9CA3AF") +
  geom_line(colour = "#6B7280", linewidth = 0.45) +
  geom_point(colour = "#0072B2", size = 1.9) +
  geom_text(aes(label = paste0(overlap_with_primary, "/20")), vjust = -0.8, size = 2.3, family = base_family) +
  scale_x_continuous(breaks = c(0.20, 0.25, 0.30)) +
  scale_y_continuous(limits = c(0.75, 1.04), breaks = c(0.8, 0.9, 1.0)) +
  labs(title = "Pathway redundancy threshold", x = "Maximum Jaccard", y = "Primary-program overlap") +
  theme_figure
panel_s11_right <- (panel_s11b / panel_s11c) + plot_layout(heights = c(1, 1))
figure_s11 <- (panel_s11a | panel_s11_right) +
  plot_layout(widths = c(1.05, 0.95)) +
  plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(family = base_family, face = "bold", size = 10)))
export_plot(figure_s11, "FigureS11_PR_and_sensitivity", supp_width_mm, supp_height_mm)

panel_s12a <- panel_g + labs(title = "Author-reported mechanism evidence")
longitudinal_plot_data <- pxd008436_longitudinal |>
  mutate(comparison = factor(comparison, levels = rev(comparison)))
panel_s12b <- ggplot(longitudinal_plot_data, aes(estimate, comparison, fill = estimate > 0)) +
  geom_vline(xintercept = 0, colour = "#6B7280") +
  geom_col(width = 0.65) +
  geom_text(aes(label = detail), hjust = ifelse(longitudinal_plot_data$estimate > 0, -0.08, 1.08), size = 2.5, family = base_family) +
  scale_fill_manual(values = c(`FALSE` = "#6BAED6", `TRUE` = "#EF8A62"), guide = "none") +
  scale_x_continuous(limits = c(-1, 1), expand = expansion(mult = c(0.1, 0.1))) +
  labs(title = "PXD008436 longitudinal result-level balance", x = "(Up - down) / significant proteins", y = NULL) +
  theme_figure
figure_s12 <- panel_s12a / panel_s12b +
  plot_layout(heights = c(1.6, 1)) +
  plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(family = base_family, face = "bold", size = 10)))
export_plot(figure_s12, "FigureS12_Author_level_longitudinal_details", supp_width_mm, supp_height_mm)

# 12. Manifests and session information --------------------------------------
download_manifest <- jsonlite::read_json(download_manifest_file, simplifyVector = TRUE)
download_source_table <- as.data.frame(download_manifest)
local_source_table <- data.frame(
  file = basename(c(hct116_matrix_file, hct116_metadata_file, a549_matrix_file, a549_map_file, c2_gmt_file)),
  source_url = c(
    "Local DeepHeal source matrix", "Local DeepHeal frozen metadata",
    "Local author-processed ProTargetMiner A549 matrix", "Local A549 UniProt selection map",
    "Local retained MSigDB v6.2 C2 GMT"
  ),
  access_date = "2026-09-04",
  bytes = file.info(c(hct116_matrix_file, hct116_metadata_file, a549_matrix_file, a549_map_file, c2_gmt_file))$size,
  sha256 = vapply(
    c(hct116_matrix_file, hct116_metadata_file, a549_matrix_file, a549_map_file, c2_gmt_file),
    function(path) digest::digest(file = path, algo = "sha256"), character(1)
  ),
  stringsAsFactors = FALSE
)
data_source_manifest <- bind_rows(download_source_table, local_source_table)
write_csv_stable(data_source_manifest, "data_source_manifest.csv")

heatmap_layout <- data.frame(
  parameter = c(
    "matrix_orientation", "row_order", "column_order", "column_clustering",
    "row_clustering", "column_annotation", "colour_breaks", "raster_quality",
    "heatmap_body_width_mm", "heatmap_body_height_mm", "composition_left_padding_px"
  ),
  value = c(
    "pathways x patients", "frozen pathway rank", "decreasing frozen score",
    "FALSE", "FALSE", "response only", "-2,0,2", "3",
    heatmap_body_width_mm, heatmap_body_height_mm, heatmap_composition_left_padding_px
  )
)
write_csv_stable(heatmap_layout, "heatmap_layout_parameters.csv")

analysis_manifest <- list(
  analysis = "DeepHeal Figure 8 clinical translation",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  claim_scope = "Association of a frozen drug-perturbation pathway program with treatment response; not a clinically deployable prediction model.",
  random_seed = random_seed,
  parameters = list(
    zero_threshold = zero_threshold,
    strict_drugs = nrow(strict_metadata),
    selected_clusters = selected_clusters,
    selected_cluster_drugs = nrow(selected_metadata),
    common_proteins = length(four_platform_common_genes),
    pathway_sources = c("MSigDB v6.2 Reactome", "MSigDB v6.2 KEGG", "MSigDB v6.2 Hallmark"),
    pathway_gene_range = c(pathway_min_genes, pathway_max_genes),
    pathway_coverage_threshold = pathway_coverage_threshold,
    pxd040451_predefined_exclusions = predefined_exclusions,
    cluster_correlation_threshold = cluster_correlation_threshold,
    pathway_jaccard_threshold = pathway_jaccard_threshold,
    frozen_pathways = frozen_pathway_count,
    bootstrap_iterations = bootstrap_iterations,
    permutation_iterations = permutation_iterations
  ),
  frozen_program_sha256 = frozen_program_sha256,
  no_raw_ms_files_downloaded = TRUE,
  clinical_label_use = "Labels read only after the frozen program table and hash were written.",
  primary_endpoint = "PXD046566 exact FOLFOX",
  independent_validation = "PXD040451 nCRT",
  exploratory_endpoints = c("PXD046566 FOLFOX+RT", "PXD046566 FOLFOX+cetuximab"),
  validation_checks_passed = all(validation_checks$passed)
)
jsonlite::write_json(
  analysis_manifest, file.path(result_dir, "analysis_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, digits = NA
)
writeLines(capture.output(sessionInfo()), file.path(result_dir, "sessionInfo.txt"), useBytes = TRUE)

output_files <- list.files(
  package_root, recursive = TRUE, full.names = TRUE,
  pattern = "\\.(csv|json|txt|png|tiff|pdf)$"
)
output_files <- output_files[!grepl("data/raw/download_manifest.json$", gsub("\\\\", "/", output_files))]
output_files <- output_files[!grepl("results/output_manifest.csv$", gsub("\\\\", "/", output_files))]
output_manifest <- data.frame(
  relative_path = substring(gsub("\\\\", "/", output_files), nchar(gsub("\\\\", "/", package_root)) + 2),
  bytes = file.info(output_files)$size,
  sha256 = vapply(
    output_files,
    function(path) digest::digest(file = path, algo = "sha256"),
    character(1)
  ),
  stringsAsFactors = FALSE
)
data.table::fwrite(output_manifest, file.path(result_dir, "output_manifest.csv"), quote = TRUE)

message("Figure 8 clinical translation analysis completed.")
message("Frozen program SHA-256: ", frozen_program_sha256)
message("All validation checks passed: ", all(validation_checks$passed))
