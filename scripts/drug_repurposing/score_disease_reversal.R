# 1. Package loading ---------------------------------------------------------
library(data.table)
library(jsonlite)

# 2. Paths and parameters ----------------------------------------------------
repo_root <- normalizePath(Sys.getenv("DEEPHEAL_ROOT", unset = "."), mustWork = TRUE)
package_root <- file.path(repo_root, "results/drug_repurposing")
original_root <- file.path(repo_root, "data/repurposing")
fig8_root <- file.path(repo_root, "data/cptac")
input_dir <- file.path(package_root, "data/processed")
table_dir <- file.path(package_root, "tables")
dir.create(table_dir, recursive=TRUE, showWarnings=FALSE)
top_fraction <- 0.10
zero_tolerance <- 1e-12
ranking_digits <- 12L
setDTthreads(8)

# 3. Import -----------------------------------------------------------------
queries <- fread(file.path(input_dir,"query_metadata.csv"))
drugs <- fread(file.path(input_dir,"drug_metadata.csv"))
candidate_annotations <- fread(file.path(original_root,"tables/candidate_evidence_matrix.csv"))
drug_ids <- drugs$drug_norm
query_ids <- queries$query_key

# 4. Validation and score functions ------------------------------------------
cosine_matrix <- function(drug_matrix, disease_matrix, direction=-1) {
  denominator <- outer(sqrt(rowSums(drug_matrix^2)), sqrt(rowSums(disease_matrix^2)))
  result <- direction * tcrossprod(drug_matrix, disease_matrix) / denominator
  result[denominator <= zero_tolerance | !is.finite(denominator)] <- NA_real_
  result
}
long_scores <- function(values, method, version, keys=query_ids) {
  stopifnot(nrow(values)==length(drug_ids),ncol(values)==length(keys))
  data.table(drug_norm=rep(drug_ids,length(keys)),query_key=rep(keys,each=length(drug_ids)),
             method=method,model_version=version,score=as.vector(values))
}
add_rank <- function(dt, groups, value="score", prefix="") {
  result <- copy(dt)
  result[, c(paste0(prefix,"rank"),paste0(prefix,"percentile"),paste0(prefix,"top10"),paste0(prefix,"n_valid")) := {
    x <- round(get(value),ranking_digits); valid <- is.finite(x); n <- sum(valid)
    r <- rep(NA_integer_,.N); p <- rep(NA_real_,.N); top <- rep(NA,.N)
    if(n>0) {
      ord <- order(-x[valid],drug_norm[valid],method="radix")
      rv <- integer(n);rv[ord] <- seq_len(n);r[valid] <- rv
      p[valid] <- rank(x[valid],ties.method="average")/n
      top[valid] <- rv <= ceiling(top_fraction*n)
    }
    list(r,p,top,rep(n,.N))
  }, by=groups]
  result
}

# 5. Latent and protein reversal --------------------------------------------
score_parts <- list()
latent_distribution <- list()
for(branch in c("average","supcon")) {
  message("Scoring ",branch)
  embed <- fread(file.path(input_dir,paste0(branch,"_embeddings.csv.gz")))
  columns <- grep("^latent_",names(embed),value=TRUE)
  zero <- as.numeric(embed[kind=="zero",..columns])
  extract <- function(type, ids) {
    block <- embed[kind==type]
    stopifnot(!anyDuplicated(block$item_id),all(ids %in% block$item_id))
    sweep(as.matrix(block[match(ids,item_id),..columns]),2,zero,"-")
  }
  drug_z <- extract("drug",drug_ids)
  disease_z <- extract("disease",query_ids)
  inverse_z <- extract("inverse",query_ids)
  summary <- fromJSON(file.path(package_root,"results",paste0(branch,"_projection_summary.json")))
  stopifnot(summary$meets_empirical_fidelity,summary$all_projection_checks_pass)
  version <- summary$model_version
  label <- if(branch=="average") "Average" else "SupCon"
  score_parts[[paste0(label,"_main")]] <- long_scores(cosine_matrix(drug_z,disease_z),label,version)
  score_parts[[paste0(label,"_inverse")]] <- long_scores(cosine_matrix(drug_z,inverse_z,1),paste0(label,"_inverse"),version)
  for(key in queries[source!="PATIENT",query_key]) {
    block <- embed[kind=="masked_drug" & context==key]
    stopifnot(nrow(block)==875L)
    masked_z <- sweep(as.matrix(block[match(drug_ids,item_id),..columns]),2,zero,"-")
    zq <- disease_z[match(key,query_ids),,drop=FALSE]
    score_parts[[paste(label,key)]] <- long_scores(cosine_matrix(masked_z,zq),paste0(label,"_masked"),version,key)
  }
  latent_distribution[[branch]] <- rbindlist(list(
    data.table(method=label,kind="Drug",item_id=drug_ids,centered_norm=sqrt(rowSums(drug_z^2))),
    data.table(method=label,kind=queries$source,item_id=query_ids,centered_norm=sqrt(rowSums(disease_z^2)))))
  rm(embed);gc(FALSE)
}
protein <- fread(file.path(input_dir,"drug_protein_matrix.csv.gz"))
disease <- fread(file.path(input_dir,"disease_protein_matrix.csv.gz"))
stopifnot(identical(names(protein)[-1],names(disease)[-1]))
protein_values <- as.matrix(protein[match(drug_ids,drug_norm),-1])
disease_values <- as.matrix(disease[match(query_ids,query_key),-1])
score_parts[["protein_control"]] <- long_scores(cosine_matrix(protein_values,disease_values),"Protein_unweighted","Aligned full-input unweighted protein cosine")
rm(protein,disease,protein_values,disease_values);gc(FALSE)

# Frozen reference scores are preserved verbatim before common re-ranking.
old_cptac <- fread(file.path(fig8_root,"tables/pancancer_drug_reversal_scores_long.csv"))
old_tpcpa <- fread(file.path(original_root,"tables/tpcpa_drug_context_scores_long.csv"))
old_patient <- fread(file.path(original_root,"tables/cptac_coad_patient_reversal_long.csv"))
score_parts[["original"]] <- rbindlist(list(
  old_cptac[,.(drug_norm,query_key=paste0("CPTAC:",toupper(cohort)),score=deepheal_reversal_score)],
  old_tpcpa[,.(drug_norm,query_key=paste0("TPCPA:",tpcpa_cancer),score=tpcpa_context_reversal_score)],
  old_patient[,.(drug_norm,query_key=paste0("PATIENT:",patient_id),score=patient_reversal_score)]
))[,`:=`(method="Original",model_version="Frozen Fig9 protein/pathway mean percentile")]
scores <- merge(rbindlist(score_parts,use.names=TRUE),queries,by="query_key",all.x=TRUE)
stopifnot(!anyNA(scores$source),!anyDuplicated(scores[,.(query_key,drug_norm,method)]))
scores[,quality_flag:=fifelse(is.finite(score),"finite","undefined_zero_norm")]
scores[,ranking_score:=round(score,ranking_digits)]
scores <- add_rank(scores,c("method","query_key"))
fwrite(scores,file.path(table_dir,"all_reversal_scores_long.csv.gz"))
fwrite(scores[,.(n_rows=.N,n_valid=sum(is.finite(score)),n_top10=sum(top10 %in% TRUE)),by=.(method,source,context)],file.path(table_dir,"score_completeness.csv"))
fwrite(rbindlist(latent_distribution),file.path(table_dir,"latent_distribution.csv"))

# Common population agreement and top-list overlap (no outcome selection).
main_methods <- c("Original","Average","SupCon","Protein_unweighted")
agreement <- list()
for(source_name in c("CPTAC","TPCPA","PATIENT")) {
  for(context_name in unique(scores[source==source_name,context])) {
    block <- scores[source==source_name & context==context_name & method %in% main_methods]
    wide <- dcast(block,drug_norm~method,value.var="ranking_score")
    complete <- wide[complete.cases(wide)]
    for(pair in combn(main_methods,2,simplify=FALSE)) {
      x <- complete[[pair[1]]];y <- complete[[pair[2]]];n<-length(x);k<-ceiling(.1*n)
      sx<-complete$drug_norm[order(-x,complete$drug_norm)][seq_len(k)]
      sy<-complete$drug_norm[order(-y,complete$drug_norm)][seq_len(k)]
      agreement[[length(agreement)+1]]<-data.table(source=source_name,context=context_name,method1=pair[1],method2=pair[2],n=n,
        rho=cor(x,y,method="spearman"),top10_overlap=length(intersect(sx,sy)),top10_jaccard=length(intersect(sx,sy))/length(union(sx,sy)))
    }
  }
}
fwrite(rbindlist(agreement),file.path(table_dir,"method_agreement.csv"))

# Lineage integration keeps the original LUAD/LSCC pooling policy.
cohort_map <- data.table(context=c("CCRCC","COAD","HNSCC","LUAD","LSCC","PDAC","OV_PAIRED","UCEC_UNPAIRED"),
 lineage=c("Kidney","Colorectal","Head and neck","Lung NSCLC","Lung NSCLC","Pancreas","Ovarian","Endometrial"))
lineages <- fread(file.path(original_root,"tables/cancer_lineage_crosswalk.csv"))
cptac <- merge(scores[source=="CPTAC"],cohort_map,by="context")
cptac <- cptac[,.(cptac_score=mean(percentile),cptac_raw_score=mean(score),model_version=unique(model_version)),by=.(method,drug_norm,lineage)]
cptac[,cptac_score:=round(cptac_score,ranking_digits)]
cptac <- add_rank(cptac,c("method","lineage"),"cptac_score","cptac_")
tpcpa <- merge(scores[source=="TPCPA"],lineages[,.(context=tpcpa_cancer,lineage)],by="context")
tpcpa <- tpcpa[,.(method,drug_norm,lineage,tpcpa_score=score,tpcpa_percentile=percentile,tpcpa_top10=top10)]
evidence <- merge(cptac,tpcpa,by=c("method","drug_norm","lineage"),all.x=TRUE)
prism <- fread(file.path(original_root,"tables/prism_lineage_selectivity.csv"))
stopifnot(all(prism$n_prism_cell_lines>=5),!anyDuplicated(prism[,.(drug_norm,lineage)]))
evidence <- merge(evidence,prism,by=c("drug_norm","lineage"),all.x=TRUE)
evidence <- merge(evidence,lineages[,.(lineage,analysis_tier)],by="lineage",all.x=TRUE)
evidence[,evidence_tier:=fcase(cptac_top10 %in% TRUE & tpcpa_top10 %in% TRUE & prism_top25_sensitive %in% TRUE,"Tier 1",
 cptac_top10 %in% TRUE & (tpcpa_top10 %in% TRUE | prism_top25_sensitive %in% TRUE),"Tier 2",default="Tier 3")]
evidence[,proteome_group:=fcase(is.na(tpcpa_top10),"Incomplete",cptac_top10 & tpcpa_top10,"Both",xor(cptac_top10,tpcpa_top10),"One",default="Neither")]
fwrite(evidence,file.path(table_dir,"drug_lineage_evidence.csv"))
fwrite(evidence[is.finite(prism_lineage_selectivity),.(n=.N,median_selectivity=median(prism_lineage_selectivity)),by=.(method,analysis_tier,proteome_group)],file.path(table_dir,"dual_proteome_prism_summary.csv"))

# Frozen candidate locks and method-specific locks are separate named analyses.
annotations <- candidate_annotations[,.(drug_norm,candidate_name,locked_cancer,cck8_hit,fit_status,ic50_uM,r_squared,
 clinical_stage_group,clinical_phase,moa,target,indication,mapping_status)]
locks <- annotations[,.(drug_norm,lineage=locked_cancer)]
locked <- merge(evidence[method %in% main_methods],locks,by=c("drug_norm","lineage"))
locked <- merge(locked,annotations,by="drug_norm")
locked[,lock_policy:="Original Fig9 lock"]
fwrite(locked,file.path(table_dir,"candidate_evidence_frozen_lock.csv"))
independent <- copy(evidence[drug_norm %in% annotations$drug_norm & method %in% main_methods])
setorder(independent,method,drug_norm,-cptac_score,lineage)
independent <- independent[, .SD[1],by=.(method,drug_norm)]
independent <- merge(independent,annotations,by="drug_norm")
independent[,lock_policy:="Method-specific CPTAC-only lock"]
fwrite(independent,file.path(table_dir,"candidate_evidence_method_lock.csv"))

# Patient heterogeneity ranks all 875 drugs before selecting the 19 candidates.
patient_scores <- scores[source=="PATIENT" & method %in% main_methods]
candidate_patient <- merge(patient_scores[drug_norm %in% annotations$drug_norm],annotations,by="drug_norm")
fwrite(candidate_patient,file.path(table_dir,"candidate_patient_scores.csv"))
patient_summary <- candidate_patient[,.(n_patients=.N,median_percentile=median(percentile),q1=quantile(percentile,.25),q3=quantile(percentile,.75),
 median_rank=median(rank),top10_patients=sum(top10),top10_fraction=mean(top10)),by=.(method,drug_norm,candidate_name,cck8_hit)]
fwrite(patient_summary,file.path(table_dir,"candidate_patient_summary.csv"))
patient_top <- copy(candidate_patient);setorder(patient_top,method,query_key,rank,drug_norm)
patient_top <- patient_top[,.SD[1],by=.(method,query_key)]
fwrite(patient_top[,.(method,query_key,drug_norm,candidate_name,rank,percentile)],file.path(table_dir,"patient_top_candidate.csv"))
fwrite(patient_top[,.(n_patients=.N),by=.(method,drug_norm,candidate_name)],file.path(table_dir,"patient_top_candidate_counts.csv"))
fwrite(patient_scores[rank==1,.(method,query_key,drug_norm,rank,percentile)],file.path(table_dir,"patient_top_all875.csv"))
hit_comparison <- patient_summary[, {
  x<-median_percentile[cck8_hit %in% TRUE];y<-median_percentile[!cck8_hit %in% TRUE]
  list(n_hit=length(x),n_nonhit=length(y),hit_median=median(x),nonhit_median=median(y),p_value=wilcox.test(x,y,exact=FALSE)$p.value)
},by=method]
hit_comparison[,q_value:=p.adjust(p_value,"BH")]
fwrite(hit_comparison,file.path(table_dir,"patient_candidate_cck8_comparison.csv"))

# 6–7. Reproducibility handoff to standalone plotting ------------------------
write_json(list(main_methods=main_methods,top10_rule="rank <= ceiling(0.1*n_valid); scores rounded to 12 decimal places for ranking only; drug ID resolves ties",zero_norm_policy="NA",
 score_definition="-cos(E(drug)-E(0),E(disease)-E(0))",n_drugs=875,n_paired_patients=96,
 cohort_pooling="mean within-cohort score percentile for LUAD and LSCC",
 evidence_note="same categorical logic as Fig9, using exact top-list membership after method-wise re-ranking"),
 file.path(package_root,"results/score_manifest.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(package_root,"results/sessionInfo_scoring.txt"))
message("Scoring and integration completed")
