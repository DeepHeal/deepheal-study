# 1. Package loading ---------------------------------------------------------
library(data.table)
library(jsonlite)

# 2. Paths and fixed parameters ---------------------------------------------
repo_root <- normalizePath(Sys.getenv("DEEPHEAL_ROOT", unset = "."), mustWork = TRUE)
package_root <- file.path(repo_root,"results/drug_repurposing")
original_root <- file.path(repo_root,"data/repurposing")
table_dir <- file.path(package_root,"tables")
result_dir <- file.path(package_root,"results")
dir.create(table_dir,recursive=TRUE,showWarnings=FALSE)
seed <- 42L
permutation_iterations <- 10000L
bootstrap_iterations <- 2000L
primary_lineages <- c("Kidney","Colorectal","Head and neck","Lung NSCLC","Pancreas")
lineage_levels <- c(primary_lineages,"Ovarian","Endometrial")
main_methods <- c("Original","Average","SupCon","Protein_unweighted")
all_methods <- c(main_methods,"Average_inverse","SupCon_inverse","Average_masked","SupCon_masked")
positive_fractions <- c(.10,.20,.25,.30)
setDTthreads(8)

# 3. Import -----------------------------------------------------------------
evidence <- fread(file.path(table_dir,"drug_lineage_evidence.csv"))

# 4. Statistical helpers and matched populations -----------------------------
rank_matrix <- function(mat) matrix(rank(as.vector(mat),ties.method="average"),nrow(mat),ncol(mat))
center_matrix <- function(mat) {
  out <- sweep(mat,1,rowMeans(mat),"-")
  sweep(out,2,colMeans(out),"-")
}
cor_fast <- function(x,y) {
  x<-as.vector(x);y<-as.vector(y);x<-x-mean(x);y<-y-mean(y)
  den<-sqrt(sum(x*x)*sum(y*y));if(den<1e-15) return(NA_real_)
  sum(x*y)/den
}
partial_rho <- function(x,y) cor_fast(center_matrix(rank_matrix(x)),center_matrix(rank_matrix(y)))
spearman <- function(x,y) cor_fast(rank(x,ties.method="average"),rank(y,ties.method="average"))
ci <- function(x) unname(quantile(x,c(.025,.975),na.rm=TRUE))
classification_metrics <- function(labels,scores) {
  ok<-is.finite(scores)&!is.na(labels);y<-as.integer(labels[ok]);x<-scores[ok]
  np<-sum(y);nn<-length(y)-np
  if(np==0 || nn==0) return(c(AUROC=NA_real_,AUPRC=NA_real_))
  auc<-(sum(rank(x,ties.method="average")[y==1])-np*(np+1)/2)/(np*nn)
  # Average precision groups equal scores into one threshold (order invariant).
  d<-data.table(x=x,y=y)[,.(positive=sum(y),n=.N),by=x];setorder(d,-x)
  precision<-cumsum(d$positive)/cumsum(d$n)
  ap<-sum(precision*d$positive)/np
  c(AUROC=auc,AUPRC=ap)
}
label_matrix <- function(y,fraction) {
  # Preserve original PRISM threshold definition, including average tied ranks.
  apply(y,2,function(v) rank(v,ties.method="average")/length(v)>=1-fraction)
}

# Keep a matched table for every resource and lineage, including sensitivity methods.
matched_parts<-list()
for(resource in c("CPTAC","TPCPA")) {
  value<-if(resource=="CPTAC") "cptac_score" else "tpcpa_percentile"
  block<-evidence[,.(drug_norm,lineage,method,score=get(value),prism_lineage_selectivity)]
  block<-block[method %in% all_methods]
  wide<-dcast(block,drug_norm+lineage+prism_lineage_selectivity~method,value.var="score")
  wide<-wide[complete.cases(wide)]
  wide[,resource:=resource]
  matched_parts[[resource]]<-wide
}
matched<-rbindlist(matched_parts)
fwrite(matched,file.path(table_dir,"prism_matched_method_scores.csv"))
fwrite(matched[,.(n_pairs=.N,n_drugs=uniqueN(drug_norm)),by=.(resource,lineage)],file.path(table_dir,"prism_matched_population.csv"))

# 5. Per-lineage tests with shared drug resampling ----------------------------
lineage_results<-list(); lineage_differences<-list()
for(resource_name in c("CPTAC","TPCPA")) for(lin in lineage_levels) {
  d<-matched[resource==resource_name & lineage==lin]
  if(nrow(d)<5L) next
  setorder(d,drug_norm);n<-nrow(d)
  set.seed(seed+match(lin,lineage_levels))
  boot_index<-matrix(sample.int(n,n*bootstrap_iterations,replace=TRUE),n,bootstrap_iterations)
  y<-d$prism_lineage_selectivity
  bs<-matrix(NA_real_,bootstrap_iterations,length(all_methods),dimnames=list(NULL,all_methods))
  observed<-setNames(numeric(length(all_methods)),all_methods)
  for(m in all_methods) {
    x<-d[[m]]; observed[m]<-spearman(x,y)
    for(b in seq_len(bootstrap_iterations)) {i<-boot_index[,b];bs[b,m]<-spearman(x[i],y[i])}
    interval<-ci(bs[,m]);p<-suppressWarnings(cor.test(x,y,method="spearman",exact=FALSE)$p.value)
    lineage_results[[length(lineage_results)+1]]<-data.table(resource=resource_name,lineage=lin,method=m,n_drugs=n,
       rho=observed[m],ci_low=interval[1],ci_high=interval[2],p_value=p,analysis_tier=if(lin %in% primary_lineages) "primary" else "sensitivity")
  }
  for(m in setdiff(all_methods,"Original")) {
    interval<-ci(bs[,m]-bs[,"Original"])
    lineage_differences[[length(lineage_differences)+1]]<-data.table(resource=resource_name,lineage=lin,method=m,
      delta_rho=observed[m]-observed["Original"],ci_low=interval[1],ci_high=interval[2])
  }
  message("Lineage bootstrap: ",resource_name," / ",lin)
}
lineage_results<-rbindlist(lineage_results)
# Main-family BH includes both branches, both resources, and all covered lineages.
lineage_results[,test_family:=fifelse(method %in% main_methods,"main","sensitivity")]
lineage_results[,q_value:=p.adjust(p_value,"BH"),by=test_family]
fwrite(lineage_results,file.path(table_dir,"prism_lineage_correlations.csv"))
fwrite(rbindlist(lineage_differences),file.path(table_dir,"prism_lineage_method_differences.csv"))

# Balanced drug blocks allow exact double-centering instead of iterative demeaning.
# This is algebraically the same two-way rank residualization on a complete grid.
global_results<-list();global_difference<-list();bootstrap_output<-list();permutation_output<-list()
classification_output<-list();classification_difference<-list();loo_output<-list();design_output<-list()
for(resource_name in c("CPTAC","TPCPA")) {
  d<-matched[resource==resource_name & lineage %in% primary_lineages]
  complete_drugs<-d[,.(n_lineages=uniqueN(lineage)),by=drug_norm][n_lineages==5,drug_norm]
  d<-d[drug_norm %in% complete_drugs]
  drug_order<-sort(unique(d$drug_norm));n<-length(drug_order);k<-length(primary_lineages)
  stopifnot(n>10,nrow(d)==n*k)
  to_matrix<-function(column) {
    z<-dcast(d,drug_norm~lineage,value.var=column)
    as.matrix(z[match(drug_order,drug_norm),..primary_lineages])
  }
  y<-to_matrix("prism_lineage_selectivity");xr<-lapply(all_methods,to_matrix);names(xr)<-all_methods
  observed<-vapply(xr,partial_rho,numeric(1),y=y)
  yr<-center_matrix(rank_matrix(y));ranked_x<-lapply(xr,rank_matrix)
  set.seed(seed+1L)
  bootstrap_index<-matrix(sample.int(n,n*bootstrap_iterations,replace=TRUE),n,bootstrap_iterations)
  # Enumerate all 120 lineage permutations, then independently sample per drug.
  patterns<-as.matrix(expand.grid(rep(list(seq_len(k)),k)))
  patterns<-patterns[apply(patterns,1,function(x) length(unique(x))==k),,drop=FALSE]
  set.seed(seed)
  choices<-matrix(sample.int(nrow(patterns),n*permutation_iterations,replace=TRUE),n,permutation_iterations)
  saveRDS(list(drug_order=drug_order,lineage_order=primary_lineages,bootstrap_index=bootstrap_index,
      permutation_patterns=patterns,permutation_choices=choices),file.path(result_dir,paste0(resource_name,"_resampling_design.rds")))
  bs<-matrix(NA_real_,bootstrap_iterations,length(all_methods),dimnames=list(NULL,all_methods))
  null<-matrix(NA_real_,permutation_iterations,length(all_methods),dimnames=list(NULL,all_methods))
  for(b in seq_len(bootstrap_iterations)) {
    idx<-bootstrap_index[,b];yb<-y[idx,,drop=FALSE]
    for(m in all_methods) bs[b,m]<-partial_rho(xr[[m]][idx,,drop=FALSE],yb)
  }
  for(b in seq_len(permutation_iterations)) {
    positions<-seq_len(n)+(patterns[choices[,b],,drop=FALSE]-1L)*n
    for(m in all_methods) null[b,m]<-cor_fast(center_matrix(matrix(ranked_x[[m]][positions],n,k)),yr)
  }
  for(m in all_methods) {
    interval<-ci(bs[,m])
    global_results[[length(global_results)+1]]<-data.table(resource=resource_name,method=m,rho=observed[m],ci_low=interval[1],ci_high=interval[2],
      permutation_p_one_sided=(1+sum(null[,m]>=observed[m]))/(permutation_iterations+1),n_drugs=n,n_pairs=n*k)
    bootstrap_output[[length(bootstrap_output)+1]]<-data.table(resource=resource_name,method=m,iteration=seq_len(bootstrap_iterations),rho=bs[,m])
    permutation_output[[length(permutation_output)+1]]<-data.table(resource=resource_name,method=m,iteration=seq_len(permutation_iterations),rho=null[,m])
    if(m!="Original") {
      interval<-ci(bs[,m]-bs[,"Original"])
      global_difference[[length(global_difference)+1]]<-data.table(resource=resource_name,method=m,delta_rho=observed[m]-observed["Original"],ci_low=interval[1],ci_high=interval[2])
    }
    for(excluded in primary_lineages) {
      keep<-primary_lineages!=excluded
      loo_output[[length(loo_output)+1]]<-data.table(resource=resource_name,method=m,excluded_lineage=excluded,rho=partial_rho(xr[[m]][,keep],y[,keep]))
    }
  }
  # Common labels and common bootstrap blocks give paired performance differences.
  for(fraction in positive_fractions) {
    truth<-label_matrix(y,fraction)
    metrics<-lapply(xr,function(x) classification_metrics(as.vector(truth),as.vector(x)))
    cb<-array(NA_real_,c(bootstrap_iterations,2,length(main_methods)),dimnames=list(NULL,c("AUROC","AUPRC"),main_methods))
    for(b in seq_len(bootstrap_iterations)) {
      idx<-bootstrap_index[,b]
      for(m in main_methods) cb[b,,m]<-classification_metrics(as.vector(truth[idx,]),as.vector(xr[[m]][idx,]))
    }
    for(m in all_methods) for(metric in c("AUROC","AUPRC")) {
      interval<-if(m %in% main_methods) ci(cb[,metric,m]) else c(NA_real_,NA_real_)
      classification_output[[length(classification_output)+1]]<-data.table(resource=resource_name,method=m,positive_fraction=fraction,
         metric=metric,estimate=metrics[[m]][metric],ci_low=interval[1],ci_high=interval[2],prevalence=mean(truth),n_pairs=n*k)
      if(m %in% setdiff(main_methods,"Original")) {
        interval<-ci(cb[,metric,m]-cb[,metric,"Original"])
        classification_difference[[length(classification_difference)+1]]<-data.table(resource=resource_name,method=m,positive_fraction=fraction,
          metric=metric,difference=metrics[[m]][metric]-metrics[["Original"]][metric],ci_low=interval[1],ci_high=interval[2])
      }
    }
  }
  design_output[[resource_name]]<-data.table(resource=resource_name,n_drugs=n,n_lineages=k,n_pairs=n*k,permutations=permutation_iterations,bootstraps=bootstrap_iterations)
  message("Global resampling completed: ",resource_name)
}
global_results<-rbindlist(global_results)
global_results[,test_family:=fifelse(method %in% main_methods,"main","sensitivity")]
global_results[,q_value:=p.adjust(permutation_p_one_sided,"BH"),by=test_family]
fwrite(global_results,file.path(table_dir,"prism_global_results.csv"))
fwrite(rbindlist(global_difference),file.path(table_dir,"prism_global_method_differences.csv"))
fwrite(rbindlist(bootstrap_output),file.path(table_dir,"prism_bootstrap_estimates.csv.gz"))
fwrite(rbindlist(permutation_output),file.path(table_dir,"prism_permutation_null.csv.gz"))
fwrite(rbindlist(classification_output),file.path(table_dir,"prism_classification_metrics.csv"))
fwrite(rbindlist(classification_difference),file.path(table_dir,"prism_classification_differences.csv"))
fwrite(rbindlist(loo_output),file.path(table_dir,"prism_leave_one_lineage_out.csv"))
fwrite(rbindlist(design_output),file.path(table_dir,"prism_resampling_audit.csv"))

# 6–7. Reproducibility checks / plotting handoff ------------------------------
# Check first stored resample against its exact saved design (not a fresh seed).
design<-readRDS(file.path(result_dir,"CPTAC_resampling_design.rds"))
d<-matched[resource=="CPTAC" & drug_norm %in% design$drug_order & lineage %in% design$lineage_order]
mx<-dcast(d,drug_norm~lineage,value.var="Average");my<-dcast(d,drug_norm~lineage,value.var="prism_lineage_selectivity")
cols<-design$lineage_order
x<-as.matrix(mx[match(design$drug_order,drug_norm),..cols]);y<-as.matrix(my[match(design$drug_order,drug_norm),..cols])
i<-design$bootstrap_index[,1]; replay<-partial_rho(x[i,],y[i,])
saved<-rbindlist(bootstrap_output)[resource=="CPTAC" & method=="Average" & iteration==1,rho]
stopifnot(abs(replay-saved)<1e-12)
write_json(list(seed=seed,permutations=permutation_iterations,bootstraps=bootstrap_iterations,
 resampling_replay_pass=TRUE,matching="complete common drug-lineage grid across all eight methods",
 global_statistic="Spearman ranks residualized for drug and lineage; exact double-centering on balanced grids",
 inference="paired drug-cluster bootstrap; identical within-drug lineage permutations across methods",
 auprc="average precision with tied scores grouped",no_outcome_model_selection=TRUE),
 file.path(result_dir,"statistics_manifest.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(result_dir,"sessionInfo_statistics.txt"))
