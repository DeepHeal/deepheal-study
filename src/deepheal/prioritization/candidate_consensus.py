from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

from analysis_utils import add_common_args, ensure_out_dirs, external_group, friendly_pathway_label, normalize_name, resolve_args, set_plot_style, write_json


SUP_AG_SHORTLIST_THRESHOLD = 0.98
SUP_LG_SHORTLIST_THRESHOLD = 0.99
AVG_AG_SHORTLIST_THRESHOLD = 0.83
AVG_LG_RECORDED_THRESHOLD = 0.99


def load_prediction_table(path: Path, score_name: str) -> pd.DataFrame:
    cols = ["Drug_Name", "prob_antineoplastic", "predicted_class"]
    df = pd.read_csv(path, usecols=cols)
    return df.rename(columns={"prob_antineoplastic": score_name, "predicted_class": f"{score_name}_class"})


def pathway_maps(repo_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    supcon = repo_root / "results" / "supcon"
    canonical = pd.read_csv(supcon / "supcon_latent_top_pathways.csv")
    oncogenic = pd.read_csv(supcon / "supcon_c6_latent_top_pathways.csv")
    canonical_map = (
        canonical.sort_values(["latent_idx", "rank"])
        .drop_duplicates("latent_idx")
        .set_index("latent_idx")["pathway"]
        .map(friendly_pathway_label)
        .to_dict()
    )
    oncogenic_map = (
        oncogenic.sort_values(["latent_idx", "rank"])
        .drop_duplicates("latent_idx")
        .set_index("latent_idx")["pathway"]
        .map(friendly_pathway_label)
        .to_dict()
    )
    return canonical_map, oncogenic_map


def build_candidate_evidence(repo_root: Path, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    supcon = repo_root / "results" / "supcon"
    average = repo_root / "results" / "average"
    tables = out_root / "tables"

    sup_ag = load_prediction_table(supcon / "supcon_autogluon_unknown_predictions.csv", "sup_ag")
    sup_lg = load_prediction_table(supcon / "supcon_lgmlvq_unknown_predictions.csv", "sup_lg")
    avg_ag = load_prediction_table(average / "average_autogluon_unknown_predictions.csv", "avg_ag")
    avg_lg = load_prediction_table(average / "average_lgmlvq_unknown_predictions.csv", "avg_lg")
    predictions = sup_ag.merge(sup_lg, on="Drug_Name").merge(avg_ag, on="Drug_Name").merge(avg_lg, on="Drug_Name")
    predictions["consensus_score"] = (
        0.35 * predictions["sup_ag"] + 0.35 * predictions["sup_lg"] + 0.20 * predictions["avg_ag"] + 0.10 * predictions["avg_lg"]
    )
    predictions.sort_values("consensus_score", ascending=False).to_csv(tables / "unknown_candidate_all_predictions_ranked.csv", index=False)

    shortlist_mask = (
        (predictions["sup_ag"] >= SUP_AG_SHORTLIST_THRESHOLD)
        & (predictions["sup_lg"] >= SUP_LG_SHORTLIST_THRESHOLD)
        & (predictions["avg_ag"] >= AVG_AG_SHORTLIST_THRESHOLD)
    )
    shortlist = predictions[shortlist_mask].copy()
    strict_shortlist = predictions[shortlist_mask & (predictions["avg_lg"] >= AVG_LG_RECORDED_THRESHOLD)].copy()
    sensitivity = pd.DataFrame(
        {
            "rule": ["script_current_rule", "strict_if_avg_lg_threshold_enforced"],
            "n_candidates": [len(shortlist), len(strict_shortlist)],
            "candidate_order": [
                "; ".join(shortlist.sort_values("consensus_score", ascending=False)["Drug_Name"].tolist()),
                "; ".join(strict_shortlist.sort_values("consensus_score", ascending=False)["Drug_Name"].tolist()),
            ],
        }
    )
    sensitivity.to_csv(tables / "unknown_candidate_avg_lg_threshold_sensitivity.csv", index=False)

    meta = pd.read_excel(repo_root / "data" / "raw" / "875_Drugs_Metainformation_ATC_Classfication.xlsx")
    meta["norm_name"] = meta["Drug_name"].map(normalize_name)
    shortlist["norm_name"] = shortlist["Drug_Name"].map(normalize_name)
    shortlist = shortlist.merge(
        meta[
            [
                "norm_name",
                "Drug_name",
                "match_status",
                "name",
                "drugbank_id",
                "Target",
                "Therapeutic_effect",
                "atc_level_1_name",
                "atc_level_2_name",
            ]
        ],
        on="norm_name",
        how="left",
    )
    shortlist["external_group"] = shortlist["Therapeutic_effect"].map(external_group)

    embed = pd.read_csv(supcon / "supcon_consensus_embedding_full_including_marginal.csv")
    latent_cols = [col for col in embed.columns if col.startswith("latent_")]
    known_anti = embed[embed["Label_Int"] == 1].reset_index(drop=True)
    known_labelled = embed[embed["Label_Int"].isin([0, 1])].copy()
    relevance = pd.read_csv(supcon / "supcon_lgmlvq_feature_relevance.csv").set_index("feature")["relevance"]
    canonical_map, oncogenic_map = pathway_maps(repo_root)
    valid_latents = {int(key) for key in canonical_map}.union(int(key) for key in oncogenic_map)

    z_scores = (embed[latent_cols] - embed[latent_cols].mean()) / embed[latent_cols].std(ddof=0).replace(0, 1.0)
    anti_matrix = known_anti[latent_cols].to_numpy()
    records: list[dict[str, object]] = []
    latent_records: list[dict[str, object]] = []

    for _, row in shortlist.iterrows():
        drug_name = row["Drug_Name"]
        candidate = embed.loc[embed["Drug_Name"] == drug_name]
        if candidate.empty:
            continue
        candidate = candidate.iloc[0]
        candidate_vec = candidate[latent_cols].to_numpy(dtype=float).reshape(1, -1)
        cosine_scores = cosine_similarity(candidate_vec, anti_matrix)[0]
        top_idx = np.argsort(cosine_scores)[-3:][::-1]

        cluster = int(candidate["Cluster"])
        cluster_known = known_labelled.loc[known_labelled["Cluster"] == cluster]
        anti_fraction = float((cluster_known["Label_Int"] == 1).mean()) if len(cluster_known) else np.nan
        z_row = z_scores.loc[embed["Drug_Name"] == drug_name, latent_cols].iloc[0]
        latent_score = (z_row.abs() * relevance).sort_values(ascending=False)
        latent_score = latent_score[[int(feature.split("_")[1]) in valid_latents for feature in latent_score.index]]
        top_latents = latent_score.head(3)

        record: dict[str, object] = {
            "Drug_Name": drug_name,
            "Target": row.get("Target"),
            "Therapeutic_effect": row.get("Therapeutic_effect"),
            "external_group": row["external_group"],
            "consensus_score": float(row["consensus_score"]),
            "sup_ag": float(row["sup_ag"]),
            "sup_lg": float(row["sup_lg"]),
            "avg_ag": float(row["avg_ag"]),
            "avg_lg": float(row["avg_lg"]),
            "Cluster": cluster,
            "cluster_antineoplastic_fraction": anti_fraction,
        }

        for rank, idx in enumerate(top_idx, start=1):
            neighbor = known_anti.iloc[int(idx)]
            record[f"neighbor_{rank}_name"] = neighbor["Drug_Name"]
            record[f"neighbor_{rank}_similarity"] = float(cosine_scores[int(idx)])

        for rank, (feature, score) in enumerate(top_latents.items(), start=1):
            latent_idx = int(feature.split("_")[1])
            canonical_label = canonical_map.get(latent_idx, "Unannotated")
            oncogenic_label = oncogenic_map.get(latent_idx, "Unannotated")
            record[f"top_latent_{rank}"] = feature
            record[f"top_latent_{rank}_score"] = float(score)
            record[f"top_latent_{rank}_canonical"] = canonical_label
            record[f"top_latent_{rank}_oncogenic"] = oncogenic_label
            latent_records.append(
                {
                    "Drug_Name": drug_name,
                    "rank": rank,
                    "feature": feature,
                    "score": float(score),
                    "canonical_pathway": canonical_label,
                    "oncogenic_pathway": oncogenic_label,
                }
            )

        record["top_latent_summary"] = "; ".join(
            f"{record.get(f'top_latent_{rank}', 'NA')} | {record.get(f'top_latent_{rank}_canonical', 'Unannotated')} | {record.get(f'top_latent_{rank}_oncogenic', 'Unannotated')}"
            for rank in range(1, 4)
            if record.get(f"top_latent_{rank}")
        )
        records.append(record)

    evidence_columns = [
        "Drug_Name",
        "Target",
        "Therapeutic_effect",
        "external_group",
        "consensus_score",
        "sup_ag",
        "sup_lg",
        "avg_ag",
        "avg_lg",
        "Cluster",
        "cluster_antineoplastic_fraction",
        "neighbor_1_name",
        "neighbor_1_similarity",
        "neighbor_2_name",
        "neighbor_2_similarity",
        "neighbor_3_name",
        "neighbor_3_similarity",
        "top_latent_1",
        "top_latent_1_score",
        "top_latent_1_canonical",
        "top_latent_1_oncogenic",
        "top_latent_2",
        "top_latent_2_score",
        "top_latent_2_canonical",
        "top_latent_2_oncogenic",
        "top_latent_3",
        "top_latent_3_score",
        "top_latent_3_canonical",
        "top_latent_3_oncogenic",
        "top_latent_summary",
    ]
    latent_columns = ["Drug_Name", "rank", "feature", "score", "canonical_pathway", "oncogenic_pathway"]
    evidence = pd.DataFrame(records, columns=evidence_columns)
    if not evidence.empty:
        evidence = evidence.sort_values("consensus_score", ascending=False).reset_index(drop=True)
    latent_long = pd.DataFrame(latent_records, columns=latent_columns)
    evidence.to_csv(tables / "unknown_candidate_consensus_evidence.csv", index=False)
    latent_long.to_csv(tables / "unknown_candidate_latent_evidence_long.csv", index=False)

    current_order = shortlist.sort_values("consensus_score", ascending=False)["Drug_Name"].tolist()
    strict_order = strict_shortlist.sort_values("consensus_score", ascending=False)["Drug_Name"].tolist()
    summary = {
        "n_unknown_total": int(len(predictions)),
        "shortlist_rule": {
            "sup_ag_min": SUP_AG_SHORTLIST_THRESHOLD,
            "sup_lg_min": SUP_LG_SHORTLIST_THRESHOLD,
            "avg_ag_min": AVG_AG_SHORTLIST_THRESHOLD,
            "avg_lg_min_recorded_but_not_enforced": AVG_LG_RECORDED_THRESHOLD,
        },
        "shortlist_note": "The source script defines AVG_LG_SHORTLIST_THRESHOLD=0.99 but does not apply avg_lg in the current shortlist filter.",
        "n_supcon_autogluon_high_threshold": int((predictions["sup_ag"] >= SUP_AG_SHORTLIST_THRESHOLD).sum()),
        "n_supcon_lgmlvq_high_threshold": int((predictions["sup_lg"] >= SUP_LG_SHORTLIST_THRESHOLD).sum()),
        "n_average_autogluon_high_threshold": int((predictions["avg_ag"] >= AVG_AG_SHORTLIST_THRESHOLD).sum()),
        "n_average_lgmlvq_high_threshold": int((predictions["avg_lg"] >= AVG_LG_RECORDED_THRESHOLD).sum()),
        "n_high_confidence_consensus": int(len(evidence)),
        "n_external_antineoplastic": int((evidence.get("external_group", pd.Series(dtype=object)) == "External antineoplastic").sum()),
        "n_external_non_antineoplastic": int((evidence.get("external_group", pd.Series(dtype=object)) == "External non-antineoplastic").sum()),
        "shortlist": evidence["Drug_Name"].tolist() if "Drug_Name" in evidence.columns else [],
        "strict_avg_lg_shortlist": strict_order,
        "enforcing_avg_lg_threshold_changes_shortlist": current_order != strict_order,
    }
    write_json(tables / "unknown_candidate_summary.json", summary)
    return evidence, latent_long, summary


def build_unknown_consensus_figure(evidence: pd.DataFrame, out_path: Path) -> None:
    set_plot_style(font_size=9, dpi=300)
    evidence = evidence.copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if evidence.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            "No high-confidence Unknown compounds under current materialized outputs",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.40,
            "Current SupCon AutoGluon unknown predictions do not meet sup_ag >= 0.98.",
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return
    evidence["label"] = evidence.apply(
        lambda row: f"{row['Drug_Name']} ({row['Target']})" if pd.notna(row["Target"]) else row["Drug_Name"], axis=1
    )
    evidence = evidence.sort_values("consensus_score", ascending=False)
    prob_cols = ["sup_ag", "sup_lg", "avg_ag", "avg_lg"]
    prob_labels = ["SupCon\nAutoGluon", "SupCon\nLGMLVQ", "Average\nAutoGluon", "Average\nLGMLVQ"]
    score_df = evidence[prob_cols].rename(columns=dict(zip(prob_cols, prob_labels)))
    colors = {
        "External antineoplastic": "#1f78b4",
        "External non-antineoplastic": "#d95f02",
        "No external match": "#666666",
    }

    fig = plt.figure(figsize=(12.4, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.18)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1], sharey=ax_heat)

    sns.heatmap(
        score_df,
        ax=ax_heat,
        cmap="YlOrRd",
        vmin=0.80,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Probability of antineoplastic class"},
        yticklabels=evidence["label"],
    )
    ax_heat.set_title("Consensus predictions for high-confidence Unknown compounds")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    y_pos = np.arange(len(evidence)) + 0.5
    ax_bar.barh(y_pos, evidence["consensus_score"], color=evidence["external_group"].map(colors), edgecolor="black", linewidth=0.4)
    ax_bar.set_xlim(0.94, 1.001)
    ax_bar.set_xlabel("Weighted consensus score")
    ax_bar.set_title("External metadata check")
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    for y, score, effect in zip(y_pos, evidence["consensus_score"], evidence["Therapeutic_effect"].fillna("No external match")):
        ax_bar.text(min(float(score) + 0.0012, 0.9995), y, str(effect), va="center", fontsize=8)
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", label=label, markerfacecolor=color, markersize=10)
        for label, color in colors.items()
    ]
    ax_bar.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute DeepHeal unknown-candidate consensus evidence and figure.")
    add_common_args(parser)
    args = resolve_args(parser.parse_args())
    ensure_out_dirs(args.out_root)
    evidence, _, summary = build_candidate_evidence(args.repo_root, args.out_root)
    build_unknown_consensus_figure(evidence, args.out_root / "figures" / "deepheal_unknown_candidate_consensus.png")
    pass  # Submission bookkeeping is outside the article analysis.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
