from src.utils.summary_functions import collect_selected_metrics
import pandas as pd
from pathlib import Path

path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
path_results = path_repo / "Results" / "final_3d_gs_classification_trophic_DK80_sc_gr_det_run2"
results_file_name = "stats_results_folds.json"
metric_bases = [
    "Balanced Accuracy (Test)",
    "AUC (Test)",
    "Sensitivity (Test)",
    "Specificity (Test)",
    "F1 (Test)",
    "Best NF"
]

df_all_metrics = collect_selected_metrics(path_results, metric_bases, results_file_name)
print(df_all_metrics.to_string(index=False))  # or df_all_metrics.head()

# Merge the Mean and Std into same column to have stats:

# For each base metric, create the merged column with ±
for base in metric_bases:
    mean_col = f'{base} Mean'
    std_col = f'{base} Std'
    df_all_metrics[base] = (df_all_metrics[mean_col].round(3).astype(str) +
                            ' ± ' + df_all_metrics[std_col].round(3).astype(str))

# Optionally drop the original Mean/Std columns
df_all_metrics = df_all_metrics.drop(
    columns=[f'{base} {suffix}' for base in metric_bases for suffix in ['Mean', 'Std']]
)

# Additionally, change the info in the Feature Set
features_strs = {
    "all_features_TL_combat": "Trophic Levels",
}
df_all_metrics["Feature Set"] = df_all_metrics["Feature Set"].apply(
    lambda x: features_strs[x]
)

# Export the results for each group in a different sheet and improve names
group_comparisons_strs = {
    "HCneg_vs_HCpos": "HC (AB-) vs HC (AB+)",
    "HCneg_vs_MCIpos": "HC (AB-) vs MCI (AB+)",
    "HCneg_vs_ADpos": "HC (AB-) vs AD (AB+)",
    "MCIpos_vs_ADpos": "MCI (AB+) vs AD (AB+)",
}

name_file = 'group_comparison_results.xlsx'
with pd.ExcelWriter(path_results / name_file, engine='xlsxwriter') as writer:
    for group in group_comparisons_strs.keys():
        # Filter the DataFrame for each group comparison
        df_group = df_all_metrics[df_all_metrics['Group Comparison'] == group]

        # Additionally, we improve the group comparisons names
        df_group["Group Comparison"] = df_group["Group Comparison"].apply(
            lambda x: group_comparisons_strs[x]
        )
        # Write to a separate sheet
        df_group.to_excel(
            writer, sheet_name=group_comparisons_strs[group], index=False
        )
