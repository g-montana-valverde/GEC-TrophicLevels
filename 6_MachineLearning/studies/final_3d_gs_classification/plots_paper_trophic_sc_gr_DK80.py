import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.plotting.roc_curves import plot_roc_curve_on_ax
from src.plotting.confusion_matrix import plot_single_cm


# ---------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.titlesize": 16
})


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
scenario = "MCIpos_vs_ADpos"

# Single feature set: TL combat
feature_set = "all_features_TL_combat"
# Will plot the first n_features_plot + 1 features, consistent with original code
n_features_plot = 9

# Colors / colormap
cm_cmap = "Blues"
color_mean_shap = "tab:blue"
color_sel_freq = "darkblue"

group_comparisons_strs = {
    "HCneg_vs_HCpos": r"HC- vs HC+",
    "HCneg_vs_MCIpos": r"HC- vs MCI+",
    "HCneg_vs_ADpos": r"HC- vs AD+",
    "MCIpos_vs_ADpos": r"MCI+ vs AD+",
}

group_comparisons_labels = {
    "HCneg_vs_HCpos": [r"HC-", r"HC+"],
    "HCneg_vs_MCIpos": [r"HC-", r"MCI+"],
    "HCneg_vs_ADpos": [r"HC-", r"AD+"],
    "MCIpos_vs_ADpos": [r"MCI+", r"AD+"],
}

features_strs = {
    "all_features_TL_combat": "Trophic Levels",
}

# ---------------------------------------------------------------------
# Abbreviation function for SHAP x-axis labels
# ---------------------------------------------------------------------
def abbreviate_feature_name(name: str) -> str:
    """
    Compact abbreviation for ROI feature names.
    """
    # Short, non-ROI features: keep as is
    if " " not in name:
        return name

    tokens = name.split()
    if len(tokens) < 2:
        return name  # defensive

    hemi_token = tokens[0].lower()
    if hemi_token == "left":
        hemi = "L"
    elif hemi_token == "right":
        hemi = "R"
    else:
        hemi = ""
        core_tokens = tokens

    if hemi_token in ("left", "right"):
        core_tokens = tokens[1:]

    abbr_map = {
        "caudal": "caud",
        "rostral": "rost",
        "anterior": "ant",
        "posterior": "post",
        "middle": "mid",
        "inferior": "inf",
        "superior": "sup",
        "lateral": "lat",
        "medial": "med",
        "orbitofrontal": "OF",
        "cingulate": "cing",
        "frontal": "front",
        "temporal": "temp",
        "parietal": "par",
        "occipital": "occ",
        "entorhinal": "entorh",
        "isthmus": "isth",
        "precuneus": "precun",
        "pars": "pars",
        "triangularis": "triang",
        "opercularis": "operc",
        "orbitalis": "orbit",
        "supramarginal": "supra",
        "paracentral": "paracent",
        "postcentral": "postcen",
        "precentral": "precen",
        "transverse": "transv",
    }

    core_abbr = []
    for w in core_tokens:
        lw = w.lower()
        if lw in abbr_map:
            core_abbr.append(abbr_map[lw])
        else:
            core_abbr.append(w[:5])

    if hemi:
        label = " ".join([hemi] + core_abbr)
    else:
        label = " ".join(core_abbr)
    return label

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
path_results = path_repo / "Results" / "final_3d_gs_classification_trophic_DK80_sc_gr_det_run2"
path_sce_results = path_results / "LogReg" / scenario / feature_set

# ---------------------------------------------------------------------
# Load ROC data
# ---------------------------------------------------------------------
roc_data_test = joblib.load(path_sce_results / "avg_roc_data_test.joblib")

# ---------------------------------------------------------------------
# Load and average confusion matrices across folds
# ---------------------------------------------------------------------
cms = []

fold_dirs = [
    d for d in os.listdir(path_sce_results)
    if (path_sce_results / d).is_dir() and re.fullmatch(r"fold_\d+", d)
]
fold_dirs.sort(key=lambda d: int(d.split("_")[1]))

if not fold_dirs:
    raise FileNotFoundError(f"No fold directories found under: {path_sce_results}")

for d in fold_dirs:
    cm_path = path_sce_results / d / "test_cm.txt"
    if not cm_path.exists():
        raise FileNotFoundError(f"Expected confusion matrix missing: {cm_path}")
    cms.append(np.loadtxt(cm_path))

avg_cm_test = np.mean(np.asarray(cms), axis=0)

# ---------------------------------------------------------------------
# Load SHAP feature importance
# ---------------------------------------------------------------------
df_imp = pd.read_csv(path_sce_results / "summary_importance.csv")

feature_importance_raw = {
    feature: value for feature, value in zip(df_imp["feature"], df_imp["mean_abs_SHAP"])
}

feature_importance_clean = {
    feat: value
    for i, (feat, value) in enumerate(feature_importance_raw.items())
    if i <= n_features_plot
}

percentage_selected = df_imp["selection_freq"].tolist()[:n_features_plot + 1]
importance_sd = df_imp["sd"].tolist()[:n_features_plot + 1]

# ---------------------------------------------------------------------
# Plotting: ROC, Confusion Matrix, SHAP + Selection Frequency
# ---------------------------------------------------------------------

# ===================== ROC (separate figure) =======================
fig_roc, ax_roc = plt.subplots(figsize=(5, 5))

plot_roc_curve_on_ax(
    roc_data_test["fpr"],
    roc_data_test["tpr"],
    features_strs[feature_set],
    ax_roc,
    roc_data_test["std_tpr"],
)
ax_roc.legend()
ax_roc.set_box_aspect(1)
ax_roc.set_title(f"ROC Curve ({group_comparisons_strs[scenario]})")

fig_roc.tight_layout()
fig_roc.savefig(
    path_results / f"classification_roc_{scenario.lower()}_tlcombat.pdf",
    dpi=600,
)
fig_roc.savefig(
    path_results / f"classification_roc_{scenario.lower()}_tlcombat.svg",
    dpi=600,
)

# ===================== Confusion Matrix (separate figure) ==========
fig_cm, ax_cm = plt.subplots(figsize=(5, 5))

plot_single_cm(
    avg_cm_test,
    group_comparisons_labels[scenario],
    ax_cm,
    vmin=0,
    vmax=1,
    cmap=cm_cmap,
    fontsize=18,
)
ax_cm.set_title(f"{features_strs[feature_set]}\nConfusion Matrix")
ax_cm.set_box_aspect(1)

fig_cm.tight_layout()
fig_cm.savefig(
    path_results / f"classification_cm_{scenario.lower()}_tlcombat.pdf",
    dpi=600,
)
fig_cm.savefig(
    path_results / f"classification_cm_{scenario.lower()}_tlcombat.svg",
    dpi=600,
)

# ===================== SHAP + Selection Frequency (separate figure)
features = list(feature_importance_clean.keys())
x_values = np.arange(len(features))
abbrev_features = [abbreviate_feature_name(f) for f in features]

total_bar_width = 0.8
single_bar_width = total_bar_width / 2

fig_shap, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar_twin = ax_bar.twinx()

# Left y-axis: mean SHAP value
ax_bar.bar(
    x_values - single_bar_width / 2,
    list(feature_importance_clean.values()),
    # yerr=importance_sd,
    width=single_bar_width,
    label="Mean SHAP value",
    color=color_mean_shap,
)

# Right y-axis: selection frequency
ax_bar_twin.bar(
    x_values + single_bar_width / 2,
    percentage_selected,
    width=single_bar_width,
    label="Selection Frequency",
    color=color_sel_freq,
)

ax_bar_twin.set(ylim=(0, 1))

ax_bar.set_xticks(x_values)
ax_bar.set_xticklabels(abbrev_features)
plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")

ax_bar.set_title(f"Feature Importance in {features_strs[feature_set]}")
ax_bar.set_ylabel("Mean SHAP value")
ax_bar_twin.set_ylabel("Feature Selection Frequency")

handles1, labels1 = ax_bar.get_legend_handles_labels()
handles2, labels2 = ax_bar_twin.get_legend_handles_labels()
ax_bar.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

ax_bar.spines['left'].set_color(color_mean_shap)
ax_bar.tick_params(axis='y', colors=color_mean_shap)
ax_bar_twin.spines['right'].set_color(color_sel_freq)
ax_bar_twin.tick_params(axis='y', colors=color_sel_freq)

ax_bar_twin.set_xlim(
    x_values[0] - 1.25 * single_bar_width,
    x_values[-1] + 1.25 * single_bar_width
)

fig_shap.tight_layout()
fig_shap.savefig(
    path_results / f"classification_shap_{scenario.lower()}_tlcombat.pdf",
    dpi=600,
)
fig_shap.savefig(
    path_results / f"classification_shap_{scenario.lower()}_tlcombat.svg",
    dpi=600,
)

plt.show()
