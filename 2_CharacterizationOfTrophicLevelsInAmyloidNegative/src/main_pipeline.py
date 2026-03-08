import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from scipy.stats import zscore

from functions_pipeline import (
    harmonize_data,
    run_GEC_metrics,
    regression_analysis,
    plot_histogram,
    plot_correlation_with_mean
)

# -------------------------------------------------------
# 1. Parse arguments
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run TL + GEC pipeline")
parser.add_argument("--data-dir", required=True, help="Folder with MAT/CSV files")
parser.add_argument("--out-dir", required=True, help="Where results will be saved")
args = parser.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------
# 2. Load data
# -------------------------------------------------------
print("Loading data...")
TL = loadmat(data_dir / "TL_ADNI3_HCn_dbs80.mat")["TL"]
GEC_ALL = loadmat(data_dir / "GEC_ADNI3_HCn_dbs80.mat")["GEC"]

# Age, gender and education for each subject
demo = pd.read_excel(data_dir / "Demographics.xlsx")
# Scan info per subject
sites = pd.read_excel(data_dir / "Sites.xlsx")

# Combine covariates and site info
cov = pd.concat([demo.reset_index(drop=True), sites.reset_index(drop=True)], axis=1)

# -------------------------------------------------------
# 3. Harmonize data
# -------------------------------------------------------
print("Harmonizing...")
TL_h = harmonize_data(TL, cov)
GEC_h = harmonize_data(GEC_ALL.reshape(len(GEC_ALL), -1), cov)
GEC_h = GEC_h.reshape(GEC_ALL.shape)

# -------------------------------------------------------
# 4. Run GEC metrics
# -------------------------------------------------------
print("Computing GEC metrics...")
metrics = run_GEC_metrics(
    GEC_h,
    out_path=out_dir / "GEC_metrics.mat"
)

# -------------------------------------------------------
# 5. Regression
# -------------------------------------------------------
print("Running regression analysis...")
metric_names = ["out/deg ratio", "clustering", "path len", "betweenness"]
regression_analysis(
    [metrics["out_in_degree_ratio_array_GEC"],
     metrics["clust_array_GEC"],
     metrics["mean_path_len_array_GEC"],
     metrics["betweenness_array_GEC"]],
    metric_names
)

# -------------------------------------------------------
# 6. Plotting
# -------------------------------------------------------
print("Plotting histogram...")
zTL=zscore(TL_h.mean(axis=0))
plot_histogram(zTL, np.percentile(zTL, 33), np.percentile(zTL, 66), bins=9, out_path=str(out_dir / "Histogram.png"))

print("Plotting correlations...")
plot_correlation_with_mean(metrics['out_in_degree_ratio_array_GEC'], TL_h, metrics['out_in_degree_ratio_array_GEC'].mean(axis=0), TL_h.mean(axis=0), 'Out-In Degree Ratio', 'Trophic Level', out_path=str(out_dir / "OutInRatio.png"))
plot_correlation_with_mean(metrics['mean_path_len_array_GEC'], TL_h, metrics['mean_path_len_array_GEC'].mean(axis=0), TL_h.mean(axis=0), 'Mean Path Length', 'Trophic Level', out_path=str(out_dir / "MeanPathLength.png"))

print("Pipeline complete.")
