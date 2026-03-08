import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import json

from functions_pipeline import (
	harmonize_data,
	ADAS_filtering,
	CDR_filtering,
	MoCA_filtering,
	MMSE_filtering,
	run_multiple_regression
)

# -------------------------------------------------------
# 1. Parse arguments
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run TL and Directedness Group Comparisons")
parser.add_argument("--data-dir", required=True, help="Folder with MAT/CSV files")
parser.add_argument("--deriv-dir", required=True, help="Folder with derivatives subject's ID")
parser.add_argument("--out-dir", required=True, help="Where results will be saved")
args = parser.parse_args()

data_dir = Path(args.data_dir)
derivatives_dir = Path(args.deriv_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

group_class = 'HCn-HCp-MCIp-ADp'

parts = [p.strip() for p in group_class.split('-') if p.strip()]
if not parts:
	raise ValueError(f"Invalid group_class: {group_class!r}")
for i, part in enumerate(parts, start=1):
	globals()[f"g{i}"] = part

# -------------------------------------------------------
# 2. Load data
# -------------------------------------------------------
region_labels = pd.read_csv(data_dir + 'dbs80_labels.txt', sep="\t", header=None)[1].values
net_names = ['VN', 'SMN', 'DAN', 'SAN', 'LN', 'CN', 'DMN']
with open(data_dir + 'reg2net.json', 'r') as f_json:
	reg2net = json.load(f_json)
  
print("Loading data...")
groups = [globals()[k] for k in sorted(globals().keys()) if k.startswith('g') and k[1:].isdigit()]
dfs = []
for grp in groups:
  data_tl = loadmat(data_dir / f"TL_ADNI3_{grp}_dbs80.mat")["TL"]
  deriv_dir = derivatives_dir / grp / 'derivatives'
  subs = sorted(int(p.name.removeprefix('sub-')) for p in deriv_dir.iterdir() if p.is_dir() and p.name.startswith('sub-'))
  df = pd.DataFrame(data_tl, columns=region_labels); df.insert(0, 'ID', subs); df.insert(1, 'Group', grp); dfs.append(df)

tl_df = pd.concat(dfs, ignore_index=True)

# Age, gender and education for each subject
demo = pd.read_excel(data_dir / "Demographics.xlsx")
# Scan info per subject
sites = pd.read_excel(data_dir / "Sites.xlsx")

# Combine covariates and site info
cov = pd.concat([demo.reset_index(drop=True), sites.reset_index(drop=True)], axis=1)

# Assessments
ADAS_df=ADAS_filtering(pd.read_csv(data_dir + 'ADAS_original.csv'), demo)
CDR_df=CDR_filtering(pd.read_csv(data_dir + 'CDR_original.csv'), demo)
MoCA_df=MoCA_filtering(pd.read_csv(data_dir + 'MoCA_original.csv'), demo)
MMSE_df=MMSE_filtering(pd.read_csv(data_dir + 'MMSE_original.csv'), demo)

assessments=['ADAS-Cog-13', 'CDR-SB', 'MoCA', 'MMSE']


# -------------------------------------------------------
# 3. Harmonize data
# -------------------------------------------------------
print("Harmonizing...")
TL_h= tl_df.copy()
TL_h[region_labels] = harmonize_data(tl_df[region_labels].to_numpy(), cov)

# -------------------------------------------------------
# 4. Convert to Network Level
# -------------------------------------------------------
print("Computing TL network-level...")
TL_net_h = TL_h[['ID', 'Group']].copy()
for net in net_names:
	regions_in_net = [region for region, sfn_map in reg2net.items() if sfn in (sfn_map or "") and region in TL_h.columns]
	exclude_subcortical = ['hippocampus', 'amygdala', 'thalamus', 'caudate', 'accumbens', 'putamen', 'gpe', 'gpi', 'stn']
	regions_in_net = [region for region in regions_in_net if not any(keyword in region.lower() for keyword in exclude_subcortical)]
	if regions_in_net:
		TL_net_h[net] = TL_h[regions_in_net].mean(axis=1)
	else:
		TL_net_h[net] = np.nan

# -------------------------------------------------------
# 5. Multiple Linear Regression Models
# -------------------------------------------------------
print("Running Multiple Linear Regression Models...")

for test in assessments:
	cog_df = df.copy()
	# 5.1. MLR - Region Level
	run_multiple_regression(TL_h, cog_df, test, demo, 'TrophicLevel', out_path=out_dir + "/RegionLevel/")
	# 5.2. MLR - Network
	run_multiple_regression(TL_net_h, cog_df, test, demo, 'TrophicLevel', out_path=out_dir + "/NetworkLevel/")
