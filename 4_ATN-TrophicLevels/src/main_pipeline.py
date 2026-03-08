import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import json

from functions_pipeline import (
	harmonize_data,
	LinearMixedEffectsRidge
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
derivatives_dir = Path(args.derivatives_dir)
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

ABeta_df=pd.read_csv(data_dir + 'ADNI3_AmyloidBeta-PET_Centiloids.csv')
Tau_df=pd.read_csv(data_dir + 'ADNI3_Tau-PET_SURV.csv')
GMV_df=pd.read_csv(data_dir + 'GMV.csv')

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
TL_h[region_labels] = harmonize_data(tl_df[region_labels].to_numpy(), cov)
ABeta_h[region_labels] = harmonize_data(ABeta_df[region_labels].to_numpy(), cov)
Tau_h[region_labels] = harmonize_data(Tau_df[region_labels].to_numpy(), cov)
GMV_h[region_labels] = harmonize_data(GMV_df[region_labels].to_numpy(), cov)

# -------------------------------------------------------
# 4. Convert to Network Level
# -------------------------------------------------------
print("Computing TL network-level...")
TL_net_h = TL_h[['ID', 'Group']].copy()
ABeta_net_h = ABeta_h[['ID', 'Group']].copy()
Tau_net_h = Tau_h[['ID', 'Group']].copy()
GMV_net_h = GMV_h[['ID', 'Group']].copy()
for net in net_names:
	regions_in_net = [region for region, sfn_map in reg2net.items() if sfn in (sfn_map or "") and region in TL_h.columns]
	exclude_subcortical = ['hippocampus', 'amygdala', 'thalamus', 'caudate', 'accumbens', 'putamen', 'gpe', 'gpi', 'stn']
	regions_in_net = [region for region in regions_in_net if not any(keyword in region.lower() for keyword in exclude_subcortical)]
	if regions_in_sfn:
		TL_net_h[net] = TL_h[regions_in_net].mean(axis=1)
		ABeta_net_h[net] = ABeta_h[regions_in_net].mean(axis=1)
		Tau_net_h[net] = Tau_h[regions_in_net].mean(axis=1)
		GMV_net_h[net] = GMV_h[regions_in_net].mean(axis=1)
	else:
			TL_net_h[net] = np.nan; ABeta_net_h[net] = np.nan; Tau_net_h[net] = np.nan; GMV_net_h[net] = np.nan

# -------------------------------------------------------
# 5. Linear Mixed Effects Models with Ridge Regularization
# -------------------------------------------------------
print("Running Linear Mixed Effects Models with Ridge Regularization...")

# 5.1. LME - Region Level
LinearMixedEffectsRidge(TL_h, ABeta_h, Tau_h, GMV_h, demo, 'TrophicLevel', out_path=out_dir + "/RegionLevel/")
# 5.2. LME - Network
LinearMixedEffectsRidge(TL_net_h, ABeta_net_h, Tau_net_h, GMV_net_h, demo, 'TrophicLevel', out_path=out_dir + "/NetworkLevel/")
