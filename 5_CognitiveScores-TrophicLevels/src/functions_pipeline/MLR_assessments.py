import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.io import savemat
import numpy as np
import pandas as pd

def significance_marker(p):
	if p < 0.001: return '***'
	elif p < 0.01: return '**'
	elif p < 0.05: return '*'
	else: return 'n.s.'
    
def run_multiple_regression(measure_df, cog_df, cog_name, demographics_df, measure_name, out_path):

	# ================================================================
	# 1. Prepare data (same as your PLS code)
	# ================================================================
	region_cols = [c for c in measure_df.select_dtypes(include=[np.number]).columns if c not in ('ID','Group')]
	
	def melt_df(df, value_name, id_vars=['ID','Group']):
		return df.melt(id_vars=id_vars, value_vars=region_cols,
						var_name='Region', value_name=value_name)

	measure_long = melt_df(measure_df, measure_name)
	dem = demographics_df.copy()
	df = measure_long.merge(dem[['ID','age','gender','edu','Subject ID', 'APOE4_CARRIER']], on='ID', how='left')
	df = df.rename(columns={'Subject ID':'SubjectID'})
	df['SubjectID'] = df['SubjectID'].astype(str)
	df = df.dropna(subset=[measure_name, 'SubjectID'])

	df_wide = df.pivot_table(index='SubjectID', columns='Region', values=measure_name).reset_index()
	cog_df=cog_df.rename(columns={'PTID':'SubjectID'})
	cog_df = cog_df.dropna(subset=[cog_name, 'SubjectID']).copy()
	dem = dem.rename(columns={'Subject ID':'SubjectID'})

	merged = df_wide.merge(dem, on='SubjectID')
	merged = merged.merge(cog_df, on='SubjectID')
	
	
	# ================================================================
	# 2. Regionwise multiple linear regression controlling covariates
	# ================================================================
	results_list = []
	
	for region in region_cols:
		# build design matrix for regression: region + covariates + group
		X = merged[[region,'age','edu', 'gender', 'APOE4_CARRIER']].copy()
	
		X = sm.add_constant(X)
		y = merged[cog_name]
		
		model = sm.OLS(y, X).fit()
		coef = model.params[region]
		pval = model.pvalues[region]

		results_list.append({'region': region, 'coef': coef, 'pval': pval})

	results_df = pd.DataFrame(results_list)

	# ================================================================
	# 3. Correct for multiple comparisons (FDR)
	# ================================================================
	reject, pvals_fdr, _, _ = multipletests(results_df['pval'], alpha=0.05, method='fdr_bh')
	results_df['pval_fdr'] = pvals_fdr
	results_df['significant'] = reject

	render_array = np.zeros(len(region_cols), dtype=float)
	sig_rows = results_df[results_df['significant']]
	for _, r in sig_rows.iterrows():
		region = str(r['region']).strip().lower()
		beta = r['coef']
		# find matching index in region_labels (case-insensitive, stripped)
		idx = next((i for i, lbl in enumerate(region_cols)
					if str(lbl).strip().lower() == region), None)
		if idx is not None:
			render_array[idx] = beta

	savemat(out_path + 'Render.mat', {'render': np.asarray(render_array)})

	# ================================================================
	# 4. Plot significant regions with slope
	# ================================================================
	sig_df = results_df[results_df['significant']].copy()
	print(f"{len(sig_df)} regions significant after FDR correction")

	groups_names = measure_df['Group'].unique()
	labels_scatter = {idx: grp for idx, grp in enumerate(groups_names)}
	colors = sns.color_palette("colorblind", n_colors=len(groups_names))

	results_plot = []

	for _, row in sig_df.iterrows():
		
		region = row['region']
		coef = row['coef']
		
		fig, ax = plt.subplots(figsize=(4, 5))

		# use the subset's Group column so masks align with 'sub'
		groups = merged['Group']
		# build a stable mapping from group values to colors (handles string or numeric groups)
		groups_vals = list(dict.fromkeys(groups.dropna().tolist()))
		desired = groups_names
		unique_groups = [g for lbl in desired for g in groups_vals if labels_scatter.get(g,str(g))==lbl]
		unique_groups += [g for g in groups_vals if g not in unique_groups]
		group_to_color = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
		for g in unique_groups:
			mask = groups == g
			plt.scatter(merged[region][mask], merged[cog_name][mask], label='_nolegend_', color=group_to_color.get(g), alpha=0.7, s=100)
		

		X = merged[[region,'age','edu', 'gender', 'APOE4_CARRIER']].copy()
		X = sm.add_constant(X)
		y = merged[cog_name]
		
		model = sm.OLS(y, X).fit()
		coef = model.params[region]
		pval = model.pvalues[region]
		# create a range of X values (only the region variable varies)
		x_vals = np.linspace(merged[region].min(), merged[region].max(), 100)
		# build a new DataFrame for prediction, using mean values for the covariates
		X_pred = pd.DataFrame({
			'const': 1,
			region: x_vals,
			'age': merged['age'].mean(),
			'edu': merged['edu'].mean(),
			'gender': merged['gender'].mode()[0],
			'APOE4_CARRIER': merged['APOE4_CARRIER'].mode()[0]
		})
		X_pred = sm.add_constant(X_pred)

		# get predicted Y from the fitted model
		y_pred = model.predict(X_pred)

		pval = row['pval_fdr']
		if pd.isnull(pval):
			pval = row.get('pval', np.nan)
		if np.isnan(pval):
			stars = ''
		else:
			marker = significance_marker(pval)
			stars = '' if marker == 'n.s.' else marker

		if measure_name == 'TrophicLevels':
			beta_label = r'$\beta_{TL}$'
		elif measure_name == 'Perturbability':
			beta_label = r'$\beta_{P}$'
		else:
			beta_label = r'$\beta$'
		plt.plot(x_vals, y_pred, color='#800080', lw=5, label=rf'{beta_label}{stars}={coef:.3f}')
		
		
		plt.xlabel('Trophic Level', fontsize=25)
		plt.ylabel(cog_name, fontsize=25)
		plt.title(f"{region}", fontsize=25)
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)		
		plt.legend(fontsize=25, handlelength=0.5)
		plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
		plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

		if region == sig_df['region'].values[0]:
			for spine in ['right', 'top']:
					ax.spines[spine].set_visible(False)
					ax.tick_params(right=False, top=False)
		elif region != sig_df['region'].values[0]:
			fig.set_size_inches(3.5, 5)
			ax.yaxis.set_visible(False)
			ax.set_ylabel('')
			# remove limit lines (spines) at right, left and top
			for spine in ['right', 'top', 'left']:
				ax.spines[spine].set_visible(False)
			# remove ticks on left, right and top
			ax.tick_params(left=False, right=False, top=False)
		
		plt.tight_layout()
		plt.savefig(os.path.join(out_path, f'{region}_{cog_name}.png'), dpi=300)
		plt.close('all')
	
