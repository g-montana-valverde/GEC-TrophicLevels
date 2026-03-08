import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D
from scipy.io import savemat

def significance_marker(p):
	if p < 0.001: return '***'
	elif p < 0.01: return '**'
	elif p < 0.05: return '*'
	else: return 'n.s.'
    
def LinearMixedEffectsRidge(measure_df, ABeta_df, Tau_df, Volume_df, demographics_df, measure_name, out_path):

	# ================================================================
	# 1. Prepare long-format dataframe (same as your original)
	# ================================================================
	# set region_cols to numeric columns in measure_df excluding ID and Group
	region_cols = [c for c in measure_df.select_dtypes(include=[np.number]).columns if c not in ('ID','Group')]
	

	def melt_df(df, value_name, id_vars=['ID','Group']):
		return df.melt(id_vars=id_vars, value_vars=region_cols,
						var_name='Region', value_name=value_name)

	measure_long = melt_df(measure_df, measure_name)
	abeta_long  = melt_df(ABeta_df, 'ABeta')
	tau_long    = melt_df(Tau_df, 'Tau')
	vol_long    = melt_df(Volume_df, 'Volume')

	df = measure_long.merge(abeta_long, on=['ID','Group','Region'], how='inner')
	df = df.merge(tau_long, on=['ID','Group','Region'], how='inner')
	df = df.merge(vol_long, on=['ID','Group','Region'], how='inner')

	dem = demographics_df.copy()
	df = df.merge(dem[['ID','age','gender','edu','Subject ID', 'APOE4_CARRIER']], on='ID', how='left')

	df = df.rename(columns={'Subject ID':'SubjectID'})
	df['SubjectID'] = df['SubjectID'].astype(str)

	df = df.dropna(subset=[measure_name,'ABeta','Tau','Volume','SubjectID'])

	cols = [measure_name,'ABeta','Tau','Volume','age','gender','edu','SubjectID','Region', 'Group', 'APOE4_CARRIER']
	df = df[cols].dropna().copy()
	
	
	# ================================================================
	# 2. Standardize numeric features
	# ================================================================

	# standardize predictors (important for ridge)
	num_features = ['ABeta','Tau','Volume','age','edu']
	
	scaler = StandardScaler()
	df[num_features] = scaler.fit_transform(df[num_features])
	
	# encode gender as numeric if needed
	df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(0)
	
	# ================================================================
	# 3. Create design matrix with dummies for SubjectID and Region
	# ================================================================

	# Create design matrix with dummies for SubjectID and Region
	X_main = df[['ABeta','Tau','Volume','age','gender','edu', 'APOE4_CARRIER']]
	
	# subject and region dummies
	subj_dummies = pd.get_dummies(df['SubjectID'], prefix='S', drop_first=False)
	reg_dummies  = pd.get_dummies(df['Region'],    prefix='R', drop_first=False)
	
	# ================================================================
	# 4. Add Region × Predictor interactions
	# ================================================================
	# multiply each pathology predictor by each region dummy
	interactions = {}	
	for var in ['ABeta','Tau','Volume']:
		for reg in df['Region'].unique():
			col_name = f'{var}_x_{reg}'
			interactions[col_name] = df[var] * (df['Region'] == reg).astype(int)

	inter_df = pd.DataFrame(interactions)
	X = pd.concat([X_main, subj_dummies, reg_dummies, inter_df], axis=1)
	y = df[measure_name].values

	# ================================================================
	# 5. RidgeCV with cross-validation
	# ================================================================

	# Use RidgeCV to choose alpha by cross-validation
	alphas = np.logspace(-3, 3, 50)
	cv = KFold(n_splits=5, shuffle=True, random_state=42)
	model = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
	model.fit(X, y)

	full_r2 = r2_score(y, model.predict(X))

	# ================================================================
	# 6. Region-wise mixed-effects significance tests
	# ================================================================


	def run_lme_for_predictor(regions, predictor):
		results = []
		print(f"\nRunning LME per top {predictor} region...")
		for reg in regions:
			sub = df[df['Region'] == reg].copy()
			formula = f'{measure_name} ~ ABeta + Tau + Volume + age + gender + edu + APOE4_CARRIER'
			
			md = smf.mixedlm(formula, data=sub, groups=sub['SubjectID'])
			mdf = md.fit(method='lbfgs', reml=False)
			coefs = mdf.summary().tables[1]
			pval = coefs.loc[predictor, 'P>|z|'] if predictor in coefs.index else np.nan
			results.append({
				'Region': reg,
				f'{predictor}_beta': mdf.params.get(predictor, np.nan),
				f'{predictor}_pval': pval
			})
		return pd.DataFrame(results)

	lme_tau_df = run_lme_for_predictor(region_cols, 'Tau')

	lme_abeta_df = run_lme_for_predictor(region_cols, 'ABeta')

	lme_volume_df = run_lme_for_predictor(region_cols, 'Volume')
	

	# FDR correction across all tests (Benjamini-Hochberg)

	# Combine into long table for correction
	def stack_results(dfs, predictor_name):
		out = []
		for df_res in dfs:
			if df_res is None or df_res.empty:
				continue
			pred = predictor_name
			for _, row in df_res.iterrows():
				out.append({'Region': row['Region'],
							'predictor': pred,
							'beta': row[f'{pred}_beta'],
							'pval': row[f'{pred}_pval'],
							'n_obs': row.get('n_obs', np.nan)})
		return pd.DataFrame(out)
	
	stacked = pd.concat([
		stack_results([lme_abeta_df], 'ABeta'),
		stack_results([lme_tau_df], 'Tau'),
		stack_results([lme_volume_df], 'Volume')
	], ignore_index=True)

	stacked['pval'] = pd.to_numeric(stacked['pval'], errors='coerce')

	# Drop any NaNs or invalid p-values before FDR
	valid_mask = stacked['pval'].notnull() & np.isfinite(stacked['pval'])
	pvals = stacked.loc[valid_mask, 'pval'].values

	if len(pvals) > 0:
		rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
		stacked.loc[valid_mask, 'pval_fdr'] = pvals_fdr
		stacked.loc[valid_mask, 'signif_fdr'] = rej
	else:
		print("⚠️ No valid p-values for FDR correction.")
		stacked['pval_fdr'] = np.nan
		stacked['signif_fdr'] = False
	

	def fill_render_array(pred_name):
		arr = np.zeros(len(region_cols), dtype=float)
		sig_rows = stacked[(stacked['predictor'] == pred_name) & (stacked['signif_fdr'] == True)]
		for _, r in sig_rows.iterrows():
			region = str(r['Region']).strip().lower()
			beta = r['beta']
			idx = next((i for i, lbl in enumerate(region_cols)
						if str(lbl).strip().lower() == region), None)
			if idx is not None:
				arr[idx] = beta
		return arr
	
	render_ABeta = fill_render_array('ABeta')
	render_Tau   = fill_render_array('Tau')
	render_Volume = fill_render_array('Volume')

	savemat(out_path + 'Render.mat', {'render_ABeta': np.asarray(render_ABeta), 'render_Tau': np.asarray(render_Tau), 'render_Volume': np.asarray(render_Volume)})
	
	
	# ================================================================
	# 7. Visualization of significant effects
	# ================================================================
	formula_str = f'{measure_name} ~ ABeta + Tau + Volume + age + gender + edu + APOE4_CARRIER'
	
	predictors = ['ABeta', 'Tau', 'Volume']

	groups_names = measure_df['Group'].unique()
	labels_scatter = {idx: grp for idx, grp in enumerate(groups_names)}
	colors = sns.color_palette("colorblind", n_colors=len(groups_names))

	legend_path = os.path.join(out_path, 'legend.png')
	if not os.path.exists(legend_path):
		os.makedirs(out_path, exist_ok=True)
		legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=16, alpha=0.7) for color in colors]
		plt.figure(figsize=(3, 3))
		new_labels = [str(lbl).replace('p', '+').replace('n', '-') for lbl in groups_names]
		plt.legend(legend_handles, new_labels, fontsize=28)
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(legend_path, dpi=300, transparent=True)
		plt.close('all')
	
	for pred in predictors:
		c=0
		sig_rows = stacked[(stacked['predictor'] == pred) & (stacked['signif_fdr'])]
	
		if pred == 'Volume' and any(df['Region'].str.contains('hippocampus', case=False, na=False)):
			
			groups_names = df['Group'].unique()
			labels_scatter = {idx: grp for idx, grp in enumerate(groups_names)}
			colors = sns.color_palette("colorblind", n_colors=len(groups_names))

			L_HIPP_df = df[df['Region'].isin(['left hippocampus '])].reset_index(drop=True)
			R_HIPP_df=df[df['Region'].isin(['right hippocampus '])].reset_index(drop=True)

			fig, ax = plt.subplots(figsize=(5, 5))

			groups = L_HIPP_df['Group']

			# build a stable mapping from group values to colors (handles string or numeric groups)
			groups_vals = list(dict.fromkeys(groups.dropna().tolist()))

			desired = groups_names
			unique_groups = [g for lbl in desired for g in groups_vals if labels_scatter.get(g,str(g))==lbl]
			unique_groups += [g for g in groups_vals if g not in unique_groups]
			group_to_color = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
			for g in unique_groups:
				mask = groups == g
				plt.scatter(L_HIPP_df['Volume'][mask], L_HIPP_df[measure_name][mask], label='_nolegend_', color=group_to_color.get(g), alpha=0.7, s=100, marker='o')
				plt.scatter(R_HIPP_df['Volume'][mask], R_HIPP_df[measure_name][mask], label='_nolegend_', color=group_to_color.get(g), alpha=0.7, s=100, marker='^')

			L_beta, L_pval =sig_rows[sig_rows['Region']=='left hippocampus '][['beta','pval_fdr']].values[0]
			R_beta, R_pval =sig_rows[sig_rows['Region']=='right hippocampus '][['beta','pval_fdr']].values[0]

			L_marker = significance_marker(L_pval)
			R_marker = significance_marker(R_pval)

			L_intercept = L_HIPP_df[measure_name].mean() - L_beta * L_HIPP_df['Volume'].mean()
			R_intercept = R_HIPP_df[measure_name].mean() - R_beta * R_HIPP_df['Volume'].mean()
			L_x_vals = np.linspace(L_HIPP_df['Volume'].min(), L_HIPP_df['Volume'].max(), 100)
			L_y_vals = L_intercept + L_beta * L_x_vals
			R_x_vals = np.linspace(R_HIPP_df['Volume'].min(), R_HIPP_df['Volume'].max(), 100)
			R_y_vals = R_intercept + R_beta * R_x_vals

			plt.plot(L_x_vals, L_y_vals, color='#800080', linewidth=5, label=rf'L-HIPP $\beta_V${L_marker}={L_beta:.3f}')
			plt.plot(R_x_vals, R_y_vals, color='#e63946', linewidth=5, label=rf'R-HIPP $\beta_V${R_marker}={R_beta:.3f}')
			legend_handles = [Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, alpha=0.7) for marker, color in zip(['o', '^'], ['#800080', '#e63946'])]
			plt.legend(legend_handles, [rf'L-HIPP $\beta_V${L_marker}={L_beta:.3f}', rf'R-HIPP $\beta_V${R_marker}={R_beta:.3f}'], fontsize=20)

			#plt.legend(fontsize=25, handlelength=0.5)
			plt.xticks(fontsize=25)
			plt.yticks(fontsize=25)
			ax.set_xlabel('GMV stand.', fontsize=22,fontname="serif")
			plt.ylabel('Trophic Level', fontsize=25)
			plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
			plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
			plt.tight_layout()
			plt.savefig(os.path.join(out_path, f'{pred}_BilateralHIPP_{measure_name}_APOE4.png'), dpi=300)
			plt.close('all')
		
		for _, row in sig_rows.iterrows():
			region = row['Region']
			beta = row['beta'] # slope from LME
			
			sub = df[df['Region'] == region].copy()

			#plt.figure(figsize=(5, 6))
			fig, ax = plt.subplots(figsize=(4, 5))
			# use the subset's Group column so masks align with 'sub'
			groups = sub['Group']
			
			# build a stable mapping from group values to colors (handles string or numeric groups)
			groups_vals = list(dict.fromkeys(groups.dropna().tolist()))
			desired = groups_names
			unique_groups = [g for lbl in desired for g in groups_vals if labels_scatter.get(g,str(g))==lbl]
			unique_groups += [g for g in groups_vals if g not in unique_groups]
			group_to_color = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
			
			for g in unique_groups:
				mask = groups == g
				plt.scatter(sub[pred][mask], sub[measure_name][mask], label='_nolegend_', color=group_to_color.get(g), alpha=0.7, s=100)


			min_tl = df[df['Region'].isin(sig_rows['Region'].values)][measure_name].min()
			max_tl = df[df['Region'].isin(sig_rows['Region'].values)][measure_name].max()
			plt.ylim([min_tl - 0.05*(max_tl - min_tl), max_tl + 0.05*(max_tl - min_tl)])

			# regression line from LME beta
			intercept = sub[measure_name].mean() - beta * sub[pred].mean()
			x_vals = np.linspace(sub[pred].min(), sub[pred].max(), 100)
			y_vals = intercept + beta * x_vals
			# prefer FDR-corrected p-value if present, otherwise fall back to raw p-value
			pval = row.get('pval_fdr', np.nan)
			if pd.isnull(pval):
				pval = row.get('pval', np.nan)
			if np.isnan(pval):
				stars = ''
			else:
				marker = significance_marker(pval)
				stars = '' if marker == 'n.s.' else marker
			if pred == 'ABeta':
				beta_label = r'$\beta_{A}$'
			elif pred == 'Tau':
				beta_label = r'$\beta_{T}$'
			elif pred == 'Volume':
				beta_label = r'$\beta_{V}$'
			else:
				beta_label = r'$\beta$'
			plt.plot(x_vals, y_vals, color='#800080', linewidth=5, label=rf'{beta_label}{stars}={beta:.3f}')
			
			plt.title(f'{region}', fontsize=30)
			if pred == 'ABeta':
				x_name = 'Aβ-PET stand.'
			elif pred == 'Tau':
				x_name = 'Tau-PET stand.'
			elif pred == 'Volume':
				x_name = 'GMV stand.'
			ax.set_xlabel(f'{x_name}', fontsize=22,fontname="serif")
			#plt.xlabel(f'{pred} (standardized)', fontsize=30)
			plt.ylabel('Trophic Level', fontsize=25)
			plt.xticks(fontsize=25)
			plt.yticks(fontsize=25)
			plt.legend(fontsize=25, handlelength=0.5)
			plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
			plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

			# hide y-axis and label for second network plot
			
			if c != 0:
				fig.set_size_inches(3.5, 5)
				ax.yaxis.set_visible(False)
				ax.set_ylabel('')
				# remove limit lines (spines) at right, left and top
				for spine in ['right', 'top', 'left']:
					ax.spines[spine].set_visible(False)
				# remove ticks on left, right and top
				ax.tick_params(left=False, right=False, top=False)
			elif c==0:
				for spine in ['right', 'top']:
					ax.spines[spine].set_visible(False)
				# remove ticks on left, right and top
				ax.tick_params(right=False, top=False)
			c+=1
			plt.tight_layout()
			plt.savefig(os.path.join(out_path, f'{pred}_{region}_{measure_name}.png'))
			plt.close('all')
