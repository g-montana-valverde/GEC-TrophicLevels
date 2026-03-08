import matplotlib.pyplot as plt
import ptitprince as pt
from matplotlib.ticker import MaxNLocator
from scipy.io import savemat, loadmat
import os
import numpy as np
import seaborn as sns


def significance_marker(p):
	if p < 0.001: return '***'
	elif p < 0.01: return '**'
	elif p < 0.05: return '*'
	else: return 'n.s.'
    
def plot_directedness(df, pvals, out_path):
	df_copy = df.copy()
	df_copy['Group'] = df_copy['Group'].str.replace('p', '+').str.replace('n', '-', regex=False)

	groups=df_copy['Group'].unique()
	
	plt.figure(figsize=(8,7))
	pt.RainCloud(x='Group', y='directedness', data=df_copy, bw=0.3, dodge = True,
		palette = sns.color_palette("colorblind", n_colors=len(groups)), width_viol=0.5, width_box=0.3, box_showfliers=False, move = 0.0, point_size=7)

	plt.xticks(fontsize=40)
	plt.yticks(fontsize=38)
	plt.xlabel('Group', fontsize=32), plt.ylabel('');
	
	y_max = df_copy['directedness'].max()
	y_min = df_copy['directedness'].min()

	# generate all pairwise index combinations
	n = len(groups)
	pairs = []
	n = len(groups)
	for delta in range(1, n):
		for i in range(0, n - delta):
			pairs.append((i, i+delta))

	y_offset = (y_max-y_min)/17
	y = y_max + y_offset
	for idx, (i, j) in enumerate(pairs):
		if pvals[idx][0]<0.05:
			plt.plot([i+0.1, j-0.1], [y, y], color='black', lw=1.5)
			plt.text((i + j) / 2, y-y*0.03, significance_marker(pvals[idx][0]), 
					ha='center', va='bottom', fontsize=40)
			if idx >= n - 2:
				y += 2*y_offset  # increase height for next bar
	
	# Adjust y limits to fit all bars
	ymin, y_max = plt.ylim()
	plt.ylim(ymin, y_max+y_offset)


	plt.title('Directedness', fontsize=40)
	plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))

	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def save_mat_for_rendering(array, file, vmin, vmax, name, out_path):
	new_data = {name: array}
	if os.path.exists(out_path):
		mat = loadmat(out_path)
		mat.update(new_data)
		savemat(out_path, mat)
	else:
		savemat(out_path, new_data)

def plot_network_level_comparison(df, pvals, name, out_path):
	df_copy = df.copy()
	df_copy['Group'] = df_copy['Group'].str.replace('p', '+').str.replace('n', '-', regex=False)

	groups=df_copy['Group'].unique()

	net_names = ['VN', 'SMN', 'DAN', 'SAN', 'LN', 'CN', 'DMN']

	df_copy = df_copy.melt(
			id_vars=[col for col in df_copy.columns if col not in net_names],
			value_vars=net_names,
			var_name='Network',
			value_name=name
		)
	
	pvals = [pvals[net_names.index(net)] for net in net_names]

	plt.figure(figsize=(22,7))

	pt.RainCloud(x='Network', y=name, hue='Group', data=df_copy, bw=0.35, dodge = True,
		palette = sns.color_palette("colorblind", n_colors=len(groups)), width_viol=0.3, width_box=0.7, box_showfliers=False, move = 0.0, point_size=7, alpha = .5)
	
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=38)
	plt.xlabel('', fontsize=32), plt.ylabel(name, fontsize=38);

	plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=4))
	handles, labels = plt.gca().get_legend_handles_labels()
	plt.legend(handles=handles[:len(groups)], labels=labels[:len(groups)], loc='upper right', bbox_to_anchor=(1.01, 1.25), fontsize=35, title_fontsize=15, ncol=len(groups))
	
	# generate all pairwise index combinations
	pairs = []
	n = len(groups)
	for delta in range(1, n):
		for i in range(0, n - delta):
			pairs.append((i, i+delta))

	
	for idx in range(len(np.unique(df_copy['Network'].values))):
		y_max = df_copy[name].max()
		y_min = df_copy[name].min()
		y_offset = (y_max-y_min)/15
		y = y_max + y_offset
		for idx2, (i, j) in enumerate(pairs):
			if pvals[idx][idx2]<0.05:
				plt.plot([idx-(1.5-i)*0.1733+0.01, idx-(1.5-j)*0.1733-0.01], [y, y], color='black', lw=1.5)
				plt.text((idx-(1.5-i)*0.1733 + idx-(1.5-j)*0.1733) / 2, y-y*0.01, significance_marker(pvals[idx][idx2]), 
							ha='center', va='bottom', fontsize=27)
			if idx2 >= n - 2:
				y += 1.5*y_offset
			
	plt.xlim(-0.75, len(np.unique(df_copy['Network'].values)) - 0.5)
	ymin, y_max = plt.ylim()
	plt.ylim(ymin, y_max+y_offset)

	plt.tight_layout()
	plt.savefig(out_path)
	
	plt.close('all')
