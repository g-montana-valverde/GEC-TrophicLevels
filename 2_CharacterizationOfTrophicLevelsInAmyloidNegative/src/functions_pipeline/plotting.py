import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, zscore
import seaborn as sns


def plot_histogram(zTL, lower_threshold, upper_threshold, bins, out_path):
	
	bins = np.linspace(-max(abs(zTL)), max(abs(zTL)), 9)
	counts, bins, _ = plt.hist(zTL, bins=bins)
	
	print("hist bin edges:", bins)
	plt.xlabel('Trophic Level (z)', fontsize=30)
	plt.ylabel('Number of Regions', fontsize=30)
	plt.xticks(fontsize=29)
	plt.yticks(fontsize=29)
	plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
	
	plt.axvline(lower_threshold, color='red', linestyle='--', linewidth=7, label=f'{lower_threshold:.2f} THR')
	plt.axvline(upper_threshold, color='purple', linestyle='--', linewidth=7, label=f'{upper_threshold:.2f} THR')
	plt.legend(fontsize=25, handlelength=0.5)
	
	plt.text(0.19, 0.55, 'Sinks\n33.3%', transform=plt.gca().transAxes, fontsize=30, color='black', va='top', ha='center')
	plt.text(0.475, 0.55, 'M\n33.3%', transform=plt.gca().transAxes, fontsize=30, color='black', va='top', ha='center')
	plt.text(0.81, 0.55, 'Sources\n33.3%', transform=plt.gca().transAxes, fontsize=30, color='black', va='top', ha='center')
	
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def plot_correlation_with_mean(x, y, x_mean, y_mean, x_label, y_label, out_path):
	
	x_array=x.flatten()
	y_array=y.flatten()
	
	x_array = zscore(x_array)
	y_array = zscore(y_array)
	x_mean = zscore(x_mean)
	y_mean = zscore(y_mean)

	r, p = pearsonr(x_array,y_array)
	
	if p < 0.001:
		p_str = "***"
	elif p < 0.01:
		p_str = "**"
	elif p < 0.05:
		p_str = "*"
	else:
		p_str = "n.s."
	title = f"r{p_str} = {r:.2f}"
	plt.figure(figsize=(8.5, 6))
	plt.scatter(x_array, y_array, alpha=0.4, s=100, color="#229AC2")
	plt.scatter(x_mean, y_mean, c=y_mean, cmap='jet', s=200)
	
	# Fit line using seaborn regplot
	ax=sns.regplot(x=x_array, y=y_array, scatter=False, line_kws={'color':'red', 'linewidth':4})
	plt.xlabel(f"{x_label} (z)", fontsize=38)
	plt.ylabel(f"{y_label} (z)", fontsize=38)
	plt.title(title, fontsize=40)
	plt.xticks(fontsize=38)
	plt.yticks(fontsize=38)
	ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
	plt.axhline(0, color='gray', linestyle='--', linewidth=1)
	plt.axvline(0, color='gray', linestyle='--', linewidth=1)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	
