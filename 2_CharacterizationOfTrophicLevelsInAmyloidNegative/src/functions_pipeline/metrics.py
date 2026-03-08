import numpy as np
import networkx as nx
import bct
from tqdm import tqdm
from scipy.io import savemat
from scipy.stats import zscore
import statsmodels.api as sm

def run_GEC_metrics(GEC_ALL):
	indegree_list, outdegree_list, out_in_degree_ratio_list, clust_list, mean_path_len_list, betweenness_list = [], [], [], [], [], []
	for sub in tqdm(range(GEC_ALL.shape[0])):
		GEC = GEC_ALL[sub, :, :]
		np.fill_diagonal(GEC, 0)
		GEC = GEC.T
		G = nx.from_numpy_array(GEC, create_using=nx.DiGraph())

		# Degrees
		indegree = np.array([G.in_degree(n, weight='weight') if G.is_directed() else G.degree(n, weight='weight') for n in G.nodes()])
		indegree_list.append(indegree)
		outdegree = np.array([G.out_degree(n, weight='weight') if G.is_directed() else G.degree(n, weight='weight') for n in G.nodes()])
		outdegree_list.append(outdegree)
		out_in_degree_ratio = np.divide(outdegree, indegree, out=np.zeros_like(outdegree), where=indegree!=0)
		out_in_degree_ratio_list.append(out_in_degree_ratio)

		# Clustering coefficient
		clust = bct.clustering_coef_wd(GEC)
		clust_list.append(clust)

		# Path length
		lengths = dict(nx.all_pairs_dijkstra_path_length(
			G, 
			weight=lambda u, v, d: 1/d['weight'] if d['weight'] > 0 else np.inf
		))
		mean_path_len = np.array([np.mean(list(lengths[i].values())) for i in range(len(GEC))])
		mean_path_len_list.append(mean_path_len)

		# Betweenness centrality (works for directed networks)
		betweenness_dict = nx.betweenness_centrality(G, weight='weight')
		betweenness_array = np.array([betweenness_dict[n] for n in G.nodes()])
		betweenness_list.append(betweenness_array)

	out_in_degree_ratio_array_GEC = np.array(out_in_degree_ratio_list)
	clust_array_GEC = np.array(clust_list)
	mean_path_len_array_GEC = np.array(mean_path_len_list)
	betweenness_array_GEC = np.array(betweenness_list)

	metrics = {
		'out_in_degree_ratio_array_GEC': out_in_degree_ratio_array_GEC,
		'clust_array_GEC': clust_array_GEC,
		'mean_path_len_array_GEC': mean_path_len_array_GEC,
		'betweenness_array_GEC': betweenness_array_GEC
	}
	return metrics

def regression_analysis(metrics, measure_array, metric_names):
	# Flatten metrics and hierarchical levels for regression
	X = np.column_stack([m.flatten() for m in metrics])
	y = zscore(measure_array.flatten())
	X = sm.add_constant(X)
	model = sm.OLS(y, X).fit()
	print("Predictors:", metric_names)
	print(model.summary())
