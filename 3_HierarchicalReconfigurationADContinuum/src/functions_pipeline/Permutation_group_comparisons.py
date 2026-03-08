import numpy as np
from numpy.linalg import lstsq
from statsmodels.stats import multitest

def fast_permutation_test(df, demographics, g1, g2, full_labels, n_perm):

	# ---- Prepare subject-wise table ----
	P = df[df['Group'].isin([g1, g2])].copy()
	P = P.merge(demographics[['ID','age','gender','edu']], on='ID', how='left')
	P['group_bin'] = P['Group'].map({g1: 0, g2: 1}).astype(float)

	# Extract arrays
	Y = P[full_labels].values                 # shape (N, R)   — all regions
	group = P['group_bin'].values[:, None]    # shape (N, 1)
	age   = P['age'].values[:, None]
	gen   = P['gender'].values[:, None]
	edu   = P['edu'].values[:, None]

	N, R = Y.shape

	# ---- Design matrices ----
	Xc = np.hstack([age, gen, edu, np.ones((N,1))])             # Covariates only
	Xf = np.hstack([group, age, gen, edu, np.ones((N,1))])      # Full model

	# ---- Reduced model fit ----
	Bc, _, _, _ = lstsq(Xc, Y, rcond=None)
	fitted_c = Xc @ Bc
	resid = Y - fitted_c     # shape (N, R)

	# ---- Full model observed effect ----
	Bf, _, _, _ = lstsq(Xf, Y, rcond=None)
	# group is column 0 in Xf
	se = np.sqrt(np.sum((Y - Xf@Bf)**2, axis=0) / (N - Xf.shape[1]))
	XtX_inv = np.linalg.inv(Xf.T @ Xf)
	t_obs = Bf[0,:] / (se * np.sqrt(XtX_inv[0,0]))

	# ---- Prepare permutations ----
	perms = np.array([np.random.permutation(N) for _ in range(n_perm)])

	# ---- Vectorized permutation Y* ----
	resid_perm = resid[perms]                       # (n_perm × N × R)
	Y_perm = fitted_c[None,:,:] + resid_perm            # broadcast

	# Solve (Xf^T Xf)^-1 Xf^T Y_perm via matrix multiply
	Xt = Xf.T                                           # (p × N)
	A = XtX_inv @ Xt                                    # (p × N)

	# permuted betas: B_perm = A @ Y_perm
	B_perm = np.einsum('pn,knr->pkr', A, Y_perm)        # (p × n_perm × R)
	B_group_perm = B_perm[0,:,:]                # (n_perm × R)

	pred_perm = np.einsum('np,pkr->knr', Xf, B_perm)

	# permuted residuals
	res_perm = Y_perm - pred_perm


	# permuted SE
	se_perm = np.sqrt(np.sum(res_perm**2, axis=1) / (N - Xf.shape[1]))

	# vectorized t-values
	t_perm = B_group_perm / (se_perm * np.sqrt(XtX_inv[0,0]))  # (n_perm × R)

	# ---- two-sided p-values ----
	pvals = (np.sum(np.abs(t_perm) >= np.abs(t_obs), axis=0) + 1) / (n_perm + 1)

	return t_obs, pvals


def run_comparisons(df, demographics, g1, g2):
	full_labels = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'ID']
	
	tvals, pvals = fast_permutation_test(df, demographics, g1, g2, full_labels, 10000)
	_, pvals_corr= np.array(multitest.fdrcorrection(pvals))
	corr_values = np.where(pvals_corr < 0.05, tvals, 0)
	return corr_values, pvals_corr
