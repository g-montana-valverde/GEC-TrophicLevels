from neuroHarmonize import harmonizationLearn, harmonizationApply
import numpy as np

def harmonize_data(X, covariates_df):
	model_X, _ = harmonizationLearn(X, covariates_df)
	X_harmonized = harmonizationApply(X, covariates_df, model_X)
	if np.isnan(X_harmonized).any():
		print("Warning: NaNs in harmonized matrix.")
	return X_harmonized
