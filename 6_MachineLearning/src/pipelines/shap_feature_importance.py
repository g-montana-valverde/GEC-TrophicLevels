
import os
import shap
import traceback

import numpy as np

from sklearn.inspection import permutation_importance
from ..plotting.plot_feature_importance import (
    plot_shap_summary_plot,
    plot_shap_bar_plot,
)


def calculate_feature_importance(model, x, y, feature_names, random_state=2, n_jobs=1):
    """Calculate feature importance using SHAP values."""

    # Create a dictionary to store feature importance
    importance_dict = {}

    try:
        # For KNN, we'll use a more direct approach with KernelExplainer
        # Create a custom prediction function that handles KNN output correctly
        def predict(x):
            # Ensure input is 2D
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

            # Get class probabilities
            probs = model.predict_proba(x)

            # For binary classification, we want the probability of positive class
            if probs.shape[1] == 2:
                return probs[:, 1]
            else:
                # For multi-class, return all probabilities
                return probs

        # Limit the number of background samples to avoid memory issues
        max_background = min(50, x.shape[0])

        # Create a background dataset that's representative (reference dataset used for feature masking)
        if x.shape[0] > max_background:
            # Deterministic (and class-balanced) background selection
            rng = np.random.RandomState(random_state)

            if y is not None and len(np.unique(y)) == 2:
                # stratified selection (binary classification)
                class0 = np.where(y == np.unique(y)[0])[0]
                class1 = np.where(y == np.unique(y)[1])[0]

                n0 = max_background // 2
                n1 = max_background - n0

                idx0 = rng.choice(class0, min(n0, len(class0)), replace=False)
                idx1 = rng.choice(class1, min(n1, len(class1)), replace=False)

                indices = np.concatenate([idx0, idx1])
            else:
                indices = rng.choice(x.shape[0], max_background, replace=False)

            background = x[indices]
        else:
            background = x

        # Initialize the explainer with our custom function
        explainer = shap.KernelExplainer(predict, background)

        # Limit the number of samples (subjects) for SHAP calculation
        max_samples = min(100, x.shape[0])
        # Deterministic subset to explain (avoid dependence on dataset ordering)
        rng_sample = np.random.RandomState(random_state + 1)

        if x.shape[0] > max_samples:
            sample_indices = rng_sample.choice(x.shape[0], max_samples, replace=False)
            x_sample = x[sample_indices]
        else:
            x_sample = x

        # Calculate SHAP values with reduced nsamples for efficiency (coalitions)
        np.random.seed(random_state + 2)
        shap_values = explainer.shap_values(x_sample, nsamples=50) #50 stable for DK80, 1000 stable for Sch400

        # Handle different output formats from SHAP
        if isinstance(shap_values, list):
            # If we get a list (possible for multi-class), use the positive class
            if len(shap_values) > 1:
                feature_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            # Otherwise, use the SHAP values directly
            feature_importance = np.abs(shap_values).mean(axis=0)

        # Create the importance dictionary
        for name, importance in zip(feature_names, feature_importance):
            importance_dict[name] = float(importance)

    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        traceback.print_exc()
        print("Falling back to permutation importance...")

        # Fallback to permutation importance
        try:
            result = permutation_importance(
                model, x, y, n_repeats=10, random_state=random_state, n_jobs=n_jobs
            )

            importances = result.importances_mean

            # Create importance dictionary
            for name, imp in zip(feature_names, importances):
                importance_dict[name] = float(imp)

        except Exception as permutation_error:
            print(f"Error in permutation importance: {str(permutation_error)}")
            traceback.print_exc()

            # If all else fails, just return empty importance values
            for name in feature_names:
                importance_dict[name] = 0.0

    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_features, shap_values


def pipeline_feature_importance(
    model, x, y, feature_names, output_folder, random_state=2, n_jobs=1
):
    """
    Pipeline to perform entire SHAPLEY's Feature Importance Analysis
    - Computes feature importance
    - Generates Shapley summary and bar plots
    - Stores the results onto a summary.txt file.

    Args:
        model: trained model (scikit-learn classifier)
        x: input-array of shape (n_samples, n_features)
        y: labels-array of shape (n_samples, )
        feature_names:
        output_folder: Folder (path) to store the results
        random_state: random state (int) to ensure reproducibility

    Returns:
        importance_dict: sorted dict (higher importance first), where keys are
        feature names and values are their importance.
    """
    # Compute Shapley Feature importance, obtain a dictionary of importance and
    # shapley values for the plots
    importance_dict, shap_values = calculate_feature_importance(
        model, x, y, feature_names, random_state=random_state
    )

    # Compute plots of feature importance:
    fig_shap = plot_shap_summary_plot(x, shap_values, feature_names)
    fig_shap.savefig(
        os.path.join(output_folder, "shap_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig_bar = plot_shap_bar_plot(x, shap_values, feature_names)
    fig_bar.savefig(
        os.path.join(output_folder, "shap_bar_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Finally, store the results to a .txt file
    file_feat_imp = os.path.join(output_folder, "feature_importance.txt")
    with open(file_feat_imp, "w") as f:
        f.write("Feature Importance Rankings (SHAP values):\n")
        f.write("----------------------------------------\n\n")
        for i, (feature, importance) in enumerate(importance_dict):
            f.write(f"{i + 1}. {feature}: {importance:.4f}\n")

    return importance_dict
