"""
Plotting functions for SHAP Feature Importance Analyses
"""

import shap
import matplotlib.pyplot as plt


def handle_shap_outputs_for_plotting(shap_values):
    # Handle different SHAP output formats for plotting
    if isinstance(shap_values, list) and len(shap_values) > 1:
        plot_values = shap_values[1]  # Use positive class for binary classification
    else:
        plot_values = shap_values
    return plot_values


def plot_shap_summary_plot(x, shap_values, feature_names):
    # Save summary plot
    fig = plt.figure(figsize=(12, 8))

    plot_values = handle_shap_outputs_for_plotting(shap_values)

    shap.summary_plot(plot_values, x, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Impact", fontsize=16)
    plt.tight_layout()
    return fig


def plot_shap_bar_plot(x, shap_values, feature_names):
    # Get Bar Plot
    fig = plt.figure(figsize=(12, 8))
    plot_values = handle_shap_outputs_for_plotting(shap_values)
    shap.summary_plot(
        plot_values,
        x,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance", fontsize=16)
    plt.tight_layout()
    return fig
