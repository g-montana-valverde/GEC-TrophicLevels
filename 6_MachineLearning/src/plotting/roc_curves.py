"""
Functions to plot ROC curves
"""

import os

import joblib
import traceback

import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve_on_ax(
    fpr,
    tpr,
    label,
    ax,
    tpr_std=None,
    roc_color="tab:orange",
    diag_color="0.5",
    lw=2.5,
    band_alpha=0.25,
):
    # --- main ROC curve ---
    ax.plot(fpr, tpr, label=label, color=roc_color, lw=lw)

    # --- std shading ---
    if tpr_std is not None:
        ax.fill_between(
            fpr,
            tpr - tpr_std,
            tpr + tpr_std,
            color=roc_color,
            alpha=band_alpha,
            linewidth=0
        )

    # --- chance diagonal (kept visually neutral) ---
    ax.plot([0, 1], [0, 1], color=diag_color, lw=2, linestyle="--")

    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )


def create_roc_curve(roc_data, output_folder, dataset_type=None, tpr_std=None):
    """
    Creates and saves an ROC curve visualization using pre-calculated ROC data.
    """
    if roc_data is None:
        print(f"No ROC data available for {dataset_type} set")
        return

    try:
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["auc"]
        if isinstance(roc_auc, str):  # For cases where roc_auc is already passed as str
            label = f"ROC curve (AUC = {roc_auc})"
        else:
            label = f"ROC curve (AUC = {roc_auc:.3f})"
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_roc_curve_on_ax(fpr, tpr, label, ax, tpr_std=tpr_std)

        # Add dataset type to title if provided
        if dataset_type:
            ax.set_title(
                f"Receiver Operating Characteristic (ROC) Curve - {dataset_type} Set"
            )
        else:
            ax.set_title("Receiver Operating Characteristic (ROC) Curve")

        ax.legend(loc="lower right")
        filename = f"roc_curve_{dataset_type.lower()}.png"
        fig.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"ROC curve saved as {filename} in {output_folder}")
    except Exception as e:
        print(f"Error creating ROC curve: {str(e)}")
        traceback.print_exc()



def plot_avg_roc_curve(list_fprs, list_tprs, list_aucs, output_folder, dataset_type=None):
    """
    :param list_fprs: List (n, for each roc) of arrays of fprs
    :param list_tprs:  List (n, for each roc) of arrays of tprs
    :param list_aucs:  List of auc scores

    """
    if (len(list_fprs) != len(list_tprs)) or (len(list_tprs) != len(list_aucs)) or (len(list_fprs) != len(list_aucs)):
        raise ValueError("All 'tprs', 'fprs', 'aucs' must have same length")

    mean_fprs = np.linspace(0, 1, 100)
    interpolated_tprs = []
    for fpr, tpr in zip(list_fprs, list_tprs):
        interpolated_tprs.append(np.interp(mean_fprs, fpr, tpr))

    mean_tprs = np.mean(np.array(interpolated_tprs), axis=0)
    std_tprs = np.std(np.array(interpolated_tprs), axis=0)
    roc_auc = f"{np.mean(list_aucs):.2f} +- {np.std(list_aucs):.2f}"

    # We add to dictionary to pass to the plotting function
    roc_data = {"fpr": mean_fprs, "tpr": mean_tprs, "auc": roc_auc, "std_tpr": std_tprs}
    create_roc_curve(roc_data, output_folder, dataset_type, tpr_std=std_tprs)

    # And we also store the results in the outputs_folder
    file_name = f"avg_roc_data_{dataset_type.lower()}" if dataset_type else "avg_roc_data"
    joblib.dump(roc_data, output_folder / f"{file_name}.joblib")
