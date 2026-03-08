"""
Module contains Summary functions

"""

import json

from pathlib import Path

from .summary_functions import (
    summary_feature_importance,
    summary_avg_roc_curve,
    summary_confusion_matrix,
    create_summary_dictionary_of_results_over_iterations,
    create_stats_summary_dict_of_results_over_iterations
)


def save_results_summary(
    scores_train,
    scores_test,
    test_acc_pvalue,
    grid_search_results,
    importance_dict,
    best_selected_features,
    outputs_folder,
):
    grid_results_keys = list(grid_search_results.keys())  # Will be 2 results + 1 score
    # Generate Summary Data dictionary
    summary_data = {
        "Accuracy (Test)": scores_test["Accuracy"],
        "Balanced Accuracy (Test)": scores_test["Balanced Accuracy"],
        "Sensitivity (Test)": scores_test["Recall"],
        "Precision (Test)": scores_test["Precision"],
        "Specificity (Test)": scores_test["Specificity"],
        "AUC (Test)": scores_test["AUC"],
        "F1 (Test)": scores_test["F1"],
        "Accuracy (Train)": scores_train["Accuracy"],
        "Balanced Accuracy (Train)": scores_train["Balanced Accuracy"],
        "Sensitivity (Train)": scores_train["Recall"],
        "Precision (Train)": scores_train["Precision"],
        "Specificity (Train)": scores_train["Specificity"],
        "AUC (Train)": scores_train["AUC"],
        "F1 (Train)": scores_train["F1"],
        "Test Bal Accuracy Permutation p-value": test_acc_pvalue,
    }
    for i in range(len(grid_results_keys)):
        summary_data[f"Best {grid_results_keys[i]}"] = grid_search_results[grid_results_keys[i]]
    summary_data["Importance"] = importance_dict

    # Add best selected features
    summary_data["Best Selected Features"] = best_selected_features

    with open(Path(outputs_folder) / "summary_results.json", "w") as f:
        json.dump(summary_data, f, indent=4)


def save_statistics_gridsearch_loocv_pipeline_with_outer_kfold_loop(
    outputs_folder, k, config_params
):
    scores = [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Precision",
        "Specificity",
        "AUC",
        "F1",
    ]
    datasets = ["Train", "Test"]

    # Create a summary dictionary, each key contains K-element list with metric
    # result for each iteration/fold.
    summary_dict_results = create_summary_dictionary_of_results_over_iterations(
        scores, datasets, outputs_folder, k, config_params
    )

    # Create a summary dictionary with well-formatted statistics for reporting using
    # the previously obtained dictionary.
    create_stats_summary_dict_of_results_over_iterations(
        summary_dict_results, scores, datasets, outputs_folder, k
    )

    # Additionally, we compute an average of the train/test confusion matrices as well
    summary_confusion_matrix(outputs_folder, k, config_params)

    # We also compute the averaged ROC curves for the training and the test
    summary_avg_roc_curve(outputs_folder, k)

    # And we compute summary statistics of feature importances
    summary_feature_importance(outputs_folder, k)
