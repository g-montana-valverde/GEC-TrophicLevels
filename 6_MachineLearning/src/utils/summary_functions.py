"""
Module contains Summary functions

"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from src.plotting.roc_curves import plot_avg_roc_curve
from src.plotting.confusion_matrix import plot_train_test_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def summary_avg_roc_curve(outputs_folder, k):
    """
    After having trained multiple folds, we take the roc data computed in each
    fold and obtain a summary roc curve
    """
    dataset_labels = {"train": "Train", "test": "Test"}
    # We load all the computed roc_data
    for data in ["train", "test"]:
        roc_data = []
        for fold in range(k):
            fold_path = outputs_folder / f"fold_{fold}"
            roc_data.append(joblib.load(fold_path / f"{data}_roc_data.joblib"))

        # Once the ROC data is loaded, then we concatenate in lists the tpr, fpr, aucs
        fprs = [roc_data[i]["fpr"] for i in range(k)]
        tprs = [roc_data[i]["tpr"] for i in range(k)]
        aucs = [roc_data[i]["auc"] for i in range(k)]

        # And we pass the results to the create_roc_avg function
        plot_avg_roc_curve(fprs, tprs, aucs, outputs_folder, dataset_labels[data])


def pvalue_summary(pvalues):
    """
    Given an iterable of p‑values, compute and return:
      • percentage of p‑values < 0.05
      • median p‑value
      • mean p‑value
      • inter‑quartile range (IQR)
    Returns
    -------
    dict
        {
            'pct_lt_0.05': float,  # percentage in (0–100]
            'median':      float,
            'mean':        float,
            'iqr':         float   # Q3 − Q1
        }
    Notes
    -----
    - NaNs are ignored.
    - An empty list returns None for every statistic.
    """
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]  # drop NaNs

    if p.size == 0:
        return {'pct_lt_0.05': None, 'median': None, 'mean': None, 'iqr': None}

    pct_lt_005 = (p < 0.05).mean() * 100         # percentage < 0.05
    median     = float(np.median(p))
    mean       = float(p.mean())
    q1, q3     = np.percentile(p, [25, 75])
    iqr        = f"{q3:.3f} - {q1:.3f}"

    return {
        'pct_lt_0.05': pct_lt_005,
        'median':      median,
        'mean':        mean,
        'iqr':         iqr
    }


def summary_model_parameters(outputs_folder, k):
    """
    Function that, for N trained models, computes summary of the model parameters.

    Currently, only implemented for LogReg, for the Beta coefficients.

    :return:
    """
    parameters = []
    for fold in range(k):
        fold_path = outputs_folder / f"fold_{fold}"
        fit_model = joblib.load(fold_path / "fit_model.joblib")
        if isinstance(fit_model, LogisticRegression):
            # For logistic regression we will store the coef_ attribute
            parameters.append(np.squeeze(fit_model.coef_))
        elif isinstance(fit_model, SVC):
            print("No implementation for summary of model_parameters for SVC")
        elif isinstance(fit_model, KNeighborsClassifier):
            print("No implementation for summary of model_parameters for KNNClassifier")
        else:
            TypeError("Loaded fit_model.joblib model type is not yet implemented")

    # Save the dictionary with the parameters (Will need to adapt if other are
    # implemented)

    # Check if all parameters are equal length
    if len(set([len(parameter) for parameter in parameters])) == 1:
        mean_model_params = np.mean(np.array(parameters), axis=0).tolist()
    else:
        print("Different sizes of parameters, cannot compute mean")
        mean_model_params = []

    return mean_model_params

def summary_feature_importance(outputs_folder, k):
    """
    Function that, for N trained models, computes the average of feature importances
    for the best selected features for each model.

    For each iteration, loads the feature importance values. If for some iteration a
    feature that has appeared previously does not appear, a 0 is assigned to the
    feature. This way, whether a feature appears or not also influences feature
    importance.

    :return:
    """
    importances = []
    # Load all the importances
    for fold in range(k):
        fold_path = outputs_folder / f"fold_{fold}"
        with open(fold_path / "summary_results.json", "r") as f:
            results = json.load(f)
            imp_list = results["Importance"]  # In [["feature", value], ... ]
            imp_dict = {imp_list[i][0]: imp_list[i][1] for i in range(len(imp_list))}
            importances.append(imp_dict)

    df_imp = pd.DataFrame(importances)  # rows = runs, cols = *all* features
    df_imp = df_imp.fillna(0)           # implicit 0 for “not selected”

    # Now compute summary stats of the features
    mean_abs = df_imp.mean()  # unconditional mean |SHAP|
    std_abs = df_imp.std(ddof=1)  # unconditional SD
    sel_freq = (df_imp > 0).mean()  # fraction of runs selected
    mean_cond = df_imp.replace(0, pd.NA).mean()  # mean |SHAP| *given selected*
    summary = (
        pd.concat([mean_abs, std_abs, sel_freq, mean_cond], axis=1)
        .reset_index()
        .rename(columns={
            "index": "feature",
            0: "mean_abs_SHAP",
            1: "sd",
            2: "selection_freq",
            3: "mean_abs_SHAP_if_selected"
        })
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(outputs_folder / "summary_importance.csv", index=False)


def summary_confusion_matrix(outputs_folder, k, config_params):
    avg_cm = {}
    for data in ["train", "test"]:
        cms = []
        for fold in range(k):
            cms.append(np.loadtxt(outputs_folder / f"fold_{fold}" / f"{data}_cm.txt"))
        avg_cm[data] = np.mean(np.array(cms), axis=0)  # Avg over folds

    fig = plot_train_test_confusion_matrix(
        avg_cm["train"], avg_cm["test"], list(config_params["GROUPS"].keys())
    )
    fig.savefig(outputs_folder / "cm_avg_folds.png", dpi=300, bbox_inches="tight")

p_val_key = "Test Bal Accuracy Permutation p-value"


def create_summary_dictionary_of_results_over_iterations(
        scores, datasets, outputs_folder, k, config_params
):
    """
    Creates a dictionary with an individual key for each relevant score or metric
    evaluated from the N iterations/folds. Each key contains an N list with the
    values for that metric for each iteration/fold.
    """
    hyperparams = list(config_params["GRID_SEARCH"]["HYPERPARAM_SWEEP"].keys())
    parameters_per_fold = [f"Best {hyperparams[i]}" for i in range(len(hyperparams))]
    parameters_per_fold.append("Best Score")
    parameters_per_fold.append("Best Selected Features")

    results_list = []
    # First add all the results to list to create a dictionary and store on .json file
    for fold in range(k):
        fold_folder = outputs_folder / f"fold_{fold}"
        with open(fold_folder / "summary_results.json", "r") as file:
            results = json.load(file)
        results_list.append(results)

    # Add to the summary_results for each metric
    summary_results = {}
    for score in scores:
        for dataset in datasets:
            metric = f"{score} ({dataset})"
            summary_results[metric] = []
            for fold in range(k):
                summary_results[metric].append(results_list[fold][metric])

    for parameter in parameters_per_fold:
        summary_results[parameter] = [
            results_list[fold][parameter] for fold in range(k)
        ]

    # We also add the p-value from the permutation tests
    summary_results[p_val_key] = [results_list[fold][p_val_key] for fold in range(k)]

    # For the important features, we just store the names instead of name + importance
    summary_results["Top Important Features"] = [
        [
            results_list[fold]["Importance"][i][0]
            for i in range(len(results_list[fold]["Importance"]))
        ]
        for fold in range(k)
    ]

    # Save the aggregated results to a json file
    with open(outputs_folder / "summary_results_folds.json", "w") as file:
        json.dump(summary_results, file, indent=4)

    return summary_results


def create_stats_summary_dict_of_results_over_iterations(
        summary_dict_results, scores, datasets, outputs_folder, k
):
    """
    Creates a dictionary with formatted metrics as keys and values, ready for
    reporting statistics over the K iterations/folds. Requires to have run the
    create_summary_dictionary_of_results_over_iterations function beforehand to
    generate the summary_dict_results.

    Additionally calls other summary functions to report well formatted stats.
    """
    stats_summary = {}
    # Then perform stats
    # For scores, obtain mean +- std
    for score in scores:
        for dataset in datasets:
            metric = f"{score} ({dataset})"
            metric_folds = [float(value) for value in summary_dict_results[metric]]
            stats_summary[metric] = (
                f"{np.mean(metric_folds):.3f} +- {np.std(metric_folds):.3f}"
            )
    # For importance, count number of appearances
    all_imp_features = []  # First we concatenate all in a single list
    for fold in range(k):
        all_imp_features += summary_dict_results["Top Important Features"][fold]
    stats_summary["Importance Frequency"] = dict(Counter(all_imp_features))

    # For p-values, additionally obtain stats
    stats_summary["Permutation Test Stats"] = pvalue_summary(
        summary_dict_results[p_val_key]
    )

    stats_summary["Mean Model Parameters"] = summary_model_parameters(outputs_folder, k)

    # Also store the mean and standard deviation of the number of features
    nf_list = summary_dict_results["Best NF"]
    stats_summary["Best NF"] = f"{np.mean(nf_list):.2f} +- {np.std(nf_list):.2f}"

    # Finally, store this result as well
    with open(outputs_folder / "stats_results_folds.json", "w") as file:
        json.dump(stats_summary, file, indent=4)


def parse_metric(metric_str):
    """Parses a metric string like '77.770 +- 5.632' into a tuple (mean, std)."""
    try:
        mean, std = metric_str.split(" +- ")
        return float(mean), float(std)
    except:
        return None, None


def collect_selected_metrics(root_dir, metric_names, results_file_name):
    """
    Collects selected metrics from all classifiers, scenarios, and comparisons.

    Parameters:
        root_dir (str): Root directory path where classifier folders are located.
        metric_names (list): List of metric names to extract.

    Returns:
        pd.DataFrame: Structured DataFrame with extracted metrics.
    """
    results = []

    # Traverse the directory structure
    for clf_name in os.listdir(root_dir):
        clf_path = os.path.join(root_dir, clf_name)
        if not os.path.isdir(clf_path):
            continue

        for group_comparison in os.listdir(clf_path):
            comparison_path = os.path.join(clf_path, group_comparison)
            if not os.path.isdir(comparison_path):
                continue

            for feature_set in os.listdir(comparison_path):
                feature_path = os.path.join(comparison_path, feature_set)
                if not os.path.isdir(feature_path):
                    continue

                results_file = os.path.join(feature_path, results_file_name)
                if not os.path.exists(results_file):
                    continue

                with open(results_file, 'r') as f:
                    data = json.load(f)

                row = {
                    "Classifier": clf_name,
                    "Group Comparison": group_comparison,
                    "Feature Set": feature_set
                }

                # Extract all specified metrics
                for metric in metric_names:
                    mean, std = parse_metric(data.get(metric, "nan +- nan"))
                    row[f"{metric} Mean"] = mean
                    row[f"{metric} Std"] = std
                # Extract permutation test stats
                perm_test_key = "Permutation Test Stats"
                for key in data.get(perm_test_key, []):
                    row[f"perm test p-val {key}"] = data[perm_test_key][key]
                results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Group Comparison", "Feature Set", "Classifier"])
    return df

