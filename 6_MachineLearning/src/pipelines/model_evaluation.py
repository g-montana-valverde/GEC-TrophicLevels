import joblib
import numpy as np

from tqdm.auto import tqdm
from sklearn.utils import check_random_state
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..plotting.roc_curves import create_roc_curve
from ..plotting.confusion_matrix import plot_train_test_confusion_matrix


def predict_with_threshold(model, x, threshold):
    """Provides predicted classes using a custom threshold, typically obtained from
    Youden's J statistic from the ROC curve"""
    y_pred_proba = model.predict_proba(x)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return y_pred


def obtain_predict_threshold(model, x, y):
    """Computes optimal threshold from Youden's J statistic using ROC-curve FPR/TPR
    optimization"""
    y_proba = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    j_scores = tpr - fpr
    optimal_threshold = thresholds[j_scores.argmax()]
    return optimal_threshold


def evaluate_scores(model, x, y):
    """
    From a trained model, evaluates typical prediction scores/metrics by
    y_pred = model(x), and comparing to ground truth y

    NOTE: This version *optimizes the threshold on the same (x, y)* via Youden's J.
    That is fine for descriptive reporting, but should NOT be used for held-out test
    evaluation if you want to avoid label leakage.
    """
    prediction_threshold = obtain_predict_threshold(model, x, y)
    y_pred = predict_with_threshold(model, x, prediction_threshold)

    accuracy = accuracy_score(y, y_pred)
    bal_accuracy = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    specificity = recall_score(y, y_pred, pos_label=0, zero_division=0)

    roc_data = None
    try:
        y_proba = model.predict_proba(x)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc_score = auc(fpr, tpr)
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc_score}
    except Exception as e:
        print(f"Error calculating AUC: {str(e)}")
        auc_score = 0

    scores_dict = {
        "Accuracy": f"{accuracy * 100:.3f}",
        "Balanced Accuracy": f"{bal_accuracy * 100:.3f}",
        "Recall": f"{recall * 100:.3f}",
        "Specificity": f"{specificity * 100:.3f}",
        "Precision": f"{precision * 100:.3f}",
        "F1": f"{f1 * 100:.3f}",
        "AUC": f"{auc_score * 100:.3f}",
    }

    return scores_dict, roc_data


# ------------------------- NEW (minimal duplication) ------------------------- #
def evaluate_scores_with_fixed_threshold(model, x, y, threshold):
    """
    Evaluate metrics on (x, y) using a pre-specified threshold (e.g., learned on train).
    Returns the same (scores_dict, roc_data) format as evaluate_scores.
    """
    y_pred = predict_with_threshold(model, x, threshold)

    accuracy = accuracy_score(y, y_pred)
    bal_accuracy = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    specificity = recall_score(y, y_pred, pos_label=0, zero_division=0)

    roc_data = None
    try:
        y_proba = model.predict_proba(x)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc_score = auc(fpr, tpr)
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc_score}
    except Exception as e:
        print(f"Error calculating AUC: {str(e)}")
        auc_score = 0

    scores_dict = {
        "Accuracy": f"{accuracy * 100:.3f}",
        "Balanced Accuracy": f"{bal_accuracy * 100:.3f}",
        "Recall": f"{recall * 100:.3f}",
        "Specificity": f"{specificity * 100:.3f}",
        "Precision": f"{precision * 100:.3f}",
        "F1": f"{f1 * 100:.3f}",
        "AUC": f"{auc_score * 100:.3f}",
    }

    return scores_dict, roc_data
# --------------------------------------------------------------------------- #


def compute_confusion_matrix(model, x, y):
    prediction_threshold = obtain_predict_threshold(model, x, y)
    y_pred = predict_with_threshold(model, x, prediction_threshold)
    conf_matrix = confusion_matrix(y, y_pred)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, None]
    return conf_matrix


# ------------------------- NEW (for leakage-free test CM) ------------------- #
def compute_confusion_matrix_with_threshold(model, x, y, threshold):
    y_pred = predict_with_threshold(model, x, threshold)
    conf_matrix = confusion_matrix(y, y_pred)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, None]
    return conf_matrix
# --------------------------------------------------------------------------- #


def custom_permutation_test(estimator, x, y, n_permutations=1000, random_state=None):
    """
    Performs a permutation test to evaluate whether the target prediction metric
    (accuracy in this current implementation), is significantly different to chance
    through permutation over the labels in y.
    """
    random_state = check_random_state(random_state)

    if not check_is_fitted(estimator):
        estimator.fit(x, y)

    def scoring(est, x_for_scor, y_for_scor):
        return balanced_accuracy_score(y_for_scor, est.predict(x_for_scor))

    base_score = scoring(estimator, x, y)

    def _permutation_score(estimator, X, y):
        y_permuted = random_state.permutation(y)
        return scoring(estimator, X, y_permuted)

    permutation_scores = []
    with tqdm(
        total=n_permutations,
        desc="Permutation Test",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for _ in range(n_permutations):
            score = _permutation_score(estimator, x, y)
            permutation_scores.append(score)
            pbar.update(1)

    permutation_scores = np.array(permutation_scores)
    p_value = (np.sum(permutation_scores >= base_score) + 1.0) / (n_permutations + 1)

    return base_score, permutation_scores, p_value


def evaluate_model(
    final_model,
    x_train,
    y_train,
    x_test,
    y_test,
    outputs_folder=None,
    random_state=2,
    group_labels=(0, 1),
    standardize_data=False,
):
    """
    Evaluates a final model (not yet fit on train data) on the train/test split.

    Refactor: Test-set threshold-dependent metrics use a Youden threshold computed
    on TRAINING data only (no test-label leakage).
    """
    if standardize_data:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Train final model with best parameters on full training set
    final_model.fit(x_train, y_train)

    # Train evaluation (kept as-is; threshold optimized on train set)
    scores_train, train_roc_data = evaluate_scores(final_model, x_train, y_train)

    # ------------------- NEW: training-derived Youden threshold ------------------- #
    thr_train = obtain_predict_threshold(final_model, x_train, y_train)

    # Test evaluation using fixed threshold from TRAIN (no leakage)
    scores_test, test_roc_data = evaluate_scores_with_fixed_threshold(
        final_model, x_test, y_test, thr_train
    )
    # ----------------------------------------------------------------------------- #

    # Store ROC results and plots if outputs_folder is provided
    if outputs_folder:
        create_roc_curve(test_roc_data, outputs_folder, dataset_type="Test")
        create_roc_curve(train_roc_data, outputs_folder, dataset_type="Training")

        joblib.dump(test_roc_data, outputs_folder / "test_roc_data.joblib")
        joblib.dump(train_roc_data, outputs_folder / "train_roc_data.joblib")

    # Confusion matrices
    # Train CM (kept as-is; train-optimized threshold)
    cm_train = compute_confusion_matrix(final_model, x_train, y_train)
    np.savetxt(outputs_folder / "train_cm.txt", cm_train)

    # ------------------- NEW: test CM with train threshold ------------------- #
    cm_test = compute_confusion_matrix_with_threshold(final_model, x_test, y_test, thr_train)
    np.savetxt(outputs_folder / "test_cm.txt", cm_test)
    # ------------------------------------------------------------------------ #

    fig = plot_train_test_confusion_matrix(cm_train, cm_test, group_labels)
    fig.savefig(outputs_folder / "conf_matrices.png", dpi=300, bbox_inches="tight")

    # Permutation test (unchanged; uses estimator.predict default threshold)
    _, _, test_acc_p_value = custom_permutation_test(
        final_model, x_test, y_test, n_permutations=1000, random_state=random_state
    )

    return (
        final_model,
        scores_train,
        scores_test,
        train_roc_data,
        test_roc_data,
        test_acc_p_value,
    )
