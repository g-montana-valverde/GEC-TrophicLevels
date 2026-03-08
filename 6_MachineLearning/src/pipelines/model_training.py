from tqdm.auto import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score

from ..feature_extraction.mrmr_feature_selection import mrmr_feature_selection_deterministic

scores = {
    "accuracy": accuracy_score,
    "AUC": auc,
    "bal_accuracy": balanced_accuracy_score,
}

classifiers = {
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
    "KNNClassifier": KNeighborsClassifier,
}


def pipeline_gridsearch_2d_with_loocv(x_train, y_train, feature_columns, config_params):
    """
    Performs a hyperparameter gridsearch for any of the following models:
    - Support Vector Classifier (SVC)
    - Logistic Regression
    - KNN Classifier

    Each hyperparameter combination is evaluated with a Leave One Out Cross
    Validation approach. That is, for each set of hyperparameters, a LOOCV algorithm is
    applied to obtain a robust estimation of the target score.

    The config_params argument is a dictionary that must contain at least the
    following keys:
    - "MODEL_TYPE": has to be either "SVC", "LogisticRegression", "KNNClassifier"
    - "HYPERPARAM_SWEEP": a dictionary with TWO keys, equal to the argument name of
        the hyperparameter that is evaluated in the scikit learn classifier arguemnts.
        For instance, for SVC, if we want to explore the C and kernel, keys must be:
        "C" and "kernel".
        The values of the dictionary will be an iterable with the values we want to
        explore in the gridsearch.
        NOTE: One hyperparameter can be "NF", which will allow to tune the number of
        features used, independently of the classifier.
    - "MODEL_KWARGS": a dictionary with additional parameters as kwargs for the model
    - "SCORE": what score to use to evaluate the model's performance: "accuracy",
        "AUC", "bal_accuracy".
    - "USE_MRMR_FEATURE_SELECTION": bool to choose whether MRMR feature selection is
        applied. If NF not in the hyperparameter list, provide additionaly a "NF"
        parameter.

    Args:
        x_train: input-array of shape (n_samples, n_features)
        y_train: labels-array of shape (n_samples, )
        feature_columns: list of str with the names of the feature columns
        config_params: dict containing configuration parameters.

    Returns:
    """

    gs_hyperparams = config_params["HYPERPARAM_SWEEP"]
    hyperparams = list(gs_hyperparams.keys())

    best_score = 0
    best_selected_features = []

    # If number of features is not explored as hyperparm -> Perform MRMR outside loop
    if "NF" not in hyperparams:
        # For fixed feature count, do the feature selection once outside the loop
        if config_params["USE_MRMR_FEATURE_SELECTION"]:
            # Only use training data for feature selection
            if config_params.get("NF", False):
                selected_indices = mrmr_feature_selection(
                    x_train, y_train, k=config_params["NF"]
                )
            else:
                print("'NF' parameter has not been passed for MRMR selection. NF=15")
                selected_indices = mrmr_feature_selection(x_train, y_train)
            x_train_selected = x_train[:, selected_indices]
            best_selected_features = [feature_columns[i] for i in selected_indices]

        else:  # If "NF" not explored and we do not use MRMR -> x_train contains all
            x_train_selected = x_train
            best_selected_features = feature_columns

    else:  # If "NF" explored, ensure that NF is the first to be explored
        hyp_not_nf = [hyp for hyp in hyperparams if hyp != "NF"][0]  # Just 2 params
        gs_hyperparams = {"NF": gs_hyperparams["NF"], hyp_not_nf: gs_hyperparams[hyp_not_nf]}
        hyperparams = list(gs_hyperparams.keys())

    # Create LOOCV object for nested cross-validation
    loo = LeaveOneOut()
    best_hyperparams = {}
    current_selected_features = best_selected_features
    # Grid search with nested LOOCV
    with tqdm(
        total=len(gs_hyperparams[hyperparams[0]]) * len(gs_hyperparams[hyperparams[1]]),
        desc="Grid Search",
        ncols=100,
    ) as pbar:
        for hyp0_val in gs_hyperparams[hyperparams[0]]:
            hyp0_key = hyperparams[0]

            # If we are exploring "NF" -> Select features with MRMR, else continue
            if hyp0_key == "NF":
                if config_params["USE_MRMR_FEATURE_SELECTION"]:
                    # Repeat MRMR with the current "NF" value
                    selected_indices = mrmr_feature_selection(x_train, y_train, k=hyp0_val)
                    current_selected_features = [
                        feature_columns[i] for i in selected_indices
                    ]
                    x_train_selected = x_train[:, selected_indices]
                else:  # If we are not using MRMR
                    # But still NF is hyperparam (which will not happen, edge case):
                    # Take the first NF (does not make a lot of sense anyway)
                    current_selected_features = feature_columns[:hyp0_val]
                    x_train_selected = x_train[:, :hyp0_val]

            for hyp1_val in gs_hyperparams[hyperparams[1]]:
                hyp1_key = hyperparams[1]

                # We generate a dictionary with kwargs
                if hyp0_key == "NF":  # The args model will not contain "NF"
                    args_model = {hyp1_key: hyp1_val}
                else:
                    args_model = {hyp0_key: hyp0_val, hyp1_key: hyp1_val}

                # We add the additional kwargs we can pass
                if config_params["MODEL_KWARGS"] is not None:
                    args_model = {**args_model, **config_params["MODEL_KWARGS"]}

                # We obtain the model from the available classifiers and initialize
                # with given args
                model = classifiers[config_params["MODEL_TYPE"]](**args_model)

                # Perform LOOCV within the training set
                y_true = []
                y_pred = []
                y_pred_proba = []

                for train_idx, val_idx in loo.split(x_train_selected):
                    x_train_loo, x_val_loo = (
                        x_train_selected[train_idx],
                        x_train_selected[val_idx],
                    )
                    y_train_loo, y_val_loo = y_train[train_idx], y_train[val_idx]

                    if config_params["STANDARDIZE_DATA"]:
                        scaler = StandardScaler()
                        x_train_loo = scaler.fit_transform(x_train_loo)
                        x_val_loo = scaler.transform(x_val_loo)

                    # Train model
                    model.fit(x_train_loo, y_train_loo)

                    # Predict probabilities
                    try:
                        y_pred.append(model.predict(x_val_loo)[0])
                        proba = model.predict_proba(x_val_loo)
                        y_pred_proba.append(proba[0][1])  # Probability of class 1
                        y_true.append(y_val_loo[0])
                    except IndexError:
                        # Handle case where all samples are of one class
                        continue

                # Calculate evaluation score
                if len(set(y_true)) > 1:
                    if config_params["SCORE"] == "AUC":
                        # First calculate ROC curve points
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                        # Then calculate AUC from the curve points
                        eval_score = auc(fpr, tpr)
                    else:
                        eval_score = scores[config_params["SCORE"]](y_true, y_pred)

                    # Update best parameters if score is higher
                    if eval_score > best_score:
                        best_hyperparams = {hyp0_key: hyp0_val, hyp1_key: hyp1_val}
                        best_score = eval_score
                        best_selected_features = current_selected_features

                pbar.update(1)
    return best_hyperparams, best_score, best_selected_features


def pipeline_gridsearch_3d_with_loocv(x_train, y_train, feature_columns, config_params):
    """
    Performs a hyperparameter gridsearch for any of the following models:
    - Support Vector Classifier (SVC)
    - Logistic Regression
    - KNN Classifier

    Each hyperparameter combination is evaluated with a Leave One Out Cross
    Validation approach. That is, for each set of hyperparameters, a LOOCV algorithm is
    applied to obtain a robust estimation of the target score.

    The config_params argument is a dictionary that must contain at leas the
    following keys:
    - "MODEL_TYPE": has to be either "SVC", "LogisticRegression", "KNNClassifier"
    - "HYPERPARAM_SWEEP": a dictionary with TWO keys, equal to the argument name of
        the hyperparameter that is evaluated in the scikit learn classifier arguemnts.
        For instance, for SVC, if we want to explore the C and kernel, keys must be:
        "C" and "kernel".
        The values of the dictionary will be an iterable with the values we want to
        explore in the gridsearch.
        NOTE: One hyperparameter can be "NF", which will allow to tune the number of
        features used, independently of the classifier.
    - "MODEL_KWARGS": a dictionary with additional parameters as kwargs for the model
    - "SCORE": what score to use to evaluate the model's performance: "accuracy",
        "AUC", "bal_accuracy".
    - "USE_MRMR_FEATURE_SELECTION": bool to choose whether MRMR feature selection is
        applied. If NF not in the hyperparameter list, provide additionaly a "NF"
        parameter.

    Args:
        x_train: input-array of shape (n_samples, n_features)
        y_train: labels-array of shape (n_samples, )
        feature_columns: list of str with the names of the feature columns
        config_params: dict containing configuration parameters.

    Returns:
    """

    gs_hyperparams = config_params["HYPERPARAM_SWEEP"]
    hyperparams = list(gs_hyperparams.keys())

    best_score = 0
    best_selected_features = []

    # In this case, NF will be always explored as one of the hyperparams.
    # So we ensure it's in the first position
    hyp_not_nf = {hyp: val for hyp, val in gs_hyperparams.items() if hyp != "NF"}
    gs_hyperparams = {"NF": gs_hyperparams["NF"], **hyp_not_nf}
    hyperparams = list(gs_hyperparams.keys())

    # Create LOOCV object for nested cross-validation
    loo = LeaveOneOut()
    best_hyperparams = {}

    total_values = 1
    for v in gs_hyperparams.values():
        total_values *= len(v)

    # Grid search with nested LOOCV
    with (tqdm(total=total_values, desc="Grid Search", ncols=100) as pbar):
        for nf_value in gs_hyperparams[hyperparams[0]]:

            if config_params["USE_MRMR_FEATURE_SELECTION"]:
                # Repeat MRMR with the current "NF" value
                selected_indices = mrmr_feature_selection_deterministic(x_train, y_train, k=nf_value)
                current_selected_features = [
                    feature_columns[i] for i in selected_indices
                ]
                x_train_selected = x_train[:, selected_indices]
            else:  # If we are not using MRMR
                # But still NF is hyperparam (which will not happen, edge case):
                # Take the first NF (does not make a lot of sense anyway)
                current_selected_features = feature_columns[:nf_value]
                x_train_selected = x_train[:, :nf_value]

            for hyp1_val in gs_hyperparams[hyperparams[1]]:
                hyp1_key = hyperparams[1]

                for hyp2_val in gs_hyperparams[hyperparams[2]]:
                    hyp2_key = hyperparams[2]

                    # We generate a dictionary with kwargs
                    args_model = {hyp1_key: hyp1_val, hyp2_key: hyp2_val}

                    # We add the additional kwargs we can pass
                    if config_params["MODEL_KWARGS"] is not None:
                        args_model = {**args_model, **config_params["MODEL_KWARGS"]}

                    # We obtain the model from the available classifiers and initialize
                    # with given args
                    model = classifiers[config_params["MODEL_TYPE"]](**args_model)

                    # Perform LOOCV within the training set
                    y_true = []
                    y_pred = []
                    y_pred_proba = []

                    for train_idx, val_idx in loo.split(x_train_selected):
                        x_train_loo, x_val_loo = (
                            x_train_selected[train_idx],
                            x_train_selected[val_idx],
                        )
                        y_train_loo, y_val_loo = y_train[train_idx], y_train[val_idx]

                        if config_params["STANDARDIZE_DATA"]:
                            scaler = StandardScaler()
                            x_train_loo = scaler.fit_transform(x_train_loo)
                            x_val_loo = scaler.transform(x_val_loo)

                        # Train model
                        model.fit(x_train_loo, y_train_loo)

                        # Predict probabilities
                        try:
                            y_pred.append(model.predict(x_val_loo)[0])
                            proba = model.predict_proba(x_val_loo)
                            y_pred_proba.append(proba[0][1])  # Probability of class 1
                            y_true.append(y_val_loo[0])
                        except IndexError:
                            # Handle case where all samples are of one class
                            continue

                    # Calculate evaluation score
                    if len(set(y_true)) > 1:
                        if config_params["SCORE"] == "AUC":
                            # First calculate ROC curve points
                            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                            # Then calculate AUC from the curve points
                            eval_score = auc(fpr, tpr)
                        else:
                            eval_score = scores[config_params["SCORE"]](y_true, y_pred)

                        # Update best parameters if score is higher
                        if eval_score > best_score:
                            best_hyperparams = {
                                "NF": nf_value, hyp1_key: hyp1_val, hyp2_key: hyp2_val
                            }
                            best_score = eval_score
                            best_selected_features = current_selected_features

                    pbar.update(1)
    return best_hyperparams, best_score, best_selected_features
