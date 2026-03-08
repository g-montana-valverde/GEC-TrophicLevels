import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def mrmr_feature_selection_deterministic(X, y, k=15, random_state=0):
    """
    Minimum redundancy maximum relevance (mRMR) feature selection (greedy).
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if n_features <= k:
        return list(range(n_features))

    # Relevance: MI(feature, y)
    relevance = mutual_info_classif(X, y, random_state=random_state)

    selected = []
    not_selected = list(range(n_features))

    first = int(np.argmax(relevance))
    selected.append(first)
    not_selected.remove(first)

    for _ in range(k - 1):
        if not not_selected:
            break

        scores = []
        for j in not_selected:
            redundancy = 0.0
            if selected:
                X_candidate = X[:, j].reshape(-1, 1)
                redundancy = float(np.mean([
                    mutual_info_regression(
                        X_candidate, X[:, sel].ravel(), random_state=random_state
                    )[0]
                    for sel in selected
                ]))

            score = relevance[j] if redundancy < 1e-10 else relevance[j] / redundancy
            scores.append(score)

        scores = np.asarray(scores, dtype=float)

        # Minimal deterministic selection (np.argmax is deterministic given scores)
        next_feature = int(not_selected[int(np.argmax(scores))])

        selected.append(next_feature)
        not_selected.remove(next_feature)

    return selected
