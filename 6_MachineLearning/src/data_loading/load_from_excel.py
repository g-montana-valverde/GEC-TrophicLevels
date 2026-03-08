import numpy as np
import pandas as pd


def get_x_arr_for_scikit_from_excel(
    file_path, dict_groups_labels, id_key="ID", group_key="Group"
):
    if len(dict_groups_labels) != 2:
        raise ValueError(
            f"Skipping {file_path} as dict_group_labels passed does not contain only two groups"
        )

    df_data = pd.read_excel(file_path)

    if group_key not in df_data.columns:
        raise ValueError(
            f"ERROR loading data from {file_path} as it does not contain a {group_key} column"
        )

    # Filter + copy to avoid SettingWithCopy issues
    df_data_groups = df_data.loc[
        df_data[group_key].isin(dict_groups_labels.keys())
    ].copy()

    # Optional: normalize group strings (helps if Excel has trailing spaces)
    df_data_groups[group_key] = df_data_groups[group_key].astype(str).str.strip()

    # Fail fast if anything didn't map (prevents silent NaNs)
    unknown = set(df_data_groups[group_key].unique()) - set(dict_groups_labels.keys())
    if unknown:
        raise ValueError(f"Unknown group labels not in mapping: {unknown}")

    # Encode group labels and force numeric dtype
    df_data_groups.loc[:, group_key] = (
        df_data_groups[group_key].map(dict_groups_labels).astype(np.int64)
    )

    # Labels as guaranteed int64 numpy array
    y_arr = df_data_groups[group_key].to_numpy(dtype=np.int64)

    # Feature matrix
    feature_columns = [c for c in df_data_groups.columns if c not in [group_key, id_key]]
    x_arr = df_data_groups[feature_columns].to_numpy()

    return x_arr, y_arr, feature_columns
