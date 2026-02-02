import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_onehot_dict(categories: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """
    Creates a dictionary mapping each category to its corresponding one-hot encoded vector.

    Parameters:
    categories (list[str] | np.ndarray): List or array of categorical values to be one-hot encoded.

    Returns:
    dict[str, np.ndarray]: Dictionary where keys are categories and values are one-hot encoded numpy arrays.
    """
    dataset_enc = OneHotEncoder()
    dataset_enc.fit(np.array(categories).reshape(-1, 1))  # Fit encoder on categories
    onehot_dict = {}
    
    for cat in categories:
        # Transform each category into a one-hot vector
        dataset_onehot = (
            dataset_enc.transform(np.array([cat]).reshape(-1, 1)).toarray().flatten()
        )
        onehot_dict[cat] = dataset_onehot
    
    return onehot_dict


def get_one_hot_encoded_protocols(
    adata,
    protocol_columns,
    obs_key_added="protocol",
    uns_key_added="protocol_one_hot",
    sep="_"
):
    # adding column in adata.obs
    adata.obs[obs_key_added] = adata.obs[protocol_columns].astype(str).apply(lambda x: sep.join(x), axis=1)

    # getting one hot conditions
    adata.uns[uns_key_added] = get_onehot_dict(adata.obs[obs_key_added].unique())
    return adata


def get_one_hot_encoded_protocol_axis(
    adata,
    protocol_columns,
    uns_key_added="one_hot",
    sep="_"
):
    # iterating over the columns to one hot encode
    for column in protocol_columns:
        key = f"{column}{sep}{uns_key_added}"
        adata.uns[key] = get_onehot_dict(adata.obs[column].unique())
    return adata


def get_concatenated_scatter_features(
    adata,
    scatter_columns,
    scatter_transformation,
    obsm_key_to_concatenate_with,
    key_added,
):
    scatter_df = adata.obs[scatter_columns]
    X_scatter = scatter_transformation(scatter_df.values)
    X_rep = adata.obsm[obsm_key_to_concatenate_with]
    adata.obsm[key_added] = np.concatenate((X_rep, X_scatter), axis=-1)
    return adata


def get_cofactor_array_from_dict(
    channel2cofactor,
    adata,
):
    cofactors = np.zeros((1, adata.X.shape[1]))
    for channel, cofactor in channel2cofactor.items():
        idx = adata.var_names.tolist().index(channel)
        cofactors[0, idx] = cofactor
    return cofactor


def get_mixed_protocol_axis(
    adata,
    protocol_columns,
    uns_key_added="one_hot",
    obsm_cond_key_added="val",
    sep="_"
):
    raise NotImplementedError
