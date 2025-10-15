from collections.abc import Callable, Sequence

from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder


# try import to rapids for faster pcas
RAPIDS_IMPORT_OKAY = True
try:
    import rapids_singlecell as rsc
except ImportError as e:
    RAPIDS_IMPORT_OKAY = False


def get_onehot_dict(
        categories: list[str] | np.ndarray
    ) -> dict[str, np.ndarray]:
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


def get_concatenated_transformed_obs_columns(
    adata: AnnData,
    column2tranform: dict[str, Callable | None],
    obsm_col="cond_concat",
):
    for column, transorm_fn in column2tranform.items():
        col_values = adata.obs[column].values[:, None]
        if isinstance(adata.obs[column].values.dtype, pd.CategoricalDtype):
            col_values = adata.obs[column].astype(float).values[:, None]
        if transorm_fn is not None:
            col_values = transorm_fn(col_values)
        adata.obsm[column] = col_values
    adata.obsm[obsm_col] = np.concatenate(
        [adata.obsm[col] for col in column2tranform.keys()], axis=-1
    )
    return adata


def annotate_perturbations(
    adata: AnnData,
    protocol_columns: Sequence[str],
    protocol_obs_key_added: str = "protocol_id",
    protocol_sep: str = "_",
    one_hot_uns_key_added: str = "one_hot",
    column2tranform: dict[str, Callable | None] = {},
    protocol_obsm_key="protocol_concat",
):

    # Perturbation data 0. write unique protocol conditions to obs
    if isinstance(protocol_columns, str):
        protocol_columns = [protocol_columns,]
    adata.obs[protocol_obs_key_added] = adata.obs[protocol_columns].astype(str).apply(lambda x: protocol_sep.join(x), axis=1)

    # Perturbation data 1. adding one hot encoded lookup dictionary for protocol id column
    adata.uns[one_hot_uns_key_added] = get_onehot_dict(adata.obs[protocol_obs_key_added].unique())

    # Perturbation data 2. adding one hot encoded lookup dictionary for protocol column
    for column in protocol_columns:
        key = f"{column}{protocol_sep}{one_hot_uns_key_added}"
        adata.uns[key] = get_onehot_dict(adata.obs[column].unique())
    
    # Perturbation data 3. adding protocol features to obsm
    for column in protocol_columns:
        # 3.1 retrieving column values and handling type
        col_values = adata.obs[column]
        if pd.api.types.is_categorical_dtype(col_values):
            col_values = adata.obs[column].astype(float).values[:, None]
        else:
            col_values = col_values.values[:, None]

        # 3.2 optionally applying transformations
        tranform_fn = column2tranform.get(column, None)
        col_values = col_values if tranform_fn is None else tranform_fn(col_values)

        # 3.3 store transformed medium condition data back in obsm
        adata.obsm[column] = col_values

    # Perturbation data 4. adding concatenating protocol features
    adata.obsm[protocol_obsm_key] = np.concatenate(
        [adata.obsm[col] for col in protocol_columns], axis=-1
    )
    return adata


def annotate_cell_state_data(
    adata: AnnData,
    scatter_columns: Sequence[str],
    base_sample_rep: str | None = None,
    scatter_obsm_key: str = "X_scatter",
    concat_obsm_key: str = "X_joint",
):
    # Cell State Data 0. writing scatter features to obsm
    adata.obsm[scatter_obsm_key] = adata.obs[scatter_columns].values

    # retrieving the two representations
    X_repr = adata.X if base_sample_rep is None else adata.obsm[base_sample_rep]
    X_scatter = adata.obsm[scatter_obsm_key]

    # concatenating them and updating anndata
    adata.obsm[concat_obsm_key] = np.concatenate((X_repr, X_scatter), axis=1)
    # Cell State Data 1. concatenating scatter features to base representation
    return adata


def apply_shared_transformations(
    train_adata: AnnData,
    ood_adata_dict: dict[int, AnnData],
    standardize_channel_features: bool = True,
    channel_obsm_key: str = "X_channel",
    channel_params_uns_key: str = "channel_params",
    compute_channel_pcs: bool = True,
    pca_obsm_key: str = "X_pca",
    standardize_scatter_features: bool = True,
    scatter_obsm_key: str = "X_scatter",
    scatter_params_uns_key: str = "scatter_params",
):

    # channel standardization
    if standardize_channel_features:
        # retrieving channel features and computing params
        X_channel_train = train_adata.X
        channel_mean = X_channel_train.mean(0)
        channel_std = X_channel_train.std(0)

        # storing standardization params
        channel_params = {"mean": channel_mean, "std": channel_std}
        train_adata.uns[channel_params_uns_key] = channel_params # failing for some reason (seg fault)
        train_adata.obsm[channel_obsm_key] = (X_channel_train - channel_mean)/channel_std

        # iterating over ood adatas
        for id, ood_adata in ood_adata_dict.items():
            X_channel_ood = ood_adata.X
            ood_adata.uns[channel_params_uns_key] = channel_params
            ood_adata.obsm[channel_obsm_key] = (X_channel_ood - channel_mean)/channel_std
            ood_adata_dict[id] = ood_adata

    # channel pca
    if compute_channel_pcs:
        # computing pcs on train data
        if RAPIDS_IMPORT_OKAY:
            rsc.pp.pca(train_adata)
        else:
            sc.pp.pca(train_adata)

        # applying transformation to validation data
        for id, ood_adata in ood_adata_dict.items():
            ood_adata.obsm[pca_obsm_key] = np.einsum("...d,dk -> ...k", ood_adata.X, train_adata.varm["PCs"])
            ood_adata_dict[id] = ood_adata

    # scatter standardization
    if standardize_scatter_features:
        # retrieving scatter features
        X_scatter_train = train_adata.obsm[scatter_obsm_key]

        # computing standardization params
        scatter_mean = X_scatter_train.mean(0)
        scatter_std = X_scatter_train.std(0)

        # storing standardization params
        scatter_params = {"mean": scatter_mean, "std": scatter_std}
        train_adata.uns[scatter_params_uns_key] = scatter_params

        # transforming data
        train_adata.obsm[scatter_obsm_key] = (X_scatter_train - scatter_mean)/scatter_std

        # iterating over ood adatas
        for id, ood_adata in ood_adata_dict.items():
            X_scatter_ood = ood_adata.obsm[scatter_obsm_key]
            ood_adata.uns[scatter_params_uns_key] = scatter_params
            ood_adata.obsm[scatter_obsm_key] = (X_scatter_ood - scatter_mean)/scatter_std
            ood_adata_dict[id] = ood_adata
    return train_adata, ood_adata_dict
