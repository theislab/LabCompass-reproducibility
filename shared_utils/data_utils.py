from collections.abc import Callable, Sequence
import logging
import sys
from typing import Literal

from anndata import AnnData
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import scanpy as sc
from scipy.linalg import cholesky
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


# try import to rapids for faster pcas
RAPIDS_IMPORT_OKAY = True
try:
    import rapids_singlecell as rsc
except ImportError as e:
    RAPIDS_IMPORT_OKAY = False

sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
from ood_utils import shuffle_split, split_adata


LOG1P_EXP_COL = [
    "gm-csf_[ng_ml]",
    "tpo_[ng_ml]",
    "sr1_[nm]",
    "um171_[nm]",
    "um729_[µm]",
    "scf_[ng_ml]",
    "butyzamide_[nm]",
    "retinoic_acid_[µm]",
    "ldl_[ng_ml]",
    "il3_[ng_ml]",
    "o2_[%]",
    "days_of_culture"
]
LOG21P_EXP_COL = []


class safe_logger:
    def __init__(self, logger_orig):
        self.logger = logger_orig
    def info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)


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


def get_protocol_tranformations(
    experimental_covariates,
    log1p_exp_cols=None,
    log21p_exp_cols=None,
    inverse=False
):
    if log1p_exp_cols is None:
        log1p_exp_cols = LOG1P_EXP_COL
    if log21p_exp_cols is None:
        log21p_exp_cols = LOG21P_EXP_COL
    col2transf = {}
    for col in experimental_covariates:
        if col in log1p_exp_cols:
            if inverse:
                col2transf[col] = np.expm1
            else:
                col2transf[col] = np.log1p
        elif col in log21p_exp_cols:
            if inverse:
                col2transf[col] = lambda x: np.exp2(x) - 1
            else:
                col2transf[col] = lambda x: np.log2(x + 1)

        else:
            col2transf[col] = None
    return col2transf


def write_unique_protocols_representations(
    adata: AnnData,
    protocol_columns: Sequence[str] ,
    column2tranform: dict[str, Callable | None] = {},
    suffix: str = "mapped",
    key_added: str = "protocol",
    unique_val_sep: str = "@",
    columns_sep: str = "+",
    registry_key: str = "col_registry"
) -> AnnData:
    # prepare registries for unique values
    col_registry = {}
    col_inverse_registry = {}
    for col in protocol_columns:
        adata.obs[col] = adata.obs[col].astype(float)
        col_values = adata.obs[col].values
        unique_values_registry = {}
        protocol_vals = np.unique(col_values)
        for idx, unique_val in enumerate(protocol_vals):
            unique_values_registry[f"{col}{unique_val_sep}{idx + 1}"] = unique_val.item()
        uniqe_values_inverse_registry = {v:k for k, v in unique_values_registry.items()}
        col_registry[col] = unique_values_registry
        col_inverse_registry[col] = uniqe_values_inverse_registry
        adata.obs[f"{col}_{suffix}"] = adata.obs[col].map(uniqe_values_inverse_registry)
    adata.uns[registry_key] = col_registry
    adata.uns[f"{registry_key}_inverse"] = col_inverse_registry

    # prepare joint column
    adata.obs[f"{key_added}_{suffix}"] = tuple(map(lambda e: columns_sep.join(e), adata.obs[[f"{col}_{suffix}" for col in protocol_columns]].values))
    comb_values = np.unique(adata.obs[f"{key_added}_{suffix}"].values)
    comb_rep_dict = {}
    for comb_value in comb_values:
        comb_rep = np.zeros((len(protocol_columns)))
        combs = comb_value.split(columns_sep)
        for comb in combs:
            mol = comb.split(unique_val_sep)[0]
            mol_transform_fn = column2tranform.get(mol, None)
            mol_transform_fn = (lambda x: x) if mol_transform_fn is None else mol_transform_fn
            mol_registry = adata.uns[registry_key][mol]
            mol_value = mol_registry[comb]
            mol_idx = protocol_columns.index(mol)
            comb_rep[mol_idx] = mol_transform_fn(np.array([mol_value]))
        comb_rep_dict[comb_value] = comb_rep
    adata.uns[f"{key_added}_{suffix}_repr"] = comb_rep_dict
    return adata


def annotate_perturbations(
    adata: AnnData,
    protocol_columns: Sequence[str],
    protocol_obs_key_added: str = "protocol_id",
    protocol_sep: str = "_",
    one_hot_uns_key_added: str = "one_hot",
    column2tranform: dict[str, Callable | None] = {},
    protocol_obsm_key="protocol_concat",
    one_hot_reps=False,
    typecast_anyway=False,
    suffix: str = "mapped",
    key_added: str = "protocol",
    unique_val_sep: str = "@",
    columns_sep: str = "+",
    registry_key: str = "col_registry",
    logger_orig: logging.Logger | None = None,
    unique_reps=False
):

    # wrap for optional logger
    logger = safe_logger(logger_orig)

    # Perturbation data 0. write unique protocol conditions to obs
    if one_hot_reps:
        logger.info("Writing unique protocol conditions to obs.")
        if isinstance(protocol_columns, str):
            protocol_columns = [protocol_columns,]
        adata.obs[protocol_obs_key_added] = adata.obs[protocol_columns].astype(str).apply(lambda x: protocol_sep.join(x), axis=1)
    logger.info("Unique protocol condition added.")

    # Perturbation data 1. adding one hot encoded lookup dictionary for protocol id column
    logger.info("Adding one hot encoder lookup dictionary for unique protocol id.")
    if one_hot_reps:
        adata.uns[one_hot_uns_key_added] = get_onehot_dict(adata.obs[protocol_obs_key_added].unique())
    logger.info("Unique one hot protocol id read.")

    # Perturbation data 2. adding one hot encoded lookup dictionary for protocol column
    if one_hot_reps:
        logger.info("Writing the unique one hot representation for each condition column.")
        for column in tqdm(protocol_columns):
            key = f"{column}{protocol_sep}{one_hot_uns_key_added}"
            adata.uns[key] = get_onehot_dict(adata.obs[column].unique())
        logger.info("One hot representation written.")
    
    # Perturbation data 3. adding protocol features to obsm
    logger.info("Adding protocol features to obsm.")
    for column in tqdm(protocol_columns):
        # 3.1 retrieving column values and handling type
        col_values = adata.obs[column]
        if pd.api.types.is_categorical_dtype(col_values) or typecast_anyway:
            col_values = adata.obs[column].astype(float).values[:, None]
        else:
            col_values = col_values.values[:, None]

        # 3.2 optionally applying transformations
        tranform_fn = column2tranform.get(column, None)
        logger.info(f"{column}, {tranform_fn}, {col_values}, {col_values.dtype}")
        col_values = col_values if tranform_fn is None else tranform_fn(col_values)

        # 3.3 store transformed medium condition data back in obsm
        adata.obsm[column] = col_values
    logger.info("All protocol features added.")

    # Perturbation data 4. adding concatenated protocol features
    logger.info("Concatenating protocol features.")
    adata.obsm[protocol_obsm_key] = np.concatenate(
        [adata.obsm[col] for col in protocol_columns], axis=-1
    )
    logger.info("Concatenated protocol features ready.")

    # Perturbation data 5. unique protocol representations
    logger.info("Writing unique value representations.")
    if unique_reps:
        adata = write_unique_protocols_representations(
            adata,
            protocol_columns,
            column2tranform=column2tranform,
            suffix=suffix,
            key_added=key_added,
            unique_val_sep=unique_val_sep,
            columns_sep=columns_sep,
            registry_key=registry_key,
        )
    logger.info("Unique value representation ready. All condition data was annotated")
    return adata


def standardize_array_and_write_to_adata(
    adata,
    X,
    name: str,
    params: dict[str, np.ndarray] | None = None,
) -> AnnData:
    """"""
    if params is None:
        mean = X.mean(0)
        std = X.std(0)
        params = {"mean": mean, "std": std}
        adata.uns[f"{name}_params"] = params
    adata.obsm[name] = X
    adata.obsm[f"{name}_standardized"] = (X - params["mean"])/params["std"]
    return adata



def get_data_params(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # rowvar=False means columns are variables, rows are observations
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False) 
    return mean, cov


def compute_whitening_matrix(
    X: np.ndarray,
    epsilon: float = 1e-5,
    method: Literal["zca", "pca", "cholesky"] = "zca"
) -> np.ndarray:
    # get params
    mean, cov = get_data_params(X)

    # SVD is generally more numerically stable than np.linalg.eig
    U, S, VT = np.linalg.svd(cov)
    
    # Standard PCA whitening matrix: W_pca = S^{-1/2} * U^T
    W_pca = np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T)

    if method == "pca":
        iW_pca = np.linalg.inv(W_pca)
        return W_pca, iW_pca, mean
    
    elif method == "zca":
        W = np.dot(U, W_pca)
        iW = np.linalg.inv(W)
        return W, iW, mean
    
    elif method == "cholesky":
        L = cholesky(cov, lower=True)
        W = np.linalg.inv(L)
        return W, L, mean

    raise ValueError(f"Unknown method: {method}")


def whiten_array_and_write_to_adata(
    adata,
    X,
    name: str,
    params: dict[str, np.ndarray] | None = None,
    epsilon: float = 1e-5,
    method: Literal["zca", "pca", "cholesky"] = "zca",
) -> AnnData:
    if params is None:
        W, iW, mean = compute_whitening_matrix(X, epsilon=epsilon, method=method)
        params = {"mean": mean, "W": W, "iW": iW}
        adata.uns[f"{name}_whitened"] = params
    adata.obsm[name] = X
    adata.obsm[f"{name}_whitened"] = (X - params["mean"]) @ params["W"].T
    return adata


def apply_shared_transformations(
    train_adata: AnnData,
    ood_adata_dict: dict[int, AnnData] | None,
    scatter_columns: Sequence[str],
    compute_channel_pcs: bool = True,
    epsilon: float = 1e-5,
    logger_orig: logging.Logger | None = None
):

    # wrap for optional logger
    logger = safe_logger(logger_orig)

    # channel pca
    if compute_channel_pcs:
        logger.info("Computing PCs of channel features")
        # computing pcs on train data
        if RAPIDS_IMPORT_OKAY:
            rsc.pp.pca(train_adata, zero_center=False)
        else:
            sc.pp.pca(train_adata, zero_center=False)
        logger.info("PCs ready.")

        # applying transformation to validation data
        if ood_adata_dict is not None:
            logger.info("Projecting validation data with PCs.")
            for id, ood_adata in ood_adata_dict.items():
                ood_adata.obsm["X_pca"] = np.einsum("...d,dk -> ...k", ood_adata.X, train_adata.varm["PCs"])
                ood_adata_dict[id] = ood_adata
            logger.info("All validation data transformed.")


    ### Z SCORE NORMALIZATION
    # Cell State Data 0. writing and standardizing scatter features to obsm
    logger.info("Standardizing scatter features with Zscore .")
    X_scatter = train_adata.obs[scatter_columns].values
    train_adata = standardize_array_and_write_to_adata(train_adata, X_scatter, "X_scatter")
    logger.info("Scatter features standardized.")
    if ood_adata_dict is not None:
        logger.info("Applying standardization to validation data.")
        for id, ood_adata in ood_adata_dict.items():
            X_scatter = ood_adata.obs[scatter_columns].values
            ood_adata = standardize_array_and_write_to_adata(ood_adata, X_scatter, "X_scatter", params=train_adata.uns["X_scatter_params"])
            ood_adata_dict[id] = ood_adata    
        logger.info("All validation data transformed.")

    # Cell state Data 1. writing and standardizing channel features
    logger.info("Standardizing channel features with Zscore .")
    X_channel = train_adata.X
    train_adata = standardize_array_and_write_to_adata(train_adata, X_channel, "X_channel")
    logger.info("Channel features standardized.")
    if ood_adata_dict is not None:
        logger.info("Applying standardization to validation data.")
        for id, ood_adata in ood_adata_dict.items():
            X_scatter = ood_adata.X
            ood_adata = standardize_array_and_write_to_adata(ood_adata, X_scatter, "X_channel", params=train_adata.uns["X_channel_params"])
            ood_adata_dict[id] = ood_adata
        logger.info("All validation data transformed.")

    # Cell state Data 2. concatenate pairs
    logger.info("Creating joint representations.")
    morphology_obsm_keys = ["X_scatter", "X_scatter_standardized"]
    marker_expression_obsm_keys = ["X_channel", "X_channel_standardized", "X_pca"]
    for morph_key in morphology_obsm_keys:
        for mark_key in marker_expression_obsm_keys:
            train_adata.obsm[f"{mark_key}+{morph_key}"] = np.concatenate(
                (train_adata.obsm[mark_key], train_adata.obsm[morph_key]), axis=-1
            )
            if ood_adata_dict is not None:
                for id, ood_adata in ood_adata_dict.items():
                    ood_adata.obsm[f"{mark_key}+{morph_key}"] = np.concatenate(
                        (ood_adata.obsm[mark_key], ood_adata.obsm[morph_key]), axis=-1
                    )
    logger.info("All joint representations ready.")

    # ZCA Whitening the concatenated data
    logger.info("Standardizing joint features with ZCA whitening .")
    X_train_concat = train_adata.obsm["X_channel+X_scatter"]
    train_adata = whiten_array_and_write_to_adata(
        train_adata,
        X_train_concat,
        "X_channel+X_scatter_zca",
        epsilon=epsilon,
        method="zca",
    )
    logger.info("Joint features standardized.")
    zca_whitening_params = train_adata.uns["X_channel+X_scatter_zca_whitened"]
    if ood_adata_dict is not None:
        logger.info("Whitening ood data.")
        for id, ood_adata in ood_adata_dict.items():
            X_ood_concat = ood_adata.obsm["X_channel+X_scatter"]
            ood_adata_dict[id] = whiten_array_and_write_to_adata(
                ood_adata,
                X_ood_concat,
                "X_channel+X_scatter_zca",
                params=zca_whitening_params,
                epsilon=epsilon,
                method="zca",
            )
        logger.info("All ood data whitened.")
    # PCA Whitening the concatenated data
    logger.info("Standardizing joint features with PCA whitening .")
    X_train_concat = train_adata.obsm["X_channel+X_scatter"]
    train_adata = whiten_array_and_write_to_adata(
        train_adata,
        X_train_concat,
        "X_channel+X_scatter_pca",
        epsilon=epsilon,
        method="pca",
    )
    logger.info("Joint features standardized.")
    pca_whitening_params = train_adata.uns["X_channel+X_scatter_pca_whitened"]
    if ood_adata_dict is not None:
        logger.info("Whitening ood data.")
        for id, ood_adata in ood_adata_dict.items():
            X_ood_concat = ood_adata.obsm["X_channel+X_scatter"]
            ood_adata_dict[id] = whiten_array_and_write_to_adata(
                ood_adata,
                X_ood_concat,
                "X_channel+X_scatter_pca",
                params=pca_whitening_params,
                epsilon=epsilon,
                method="pca",
            )
        logger.info("All ood data whitened.")
    # Cholesky Whitening the concatenated data
    logger.info("Standardizing joint features with Cholensky whitening .")
    X_train_concat = train_adata.obsm["X_channel+X_scatter"]
    train_adata = whiten_array_and_write_to_adata(
        train_adata,
        X_train_concat,
        "X_channel+X_scatter_cholesky",
        epsilon=epsilon,
        method="cholesky",
    )
    logger.info("Joint features standardized.")
    cholesky_whitening_params = train_adata.uns["X_channel+X_scatter_cholesky_whitened"]
    if ood_adata_dict is not None:
        logger.info("Whitening ood data.")
        for id, ood_adata in ood_adata_dict.items():
            X_ood_concat = ood_adata.obsm["X_channel+X_scatter"]
            ood_adata_dict[id] = whiten_array_and_write_to_adata(
                ood_adata,
                X_ood_concat,
                "X_channel+X_scatter_cholesky",
                params=cholesky_whitening_params,
                epsilon=epsilon,
                method="cholesky",
            )
        logger.info("All ood data whitened.")
    return train_adata, ood_adata_dict


def get_adata_splits(
    config: DictConfig,
    epsilon: float = 1e-5,
    logger_orig: logging.Logger | None = None,
):

    # wrap for optional logger
    logger = safe_logger(logger_orig)

    # Data 0. read data
    logger.info("Reading data...")
    adata = sc.read_h5ad(config.paths.h5ad_path)
    logger.info(f"Data read! {adata}")

    # Data 1. annotate perturbation data
    logger.info("Annotating perturbation data...")
    column2tranform = get_protocol_tranformations(
        config.annotation.protocol_columns,
        log1p_exp_cols=config.annotation.log1p_exp_cols,
        log21p_exp_cols=config.annotation.log21p_exp_cols,
    )
    adata = annotate_perturbations(
        adata,
        config.annotation.protocol_columns,
        protocol_obs_key_added=config.annotation.protocol_obs_key_added,
        protocol_sep=config.annotation.protocol_sep,
        one_hot_uns_key_added=config.annotation.one_hot_uns_key_added,
        column2tranform=column2tranform,
        protocol_obsm_key=config.annotation.protocol_obsm_key,
        logger_orig=logger
    )
    logger.info(f"Perturbation data annotated! {adata}")

    # Data 2. split data
    if config.split.mode == "ood":
        logger.info(f"Splitting data...\n\tPerforming validation split over column {config.split.obs_column} with unique value {config.split.unique_value_ids}")
        train_adata, ood_adatas_dict = split_adata(
            adata,
            config.split.obs_column,
            config.split.unique_value_ids,
        )
        logger.info(f"Data split performed!\n \tTrain data of shape {train_adata.shape}")
        for k, v in ood_adatas_dict.items():
            logger.info(f"\tValidation split {k} of shape {v.shape}")
    elif config.split.mode == "in-distribution":
        train_adata, ood_adatas_dict = shuffle_split(
            adata,
            K=config.split.K,
            test_size=config.split.test_size,
            random_state=config.split.random_state,
            split_to_retrieve=config.split.split_to_retrieve,
        )
    else:
        ValueError(f"Invalid split mode {config.split.mode}")

    # Data 3. apply shared transformations
    logger.info("Computing tranformation params on train data and applying to both train and ood data...")
    train_adata, ood_adatas_dict = apply_shared_transformations(
        train_adata,
        ood_adatas_dict,
        config.transforms.scatter_columns,
        compute_channel_pcs=config.transforms.compute_channel_pcs,
        epsilon=config.transforms.epsilon,
        logger_orig=logger,
    )
    logger.info("Shared tranformations applied!")
    return train_adata, ood_adatas_dict


def get_condition_data_from_file(
    data_manager,
    condition_metadata_path,
    protocol_columns,
    log1p_exp_cols,
    log21p_exp_cols,
    protocol_obs_key_added,
    protocol_sep,
    one_hot_uns_key_added,
    protocol_obsm_key,
    experiment_id="LPHO012",
    experiment_col="experiment_id",
):

    # prepare data and annotate perturbation data
    condition_df = pd.read_excel(condition_metadata_path)
    condition_df.columns = [col.replace("/", "_") for col in condition_df.columns]
    condition_df = condition_df.loc[
        condition_df[experiment_col] == experiment_id, protocol_columns
    ]
    condition_adata = AnnData(
        X=np.empty((len(condition_df), 2)),
        obs=condition_df
    )
    column2tranform = get_protocol_tranformations(
        protocol_columns,
        log1p_exp_cols=log1p_exp_cols,
        log21p_exp_cols=log21p_exp_cols
    )
    print(column2tranform)
    annotate_perturbations(
        condition_adata,
        protocol_columns,
        protocol_obs_key_added,
        protocol_sep=protocol_sep,
        one_hot_uns_key_added=one_hot_uns_key_added,
        column2tranform=column2tranform,
        protocol_obsm_key=protocol_obsm_key,
        one_hot_reps=False,
        typecast_anyway=True
    )
    return data_manager.perturbation_data_schema.get_data(condition_adata), condition_adata


def drop_duplicates(adata, logger=None):
    duplicates_mask = adata.obs_names.duplicated(keep='first')
    n_dups = duplicates_mask.sum()
    if n_dups > 0:
        if logger is not None:
            logger.info(f"Found {n_dups} duplicate observation names. Dropping...")
        # 2. Subset to keep only the first occurrence
        adata = adata[~duplicates_mask].copy()
        if logger is not None:
            logger.info(f"New shape: {adata.n_obs} rows")
    else:
        if logger is not None:
            logger.info("No duplicate observation names found.")
    return adata


def ensure_type_safety(adata):
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            adata.obs[col] = adata.obs[col].astype(str)
    return adata


def transform_validation_data(
    scatter_columns,
    train_adata_phi,
    target_adata,
    logger=None,
):
    if logger is not None:
        logger.info("="*80)
        logger.info("Standardizing Scatter Features.")
    X_scatter = target_adata.obs[scatter_columns].values
    target_adata = standardize_array_and_write_to_adata(
        target_adata, X_scatter, "X_scatter", params=train_adata_phi.uns["X_scatter_params"]
    )
    if logger is not None:
        logger.info(f"{target_adata=}")
    if logger is not None:
        logger.info("="*80)
        logger.info("Standardizing Channel Features.")
    X_channel = target_adata.X
    target_adata = standardize_array_and_write_to_adata(
        target_adata,
        X_channel,
        "X_channel",
        params=train_adata_phi.uns["X_channel_params"])
    if logger is not None:
        logger.info(f"{target_adata=}")

    if logger is not None:    
        logger.info("Creating joint representations.")
    morphology_obsm_keys = ["X_scatter", "X_scatter_standardized"]
    marker_expression_obsm_keys = ["X_channel", "X_channel_standardized"]
    if logger is not None:    
        logger.info("="*80)
    for morph_key in morphology_obsm_keys:
        for mark_key in marker_expression_obsm_keys:
                target_adata.obsm[f"{mark_key}+{morph_key}"] = np.concatenate(
                    (target_adata.obsm[mark_key], target_adata.obsm[morph_key]), axis=-1
                )
    if logger is not None:    
        logger.info(f"{target_adata=}")
    return target_adata


def get_feature_mask(
    adata,
    target_marker_names,
    target_morph_feat_names=None,
    scatter_columns=None,
):
    target_morph_feat_names = [] if target_morph_feat_names is None else target_morph_feat_names
    scatter_columns = [] if scatter_columns is None else scatter_columns
    target_feat_names = target_marker_names + target_morph_feat_names
    all_feat_names = adata.var_names.to_list() + scatter_columns
    return [name in target_feat_names for name in all_feat_names]


def get_target_marker_features(
    adata_ref,
    target_marker_names,
    agg_fn,
    agg_fn_kwargs={"axis": 0, "keepdims": True}
):
    X_target = np.zeros(adata_ref.shape)
    target_values = adata_ref[:,  target_marker_names].X

    for idx, marker_name in enumerate(target_marker_names):
        marker_val = target_values[:, idx]
        marker_idx = adata_ref.var_names.to_list().index(marker_name)
        X_target[:, marker_idx] = marker_val
    return agg_fn(X_target, **agg_fn_kwargs)


def get_target_scatter_features(
    adata_ref,
    scatter_columns,
    target_scatter_names,
    agg_fn,
    agg_fn_kwargs={"axis": 0, "keepdims": True}
):
    target_values = adata_ref.obs[scatter_columns].values
    X_target = np.zeros_like(target_values)

    for idx, scatter_name in enumerate(target_scatter_names):
        scatter_val = target_values[:, idx]
        scatter_idx = scatter_columns.index(scatter_name)
        X_target[:, scatter_idx] = scatter_val
    return agg_fn(X_target, **agg_fn_kwargs)


def get_target_group(
    adata,
    filter_dict,
    target_feat_names,
    target_quantile,
):
    # prepare mask for columns
    # and filter adata
    mask = np.full(adata.shape[0], True)
    for col, val in filter_dict.items():
        col_mask = adata.obs[col] == val
        mask = mask & col_mask
    adata_group = adata[mask]

    # filter by target quantile on target features
    X_target = adata_group[:, target_feat_names].X
    if hasattr(X_target, 'toarray'):
        X_target = X_target.toarray()

    # Keep cells with all markers above percentile
    thresholds = np.percentile(X_target, target_quantile, axis=0)
    keep_mask = np.all(X_target >= thresholds, axis=1)
    return adata_group[keep_mask]


def aggregate_target_adata_marker_opt(
    ref_adata,
    target_marker_names,
    target_morph_feat_names,
    scatter_columns,
    agg_fn,
    agg_fn_kwargs=None
):
    # prepare keyword arguments for aggregation
    agg_fn_kwargs = {"axis": 0, "keepdims": True} if agg_fn_kwargs is None else agg_fn_kwargs

    # prepare target channel features 
    X_channel_target = get_target_marker_features(
        ref_adata,
        target_marker_names,
        agg_fn,
        agg_fn_kwargs=agg_fn_kwargs
    )

    # prepare target marker features 
    X_scatter_target = get_target_scatter_features(
        ref_adata,
        scatter_columns,
        target_morph_feat_names,
        agg_fn,
        agg_fn_kwargs=agg_fn_kwargs
    )

    # Build obs dictionary correctly
    n_obs = X_scatter_target.shape[0]
    obs_dict = {}
    for idx, feat in enumerate(scatter_columns):
        values = X_scatter_target[..., idx]
        obs_dict[feat] = values

    # Create AnnData
    return sc.AnnData(
        X=X_channel_target,
        obs=obs_dict,
        var=pd.DataFrame(index=ref_adata.var_names)  # ensure correct length
    )


def get_target_adata_marker_opt(
    adata_ref,
    adata_query,
    agg_fn,
    target_marker_names,
    target_morph_feat_names,
    scatter_columns,
    filter_dict,
    target_quantile,
    agg_fn_kwargs=None,
    logger=None,
):

    # prepare target feature identifiers
    target_feat_names = target_marker_names + target_morph_feat_names
    target_feats_mask = get_feature_mask(
        adata_query,
        target_marker_names,
        scatter_columns=scatter_columns
    )

    # get target group
    target_adata = get_target_group(
        adata_query,
        filter_dict,
        target_feat_names,
        target_quantile,
    )
    target_adata.uns["target_feats_mask"] = target_feats_mask

    # aggregate target group
    target_adata = aggregate_target_adata_marker_opt(
        target_adata,
        target_marker_names,
        target_morph_feat_names,
        scatter_columns,
        agg_fn,
        agg_fn_kwargs=agg_fn_kwargs
    )

    # apply share transformations with reference data
    return transform_validation_data(
        scatter_columns,
        adata_ref,
        target_adata,
        logger=logger,
    )
