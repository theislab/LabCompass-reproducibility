from collections.abc import Sequence

from anndata import AnnData
import numpy as np
from sklearn.model_selection import ShuffleSplit


def split_adata(
    adata: AnnData,
    obs_column: str,
    unique_value_ids: Sequence[int],
):
    """"""


    if isinstance(unique_value_ids, int):
        unique_value_ids = [unique_value_ids, ]

    # retrieving unique values of column and performing sanity check
    unique_vals = adata.obs[obs_column].unique()
    for val_id in unique_value_ids:
        assert val_id < len(unique_vals)
    
    # store for ood indices and adatas
    odd_idxs_dict = {}
    odd_adatas_dict = {}

    # iterating over each unique value index to leave out
    for val_id in unique_value_ids:
        # retrieving corresponding value and computing mask
        ood_val = unique_vals[val_id]
        ood_idxs = adata.obs[obs_column].map(lambda e: e == ood_val).values
        # storing indices and sliced adata
        odd_idxs_dict[val_id] = ood_idxs
        odd_adatas_dict[val_id] = adata[ood_idxs].copy()

    # slicing the data
    is_ood = np.any(
        np.stack(list(odd_idxs_dict.values()), axis=0), axis=0
    )
    train_adata = adata[~is_ood].copy()
    return train_adata, odd_adatas_dict


def shuffle_split(adata, obs_col, K=3, test_size=0.3, random_state=42, split_to_retieve=0):
    skf = ShuffleSplit(n_splits=K, test_size=test_size, random_state=random_state)
    splits = skf.split(np.zeros(len(adata)), adata.obs[obs_col].values)
    for idx, (train_index, test_index) in enumerate(splits):
        if idx == split_to_retieve:
            break
    train_adata = adata[train_index]
    val_adata = adata[test_index]
    return train_adata, val_adata
