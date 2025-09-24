from typing import Literal

from sklearn.model_selection import (
    ShuffleSplit,
)
from anndata import AnnData

def split_train_validation_adata(
    adata: AnnData,
    mode: Literal["shuffle"] = "shuffle",
    validation_size: float = 0.3,
    n_splits: int = 5,
    split_idx_to_return: int = 1,
    random_state: int = 42,
) -> tuple[AnnData, AnnData]:
    """"""
    # simple shuffle split
    if mode == "shuffle":
        # defining the splitter
        splitter = ShuffleSplit(
            n_splits=n_splits,
            test_size=validation_size,
            random_state=random_state,
        )
        # splitting the anndata
        splits = splitter.split(adata)

        # iterating over the splits to retrieve the desired one
        for split_idx, (train_idxs, validation_idxs) in enumerate(splits):
            if split_idx == split_idx_to_return:
                break
    else:
        msg = f""
        raise ValueError(msg)

    # slicing the anndata
    train_adata = adata[train_idxs]
    validation_adata = adata[validation_idxs]
    return train_adata, validation_adata