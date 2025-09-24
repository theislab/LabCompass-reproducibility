from collections.abc import Callable
from typing import Any, Literal
import sys

from anndata import AnnData

from fetch_data import fetch_adata_goattgens_sfc

def fetch_adata(
    dataset_name: Literal["goettgens_sfc",],
    h5ad_data_path: str,
    h5ad_data_kwargs: dict[str, Any],
    validation_split_fn: Callable[[AnnData], tuple[AnnData, AnnData]] | None = None,
) -> tuple[AnnData, AnnData | None]:
    """"""
    # morphogen data
    if dataset_name == "goettgens_sfc":
        train_adata = fetch_adata_goattgens_sfc(
            h5ad_data_path,
            **h5ad_data_kwargs,
        )
    else:
        msg = f"{dataset_name} is not available, possible options are: `[\"morphogen\"]`."
        raise ValueError(msg)

    # optional validation split
    validation_adata = None
    if validation_split_fn is not None:
        train_adata, validation_adata = validation_split_fn(train_adata)
    return train_adata, validation_adata