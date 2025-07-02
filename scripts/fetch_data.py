from collections.abc import Sequence
import logging
from typing import Literal

from anndata import AnnData
import numpy as np
from scanpy import read_h5ad

logger = logging.getLogger(__name__)


def fetch_adata_goattgens_sfc(
    h5ad_data_path: str,
    mode: Literal["unconditional"] = "unconditional",
    transformation: Literal["arcsinh", "logabs"] = "arcsin",
    cofactor: int = 100,
) -> tuple[AnnData, AnnData | None]:
    """"""
    # reading h5ad file
    adata = read_h5ad(h5ad_data_path)

    if transformation == "arcsinh":
        key_arcsin = f"X_arcsinh_cof{cofactor}" 
        adata.obsm[key_arcsin] = np.arcsinh(adata.X/cofactor)
    if transformation == "logabs":
        key_logabs = f"X_logabs_cof{cofactor}" 
        adata.obsm[key_logabs] = np.sign(adata.X)*np.log(np.abs(adata.X / cofactor))
    else:
        msg = f"Tranformation {transformation} is not supported, available options are [\"arcsinh\", \"logabs\"]"
        raise ValueError(msg)
    
    if mode == "unconditional":
        return adata
    else:
        msg = f"Mode {mode} is not supported, available options are [\"unconditional\"]"
        raise ValueError(msg)
