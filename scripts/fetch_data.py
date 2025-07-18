from collections.abc import Sequence
import logging
from typing import Literal

from anndata import AnnData
import numpy as np
from scanpy import read_h5ad

from condition_utils import get_one_hot_encoded_protocols, get_one_hot_encoded_protocol_axis, get_concatenated_scatter_features

logger = logging.getLogger(__name__)

EXPERIMENTAL_COLUMNS = [
    'mtg_[µm]', 'rhflt3l_[ng/ml]', 'gm-csf_[ng/ml]', 'tpo_[ng/ml]', 'sr1_[nm]',
    'um171_[nm]', 'um729_[µm]', 'scf_[ng/ml]', 'butyzamide_[nm]', 'retinoic_acid_[µm]',
    'ldl_[ng/ml]', 'il3_[ng/ml]', 'o2_[%]',
]
SCATTER_COLUMNS = ['FSC - Area', 'FSC - Height', 'FSC - Width', 'SSC - Area', 'SSC - Height', 'SSC - Width']


def fetch_adata_goattgens_sfc(
    h5ad_data_path: str,
    mode: Literal["unconditional", "protocol_one_hot", "protocol_axis_one_hot"] = "unconditional",
    transformation: Literal["arcsinh", "logabs"] = "arcsinh",
    cofactor: int = 100,
    protocol_columns: Sequence[str] = EXPERIMENTAL_COLUMNS,
    obsm_key_added="X_repr",
    obs_key_added="protocol",
    uns_key_added="protocol_one_hot",
    sep="_",
    concatenate_scatter_feats: bool = False,
    scatter_columns=SCATTER_COLUMNS,
    scatter_transformation=lambda x:x,
    scatter_key_added="X_sct"
) -> tuple[AnnData, AnnData | None]:
    """"""
    # reading h5ad file
    adata = read_h5ad(h5ad_data_path)

    if transformation == "arcsinh": 
        adata.obsm[obsm_key_added] = np.arcsinh(adata.X/cofactor)
    elif transformation == "logabs":
        adata.obsm[obsm_key_added] = np.sign(adata.X)*np.log(np.abs(adata.X / cofactor))
    else:
        msg = f"Transformation {transformation} is not supported, available options are [\"arcsinh\", \"logabs\"]"
        raise ValueError(msg)

    if concatenate_scatter_feats:
        adata = get_concatenated_scatter_features(
            adata,
            scatter_columns,
            scatter_transformation,
            obsm_key_added,
            scatter_key_added,
        )

    if mode == "unconditional":
        return adata
    elif mode == "protocol_one_hot":
        return get_one_hot_encoded_protocols(
            adata,
            protocol_columns=protocol_columns,
            obs_key_added=obs_key_added,
            uns_key_added=uns_key_added,
            sep=sep,
        )
    elif mode == "protocol_axis_one_hot":
        return get_one_hot_encoded_protocol_axis(
            adata,
            protocol_columns=protocol_columns,
            uns_key_added=uns_key_added,
            sep=sep,
        )
    else:
        msg = f"Mode {mode} is not supported, available options are [\"unconditional\"]"
        raise ValueError(msg)
