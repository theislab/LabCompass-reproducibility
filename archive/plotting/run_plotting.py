from itertools import combinations
import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import fastkde


PATH = "../../theis/HID01_fcs_concatenated.h5ad"
KEEP_COPY = True
SUBSAMPLE = True
SUBSAMPLING_DONE = False
SUBSAMPLE_SIZE = 5e-3
COFACTORS = [0.1, 1, 10, 50, 100, 300, 500, 1_000, 2_500, 5_000, 10_000]
COMPUTE_KERNEL = True
DUMP_PATH = "../../out"
SCATTER_COLUMNS = ['FSC - Area', 'FSC - Height', 'FSC - Width', 'SSC - Area', 'SSC - Height', 'SSC - Width']

def NxNPlot(
    data_id: str,
    data: np.ndarray,
    int2antibody: dict[str, int],
    compute_kernel: bool,
    dump_path: str,
):
    fig, axes = plt.subplots(data.shape[1], data.shape[1], figsize=(45, 45))
    fig.suptitle(data_id)
    for ridx in range(axes.shape[0]):
        for cidx in range(axes.shape[0]):
            if cidx == ridx:
                continue
            rantibody = int2antibody[ridx]
            cantibody = int2antibody[cidx]
            X, Y = data[:, ridx], data[:, cidx]
            values = np.vstack([X, Y])
            kernel = None
            if compute_kernel:
                PDF = fastkde.pdf(X, Y, var_names = [rantibody, cantibody])
                PDF.plot(ax=axes[ridx, cidx])
            sns.scatterplot(x=X,y=Y,ax=axes[ridx, cidx])
            axes[ridx, cidx].set_xlabel(rantibody)
            axes[ridx, cidx].set_ylabel(cantibody)
            axes[ridx, cidx].grid(True)
    fig.savefig(os.path.join(dump_path, f"{data_id}_NxNPlot.png"))


def density_plot(
    data_id: str,
    data: np.ndarray,
    int2antibody: dict[str, int],
    dump_path: str,
):
    
    fig, axes = plt.subplots(1, data.shape[1], figsize=(45, 5))
    fig.suptitle(data_id)
    for idx in range(data.shape[1]):
        antibody = int2antibody[idx]
        PDF = fastkde.pdf(data[idx], var_names = [antibody,])
        PDF.plot(ax=axes[idx])
        axes[idx].grid(True)
        axes[idx].set_title(antibody)
    fig.savefig(os.path.join(dump_path, f"{data_id}_DensityPlot_channels.png"))


def compute_summary_stats(data):
    output = {}
    stats ={
        "min": np.min, "max": np.max, "mean": np.mean, "median": np.median, "std": np.std
    }
    for idx in range(2):
        output[idx] = {
            stat_id: stat_fn(data, axis=idx).tolist() for stat_id, stat_fn in stats.items()
        }
    output[None] = {
        stat_id: stat_fn(data).tolist() for stat_id, stat_fn in stats.items()
    }
    return output


if __name__ == "__main__":

    # data
    adata = sc.read_h5ad(PATH)
    if SUBSAMPLE and not SUBSAMPLING_DONE:
        if KEEP_COPY:
            adata_original = adata.copy()
            sc.pp.subsample(adata, SUBSAMPLE_SIZE)
        SUBSAMPLING_DONE = True

    # mapping to antibody
    int2antibody = {
        idx: adata.var.iloc[idx][["antibody"]].values[0] for idx in range(adata.var.shape[0])
    }
    antibody2idx = {v:k for k, v in int2antibody.items()}

    # mapping to scatter features
    int2scatter = {
        idx:feat for idx, feat in enumerate(SCATTER_COLUMNS)
    }
    scatter2idx = {v:k for k, v in int2scatter.items()}

    # raw channel features
    key_raw = f"X_channel_raw"
    X_channel_raw = adata.X
    NxNPlot(
        key_raw,
        X_channel_raw,
        int2antibody,
        COMPUTE_KERNEL,
        DUMP_PATH,
    )
    density_plot(
        key_raw,
        X_channel_raw,
        int2antibody,
        DUMP_PATH,
    )

    # raw scatter features
    key_raw = f"X_scatter_raw"
    scatter_df = adata_original.obs[SCATTER_COLUMNS]
    X_scatter_raw = scatter_df.values
    NxNPlot(
        key_raw,
        X_scatter_raw,
        int2scatter,
        COMPUTE_KERNEL,
        DUMP_PATH,
    )
    density_plot(
        key_raw,
        X_scatter_raw,
        int2scatter,
        DUMP_PATH,
    )

    # applying transformations
    for cofactor in COFACTORS:
        print(f"Applying transformation for {cofactor=}")

        # channel arcnish
        key_arcsin = f"X_channel_arcsinh_cof{cofactor}"
        X_channel_arcsinh = np.arcsinh(X_channel_raw / cofactor)
        NxNPlot(
            key_arcsin,
            X_channel_arcsinh,
            int2antibody,
            COMPUTE_KERNEL,
            DUMP_PATH,
        )
        density_plot(
            key_arcsin,
            X_channel_arcsinh,
            int2antibody,
            DUMP_PATH,
        )
        X_channel_arcsinh_stats = compute_summary_stats(X_channel_arcsinh)

        # channel logabs
        key_logabs = f"X_channel_logabs_cof{cofactor}"
        X_channel_logabs = np.sign(X_channel_raw)*np.log(np.abs(X_channel_raw / cofactor))
        NxNPlot(
            key_logabs,
            X_channel_logabs,
            int2antibody,
            COMPUTE_KERNEL,
            DUMP_PATH,
        )
        density_plot(
            key_logabs,
            X_channel_logabs,
            int2antibody,
        )
        X_channel_logabs_stats = compute_summary_stats(X_channel_logabs)

        # scatter logabs
        key_log = f"X_scatter_log_cof{cofactor}"
        X_scatter_log = np.log(np.abs(X_scatter_raw / cofactor))
        NxNPlot(
            key_log,
            X_scatter_log,
            int2scatter,
            COMPUTE_KERNEL,
            DUMP_PATH,
        )
        density_plot(
            key_log,
            X_scatter_log,
            int2antibody,
        )
        X_scatter_log_stats = compute_summary_stats(X_scatter_log)
