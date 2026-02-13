from collections import defaultdict
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

from sc_exp_design.metrics import compute_e_distance


def generated_density_knn(X_true, X_generated, k=20, bandwidth=None):

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_true)
    distances, indices = nbrs.kneighbors(X_generated)
    
    # Auto-bandwidth: median distance of neighbors
    if bandwidth is None:
        bandwidth = np.median(distances)
    density = np.zeros(X_true.shape[0])

    # Gaussian kernel weights
    weights = np.exp(-(distances**2) / (2 * bandwidth**2))

    # Add weighted contributions
    for gen_idx in range(len(X_generated)):
        for neighbor_pos, real_idx in enumerate(indices[gen_idx]):
            density[real_idx] += weights[gen_idx, neighbor_pos]
    return density


def get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, min_loss_idx, compute_stuff=True, n_scatter_feats=6):
    X_gen = fwd_results["X"][min_loss_idx]
    ct_label_gen = ct_le.inverse_transform(
        fwd_results["ct_probs"][min_loss_idx].argmax(1)
    )

    X = np.concat((X_gen, X_true), axis=0)
    X_channel = X[:, :-n_scatter_feats]
    X_scatter = X[:, -n_scatter_feats:]
    G = np.concatenate((ct_label_gen, adata_g.obs["cell_type"].values), axis=0)
    ct_adata_gen = sc.AnnData(
        X=X_channel,
        obsm={"X_scatter": X_scatter},
        obs={
            "cell_type": G,
            "data_type": ["gen"]*X_gen.shape[0] + \
                ["real"]*len(adata_g)
        },
        var=pd.DataFrame(index=adata_g.var_names)
    )
    if compute_stuff:
        sc.pp.pca(ct_adata_gen)
        sc.pp.neighbors(ct_adata_gen)
        sc.tl.umap(ct_adata_gen)
    return ct_adata_gen


def plot_embedding(ct_adata, target_ct, base="X_umap"):
    X_umap = ct_adata.obsm[base]
    ct_mask = (ct_adata.obs["cell_type"] == target_ct) & (ct_adata.obs["data_type"] != "gen")
    gen_mask = ct_adata.obs["data_type"] == "gen"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    fig.suptitle(target_ct)
    axes[0].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        s=5,
    )
    axes[0].scatter(
        X_umap[ct_mask, 0],
        X_umap[ct_mask, 1],
        s=40,
    )
    axes[0].set_title(f"Is Target")
    axes[0].grid(1)

    axes[1].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        s=5,
    )
    axes[1].scatter(
        X_umap[gen_mask, 0],
        X_umap[gen_mask, 1],
        s=40,
    )
    axes[1].set_title("Is Generated")
    axes[1].grid(1)

    for ax in axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    plt.tight_layout()
    return fig


def dotplot(adata, target_ct, vmin=0.0, vmax=30.0):
    adata_gen = adata[adata.obs["data_type"]=="gen"]
    adata_real = adata[(adata.obs["data_type"]=="real") & (adata.obs["cell_type"] == target_ct)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    sc.pl.dotplot(
        adata_gen,
        adata.var_names,
        groupby="data_type",
        ax=axes[0],
        show=False, vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f"{target_ct} – Generated")

    sc.pl.dotplot(
        adata_real,
        adata_real.var_names,
        groupby="cell_type",
        ax=axes[1],
        show=False, vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f"{target_ct} – Target")
    fig.tight_layout()
    return fig


def get_distance_df(adata, obsm_key, classes, distance_fn=compute_e_distance):

    adata_pred = adata[adata.obs["data_type"] == "gen"]
    adata_true = adata[adata.obs["data_type"] != "gen"]

    edist_dict = defaultdict(list)
    for ct in tqdm(classes):
        adata_ct = adata_true[adata_true.obs.cell_type == ct]
        x_true = adata_ct.obsm[obsm_key] if obsm_key is not None else adata_ct.X
        edist = distance_fn(adata_pred.obsm[obsm_key]if obsm_key is not None else adata_pred.X, x_true)
        edist_dict["ct"].append(ct)
        edist_dict["edist"].append(edist)
    return pd.DataFrame(edist_dict).sort_values("edist", axis=0, ascending=False)


def barplot_clusters_distances(target_cell_type, adata, obsm_key, classes, distance_fn=compute_e_distance):
    # Create a color column
    edist_channel_df = get_distance_df(adata, obsm_key, classes, distance_fn=distance_fn)
    edist_channel_df["color"] = edist_channel_df["ct"].apply(
        lambda x: "red" if x == target_cell_type else "gray"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=edist_channel_df,
        x="ct",
        y="edist",
        palette=edist_channel_df["color"].tolist(),
        ax=ax
    )

    ax.grid(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("E-distance")
    ax.set_xlabel("Cell type")
    ax.set_title("E-distance per cell type (Channel Features)")
    fig.tight_layout()
    return fig


def plot_heatmap(
    base_dir,
    classes,
    target_ct,
    fwd_results,
    inverse_results,
    annotation_dict,
    samples_vmin,
    samples_vmax, 
    loss_vmin,
    loss_vmax,
    plot_pheno=False
):
    trajectory = inverse_results["trajectory"]
    loss_history = inverse_results["loss_history"]
    lambda_history = inverse_results["lambda_history"]
    noise = inverse_results["noise"]
    ct_probs = fwd_results["ct_probs"].mean(1)

    samples = np.maximum(trajectory[:, -1, :], 0)
    loss = loss_history[:, -1]

    clustering_ok = True
    try:
        cg = sns.clustermap(
            ct_probs if plot_pheno else samples,
            figsize=(14, 12),
            annot=False,
            xticklabels=classes if plot_pheno else annotation_dict["protocol_columns"],
            yticklabels=[],
            vmin=samples_vmin,
            vmax=1.0 if plot_pheno else samples_vmax,
        )
    except ValueError:
        cg = sns.clustermap(
            ct_probs if plot_pheno else samples,
            figsize=(14, 12),
            annot=False,
            xticklabels=classes if plot_pheno else annotation_dict["protocol_columns"],
            yticklabels=[],
            vmin=samples_vmin,
            vmax=1.0 if plot_pheno else samples_vmax,
            row_cluster=False,
            col_cluster=False,
        )
        clustering_ok = False


    cg.figure.subplots_adjust(top=0.9)
    cg.figure.suptitle(target_ct, y=0.97, fontsize=16)
    cg.cax.set_position([0.05, 0.2, 0.02, 0.3])

    if clustering_ok:
        row_order = cg.dendrogram_row.reordered_ind
        loss = loss[row_order]
    heatmap_pos = cg.ax_heatmap.get_position()
    loss_ax = cg.figure.add_axes([
        heatmap_pos.x1 + 0.01,  # small gap to the right
        heatmap_pos.y0,         # align bottom
        0.02,                   # width
        heatmap_pos.height      # same height as heatmap
    ])
    sns.heatmap(
        loss[:, None],
        ax=loss_ax,
        cbar=True,
        yticklabels=False,
        xticklabels=["loss"],
        vmin=loss_vmin,
        vmax=loss_vmax
    )

    min_loss_idx = np.argmin(loss)
    ax = cg.ax_heatmap
    xmin, xmax = ax.get_xlim()  # full heatmap width
    ymin, ymax = ax.get_ylim()  # row coordinates (usually ymax < ymin)


    if plot_pheno:
        for label in ax.get_xticklabels():
            if label.get_text() == target_ct:
                label.set_color("red")
                label.set_fontweight("bold")
                label.set_fontsize(12)

    # Because y-axis is inverted in heatmaps, compute height correctly
    rect_height = 1  # one row

    # Draw rectangle over the sample with minimum loss
    hrect = Rectangle(
        (xmin, min_loss_idx),    # left-bottom corner
        xmax - xmin,             # full width
        rect_height,             # one row
        fill=False,
        edgecolor="red",        # color of the rectangle
        linewidth=2.5,
        zorder=10
    )
    ax.add_patch(hrect)

    # ct_dir = os.path.join(base_dir, target_ct.replace("/", "_"))
    # create_dir(ct_dir, logger=logger)
    heatmap_path = os.path.join(base_dir,  "induced_pheno.png" if plot_pheno else "posterior_samples.png")

    bbox = cg.figure.get_tightbbox(cg.figure.canvas.get_renderer())
    bbox = bbox.expanded(1.15, 1.0)  # expand width only
    cg.figure.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches=bbox
    )
    plt.close(cg.figure)


def plot_loss_history(target_ct, loss_history):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=50)
    t = np.arange(0, 1, 1/loss_history.shape[1])
    try:
        lines = ax.plot(t, loss_history.T)
    except:
        lines = ax.plot(loss_history.T)
    fig.suptitle(f"{target_ct}")
    ax.set_xlabel("Sampling Time.")
    ax.set_ylabel("Loss.")
    ax.grid(1)
    fig.tight_layout()
    return fig


def xy_plot_summary_stats(
    gen_samples,
    true_samples,
    var_names,
    title=""
):
    # mean
    prior_mean = gen_samples.mean(0)
    real_mean = true_samples.mean(0)

    # standard deviation
    prior_std = gen_samples.std(0)
    real_std = true_samples.std(0)

    # initializing plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig.suptitle(title)

    # contructing linear space
    linspace = np.linspace(0.0, 6.5, 100)

    # plotting mean
    ax[0].scatter(prior_mean, real_mean); ax[0].grid(True)
    ax[0].set_title("Mean"); ax[0].set_xlabel("Generated"); ax[0].set_ylabel("Real")
    ax[0].plot(linspace, linspace)
    yoffset = 0.0
    xoffset = 0.3
    for i, txt in enumerate(var_names):
        ax[0].annotate(txt, (prior_mean[i] + xoffset, real_mean[i] + yoffset), size=6)

    # plotting standard deviation
    ax[1].scatter(prior_std, real_std); ax[1].grid(True)
    ax[1].set_title("Standard Deviation"); ax[1].set_xlabel("Generated"); ax[1].set_ylabel("Real")
    for i, txt in enumerate(var_names):
        ax[1].annotate(txt, (prior_std[i] + xoffset, real_std[i] + yoffset), size=6)
    ax[1].plot(linspace, linspace)
    fig.tight_layout()
    return fig, ax


def plot_marginals(X_real, X_gen=None, col_names=None, title=""):
    n_channels = X_real.shape[1]
    if X_gen is not None:
        assert X_gen.shape[1] == n_channels
    if col_names is not None:
        assert len(col_names) == n_channels

    fig, ax = plt.subplots(1, n_channels, figsize=(45, 5))
    fig.suptitle(f"{title}", fontsize=16)
    for i in range(n_channels):
        ax[i].grid(True)
        sns.kdeplot(X_real[:, i], ax=ax[i], color="green", label="real")
        sns.histplot(X_real[:, i], alpha=0.4, ax=ax[i], color="green", stat="density")
        if X_gen is not None:
            sns.kdeplot(X_gen[:, i], ax=ax[i], color="red", label="gen")
            sns.histplot(X_gen[:, i], alpha=0.4,  ax=ax[i], color="red", stat="density")
        ax[i].legend()
        if col_names is not None:
            ax[i].set_title(col_names[i])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def pairwise_scatter_plot(
    X,
    Y=None,
    title="",
    X_names=None,
    Y_names=None,
    base_size=5,
    dpi=500,
    compute_kde=True,
    figkwargs=None,
    scatterkwargs=None,
    show=True,
    use_seaborn=False
):

    # handling optional kwargs arguments
    figkwargs = {} if figkwargs is None else figkwargs
    scatterkwargs = {} if scatterkwargs is None else scatterkwargs

    # check for symmetric scatter plot
    if Y is None:
        Y=X
        is_symmetric = True
    else:
        is_symmetric = False
    
    # sanity check
    assert Y.shape[0] == X.shape[0], f"Shape error: {Y.shape=}, {X.shape=}"

    # retrieving number of cols and rows
    nrows = X.shape[-1]
    ncols = Y.shape[-1]

    # creating plot
    figsize = (base_size*ncols, base_size*nrows) 
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **figkwargs)
    fig.suptitle(title)

    # iterating over each axes
    for ridx in range(nrows):
        for cidx in range(ncols):
            # skipping diagonal and lower triangular
            # for symmetric plots
            if is_symmetric and cidx <= ridx:
                continue

            # retrieving axes
            ax = axes[ridx, cidx]
            ax.grid(True)

            # retrieving data
            x = X[:, ridx]
            y = Y[:, cidx]

            # retrieving name of variables
            if X_names is not None:
                assert len(X_names) == nrows, f"name length mismatch for X {len(X_names)} exp {nrows}"
                x_name = X_names[ridx]
            else:
                x_name = f"X comp {ridx}"
            if Y_names is not None:
                assert len(Y_names) == ncols, f"name length mismatch for Y {len(Y_names)} exp {ncols}"
                y_name = Y_names[cidx]
            else:
                if is_symmetric and X_names is not None:
                    y_name = X_names[cidx]
                else:
                    y_name = f"Y comp {cidx}"
            
            # creating df to plot
            df = pd.DataFrame(
                {
                    x_name: x.tolist(),
                    y_name: y.tolist()
                }
            )

            # optionally computing kde
            if compute_kde:
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                density = kde(xy)
                df["_density"] = density
                hue_arg = "_density"
            else:
                density = None
                hue_arg = None

            # scatter plot
            if use_seaborn:
                sns.scatterplot(df, x=x_name, y=y_name, ax=ax, hue=hue_arg, **scatterkwargs)
            else:
                ax.scatter(x, y, c=density, **scatterkwargs)
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes


def plot_adata(adata_pred, target_ct, classes, plots_dir, suffix, annotation_dict, save=True):
    # pca
    fig_first_quartile0 = plot_embedding(adata_pred, target_ct, base="X_pca")
    if save:
        fig_first_quartile0.savefig(
            os.path.join(plots_dir, f"pca_cell_type_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_first_quartile0)

    # umap
    fig_first_quartile1 = plot_embedding(adata_pred, target_ct)
    if save:
        fig_first_quartile1.savefig(
            os.path.join(plots_dir, f"umap_cell_type_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_first_quartile0)

    # dotplot
    fig_first_quartile2 = dotplot(adata_pred, target_ct)
    if save:
        fig_first_quartile2.savefig(
            os.path.join(plots_dir, f"dotplot_cell_type_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_first_quartile2)

    # barplot cluster distances
    fig_first_quartile3 = barplot_clusters_distances(
        target_ct, adata_pred, None, classes
    )
    if save:
        fig_first_quartile3.savefig(
            os.path.join(plots_dir, f"barplot_cluster_dists_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_first_quartile3)

    # parsing data
    adata_gen = adata_pred[adata_pred.obs["data_type"] == "gen"]
    adata_tgt = adata_pred[(adata_pred.obs["data_type"] != "gen") & (adata_pred.obs["cell_type"] == target_ct)]

    X_channel_gen = adata_gen.X
    X_channel_tgt = adata_tgt.X

    X_scatter_gen = adata_gen.obsm["X_scatter"]
    X_scatter_tgt = adata_tgt.obsm["X_scatter"]

    # xy plot marker and scatter
    fig_xy_channel = xy_plot_summary_stats(
        X_channel_gen,
        X_channel_tgt,
        adata_pred.var_names,
        title="Channel Features"
    )
    fig_xy_scatter = xy_plot_summary_stats(
        X_scatter_gen,
        X_scatter_tgt,
        annotation_dict["scatter_columns"],
        title="Scatter Features"
    )

    # marginals plot
    fig_marginals_channel = plot_marginals(
        X_channel_tgt,
        X_gen=X_channel_gen,
        col_names=adata_pred.var_names,
        title=""
    )
    fig_marginals_scatter = plot_marginals(
        X_scatter_tgt,
        X_gen=X_scatter_gen,
        col_names=annotation_dict["scatter_columns"],
        title=""
    )

    # scatter plot
    pairwise_scatter_plot(
        X,
        Y=None,
        title="",
        X_names=None,
        Y_names=None,
        base_size=5,
        dpi=500,
        compute_kde=True,
        figkwargs=None,
        scatterkwargs=None,
        show=True,
        use_seaborn=False
    )
    return fig_first_quartile0, fig_first_quartile1, fig_first_quartile2, fig_first_quartile3


def get_dimensionality_reduced_condition_space(
    original_data,
    perturbation_prediction_model,
    n_noise_samples
):
    # extract latent representation from model
    perturbation_reps = next(
        iter(
            perturbation_prediction_model.forward_model.train_data.data.perturbation_covariates
        )
    )
    latent_rep = perturbation_prediction_model.velocity_field.get_condition_embedding(
        {
            perturbation_reps: torch.from_numpy(
                original_data).float().to(perturbation_prediction_model.forward_model.device
            ),
        }
    ).detach().cpu().numpy()

    # sample noise for projection
    noise_orig = np.random.randn(n_noise_samples, original_data.shape[1], 2)
    noise_latent = np.random.randn(n_noise_samples, latent_rep.shape[1], 2)

    # project latent representation and data
    orig_rep_rand_proj = np.einsum("nd,mdk->nmk", original_data, noise_orig) / math.sqrt(2)
    latent_rep_rand_proj = np.einsum("nd,mdk->nmk", latent_rep, noise_latent) / math.sqrt(2)

    # compute pcs
    orig_rep_pcs = PCA(2).fit_transform(original_data)
    latent_rep_pcs = PCA(2).fit_transform(latent_rep)

    return (
        orig_rep_rand_proj, latent_rep_rand_proj, orig_rep_pcs, latent_rep_pcs
    )


def plot_dimensionality_reduced_condition_space(
    data_samples,
    gen_samples,
    data_color_val,
    gen_color_val,
    perturbation_prediction_model,
    n_noise_samples,
):
    # concatenate conditions
    data = np.concat((data_samples, gen_samples), axis=0)

    # get projections
    orig_rep_rand_proj, latent_rep_rand_proj, orig_rep_pcs, latent_rep_pcs = get_dimensionality_reduced_condition_space(
        data,
        perturbation_prediction_model,
        n_noise_samples
    )

    # plot projections
    ...


def run_level_plots(
    perturbation_prediction_model,
    data_samples,
    target_ct,
    run_level_plots_dir,
    fwd_results,
    inverse_results,
    annotation_dict,
    classes,
    data_color_val,
    gen_color_val,
    n_noise_samples=1000,
):

    # parse inverse results archive
    loss_history = inverse_results["loss_history"]
    loss = loss_history[:, -1]

    # loss history
    loss_history_fig = plot_loss_history(target_ct, loss_history)
    loss_history_fig.savefig(
        os.path.join(run_level_plots_dir, "loss_history.png"),
        dpi=300,
    )
    plt.close(loss_history_fig)

    # heatmap samples
    plot_heatmap(
        run_level_plots_dir,
        classes,
        target_ct,
        fwd_results,
        inverse_results,
        annotation_dict,
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=False
    )

    # heatmap pheno
    plot_heatmap(
        run_level_plots_dir,
        classes,
        target_ct,
        fwd_results,
        inverse_results,
        annotation_dict,
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=True
    )

    # plot dimensionality reduced conditions
    gen_samples = inverse_results["trajectory"][-1]
    plot_dimensionality_reduced_condition_space(
        data_samples,
        gen_samples,
        data_color_val,
        gen_color_val,
        perturbation_prediction_model,
        n_noise_samples,
    )

    # plot distance to nn in real data (condition space)
    ...

    # plot distance to target (phenotype space)
    ...

def covariate_level_plots():
    # histogram over each covariate axes colore by target prop
    ...

    # scatter plot vs target proportions
    ...
