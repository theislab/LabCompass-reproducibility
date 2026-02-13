import os
import sys

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

from sc_exp_design.metrics import compute_e_distance

sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
from experiment_utils import (
    get_distance_df,
    get_dimensionality_reduced_condition_space,
    get_distance_matrix,
)


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


def binplot_with_colors(
    protocol_columns,
    X,
    c,
    ct_string,
    bins=100,
    agg = "mean",
    cmap_name="viridis",
):
    agg_funcs = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
        "min": np.min,
        "max": np.max,
    }
    agg_func = agg_funcs[agg]

    fig, ax = plt.subplots(nrows=1, ncols=len(protocol_columns), figsize=(50, 12))

    for cidx, col in enumerate(protocol_columns):
        x = X[:, cidx]
        # Histogram
        counts, bin_edges = np.histogram(x, bins=bins)
        bin_ids = np.digitize(x, bin_edges) - 1
        # Aggregate color column per bin
        bin_color = np.array([
            agg_func(c[bin_ids == i]) if np.any(bin_ids == i) else np.nan
            for i in range(len(bin_edges) - 1)
        ])
        cmap = cm.get_cmap(cmap_name)
        norm = colors.Normalize(
            vmin=np.nanmin(bin_color),
            vmax=np.nanmax(bin_color)
        )

        # Plot
        ax[cidx].bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            align="edge",
            color=cmap(norm(bin_color)),
            edgecolor="none"
        )
        ax[cidx].set_xlabel(col)
        ax[cidx].set_title(ct_string)
    return fig


def scatter_protocol_covariate_against_prop(
    protocol_columns,
    X,
    y,
    ubounds,
    margin=10,
    use_density=True
):
    fig, ax = plt.subplots(nrows=1, ncols=len(protocol_columns), figsize=(50, 12))
    
    for cidx, col in enumerate(protocol_columns):
        x_data = X[:, cidx]
        
        if use_density:
            # Calculate point density for coloring
            xy = np.vstack([x_data, y])
            z = gaussian_kde(xy)(xy)
            # Sort the points by density so that the densest points are on top
            idx = z.argsort()
            x_plot, y_plot, z_plot = x_data[idx], y[idx], z[idx]
            
            scatter = ax[cidx].scatter(x_plot, y_plot, c=z_plot, s=50, cmap='viridis')
            plt.colorbar(scatter, ax=ax[cidx], label='Density')
        else:
            ax[cidx].scatter(x_data, y, alpha=0.5)

        ax[cidx].set_title(f"{col}", fontsize=16)
        ax[cidx].set_ylabel(f"Target Prob")
        ax[cidx].set_xlabel(col)

        # Draw the experimental boundary
        max_val = ubounds[cidx] # Fixed indexing (assumes ubounds is a vector)
        ax[cidx].axvline(x=max_val, color="red", linestyle="--", linewidth=2, label="Constraint")
        ax[cidx].set_xlim(0, max_val + margin)
        ax[cidx].legend()
        ax[cidx].grid(True, alpha=0.3)
        
    return fig



def barplot_clusters_distances(
    target_cell_type,
    adata,
    obsm_key,
    classes,
    distance_fn=compute_e_distance
):
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

    heatmap_path = os.path.join(base_dir,  "induced_pheno.png" if plot_pheno else "posterior_samples.png")

    bbox = cg.figure.get_tightbbox(cg.figure.canvas.get_renderer())
    bbox = bbox.expanded(1.15, 1.0)  # expand width only
    cg.figure.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches=bbox
    )
    plt.close(cg.figure)


def plot_loss_history(
    target_ct,
    loss_history,
    take_transpose=True
):
    if take_transpose:
        loss_history = loss_history.T
    fig, ax = plt.subplots(figsize=(7, 7), dpi=50)
    t = np.arange(0, 1, 1/loss_history.shape[1])
    try:
        _ = ax.plot(t, loss_history)
    except:
        _ = ax.plot(loss_history)
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
    return fig


def plot_marginals(
    X_real,
    X_gen=None,
    col_names=None,
    title=""
):
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
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False, **figkwargs)
    fig.suptitle(title)

    # iterating over each axes
    for ridx in range(nrows):
        for cidx in range(ncols):
            # skipping diagonal and lower triangular
            # for symmetric plots
            if is_symmetric and cidx <= ridx:
                axes[ridx, cidx].axis('off')
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
    return fig


def plot_dimensionality_reduced_condition_space(
    data_samples,
    gen_samples,
    data_color_val,
    gen_color_val,
    perturbation_prediction_model,
    n_noise_samples,
):
    data = np.concatenate((data_samples, gen_samples), axis=0)
    n_real = len(data_samples)
    
    # Get projections from your previously defined function
    orig_rand, latent_rand, orig_pcs, latent_pcs = get_dimensionality_reduced_condition_space(
        data, perturbation_prediction_model, n_noise_samples
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["Orig (Rand Proj)", "Latent (Rand Proj)", "Orig (PCA)", "Latent (PCA)"]
    projs = [orig_rand.mean(1), latent_rand.mean(1), orig_pcs, latent_pcs]

    for i, ax in enumerate(axes.flat):
        # Plot real data
        ax.scatter(projs[i][:n_real, 0], projs[i][:n_real, 1], 
                   c=data_color_val, cmap='viridis', s=10, label='Real', alpha=0.6)
        # Plot generated data
        ax.scatter(projs[i][n_real:, 0], projs[i][n_real:, 1], 
                   c=gen_color_val, cmap='magma', s=30, marker='x', label='Gen')
        ax.set_title(titles[i])
        ax.legend()

    plt.tight_layout()
    return fig


def plot_manifold_distances(X_real, X_gen, title):
    """Helper to plot distribution of distances to nearest real neighbors."""
    fig, ax = plt.subplots(figsize=(12, 10))
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_real)
    distances, _ = nbrs.kneighbors(X_gen)
    
    sns.kdeplot(distances.flatten(), ax=ax, fill=True, color="crimson")
    ax.set_title(f"Distance to Real Manifold ({title})")
    ax.set_xlabel("L2 Distance")
    ax.grid(True)
    return fig


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
    fig_proj = plot_dimensionality_reduced_condition_space(
        data_samples,
        gen_samples,
        data_color_val,
        gen_color_val,
        perturbation_prediction_model,
        n_noise_samples,
    )
    fig_proj.savefig(
        os.path.join(run_level_plots_dir, "dimensionality_reduced_conds.png"),
        dpi=300,
    )
    plt.close(fig_proj)

    # plot distance to nn in real data (condition space)
    fig_cond_dist = plot_manifold_distances(data_samples, gen_samples, "Condition Space")
    fig_cond_dist.savefig(
        os.path.join(run_level_plots_dir, "distance_condition_space.png"),
        dpi=300,
    )
    plt.close(fig_cond_dist)


def covariate_level_plots(
    target_ct,
    gen_samples,
    target_probs,
    protocol_columns,
    upper_bounds,
    plots_dir,
    suffix=""
):
    """
    Orchestrates protocol-level visualizations to see how experimental 
    variables drive the target phenotype.
    """
    
    # 1. Binned Histogram: Frequency of conditions colored by Target Prop
    fig_bins = binplot_with_colors(
        protocol_columns,
        gen_samples,
        target_probs,
        target_ct,
        bins=100,
        agg = "mean",
        cmap_name="viridis",
    )
    fig_bins.savefig(os.path.join(plots_dir, f"covariate_bins_{suffix}.png"), dpi=300)
    plt.close(fig_bins)

    # 2. Scatter vs Proportions: Direct sensitivity check
    fig_scatter = scatter_protocol_covariate_against_prop(
        protocol_columns=protocol_columns,
        X=gen_samples,
        y=target_probs,
        ubounds=upper_bounds
    )
    fig_scatter.savefig(os.path.join(plots_dir, f"covariate_scatters_{suffix}.png"), dpi=300)
    plt.close(fig_scatter)


def sample_level_plots(
    adata_pred,
    target_ct,
    classes,
    plots_dir,
    suffix,
    annotation_dict,
    save=True
):
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
        plt.close(fig_first_quartile1)

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
    if save:
        fig_xy_channel.savefig(
            os.path.join(plots_dir, f"xy_plt_channel_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_xy_channel)
    fig_xy_scatter = xy_plot_summary_stats(
        X_scatter_gen,
        X_scatter_tgt,
        annotation_dict["scatter_columns"],
        title="Scatter Features"
    )
    if save:
        fig_xy_scatter.savefig(
            os.path.join(plots_dir, f"xy_plt_scatter_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_xy_scatter)

    # marginals plot
    fig_marginals_channel = plot_marginals(
        X_channel_tgt,
        X_gen=X_channel_gen,
        col_names=adata_pred.var_names,
        title=""
    )
    if save:
        fig_marginals_channel.savefig(
            os.path.join(plots_dir, f"marginals_channel_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_marginals_channel)
    fig_marginals_scatter = plot_marginals(
        X_scatter_tgt,
        X_gen=X_scatter_gen,
        col_names=annotation_dict["scatter_columns"],
        title=""
    )
    if save:
        fig_marginals_scatter.savefig(
            os.path.join(plots_dir, f"marginals_scatter_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_marginals_scatter)

    # scatter plot channel
    fig_channel_vs_channel = pairwise_scatter_plot(
        X_channel_gen,
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
    if save:
        fig_channel_vs_channel.savefig(
            os.path.join(plots_dir, f"pairwise_channel_channel_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_channel_vs_channel)

    # scatter plot scatter
    fig_scatter_vs_scatter = pairwise_scatter_plot(
        X_scatter_gen,
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
    if save:
        fig_scatter_vs_scatter.savefig(
            os.path.join(plots_dir, f"pairwise_scatter_scatter_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_scatter_vs_scatter)

    # scatter plot scatter-channel
    fig_channel_vs_scatter = pairwise_scatter_plot(
        X_scatter_gen,
        Y=X_channel_gen,
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
    if save:
        fig_channel_vs_scatter.savefig(
            os.path.join(plots_dir, f"pairwise_scatter_channel_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig_channel_vs_scatter)

    return (
        fig_first_quartile0, fig_first_quartile1, fig_first_quartile2, fig_first_quartile3,
        fig_xy_channel, fig_xy_scatter, fig_marginals_channel, fig_marginals_scatter,
        fig_channel_vs_channel, fig_scatter_vs_scatter, fig_channel_vs_scatter
    )
