from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def compute_e_distance(
    pred,
    target
) -> float:
    """Compute the energy distance as in Peidli et al."""
    # computing energy distance
    sigma_pred = pairwise_distances(pred, pred, metric="sqeuclidean").mean()
    sigma_target = pairwise_distances(target, target, metric="sqeuclidean").mean()
    delta = pairwise_distances(pred, target, metric="sqeuclidean").mean()
    return 2 * delta - sigma_pred - sigma_target


def compute_distance_fn(
    adata,
    groups,
    state_repr = None,
    distance_fn = compute_e_distance,
    sep = "|",
    **kwargs
):
    # define dictionary to store results
    results_dict = {}

    # define progress bar
    n_groups = len(groups)
    pbar = tqdm(range(n_groups))

    # outer loop: iterating over each group
    for idx0, (group0_id, group0_idxs) in enumerate(groups.items()):
        # retrieving states
        if state_repr is not None:
            X0 = adata[group0_idxs].obsm[state_repr]
        else:
            X0 = adata[group0_idxs].X

        # inner loop: iterating over each group
        for idx1, (group1_id, group1_idxs) in enumerate(groups.items()):
            # skipping already computed distances
            if idx1 <= idx0:
                continue
            # retrieving states
            if state_repr is not None:
                X1 = adata[group1_idxs].obsm[state_repr]
            else:
                X1 = adata[group1_idxs].X

            # updating progress bar
            pbar.set_description(f"Computing Distances {group0_id}:{X0.shape[0]}:({idx0}/{n_groups}) <-> {group1_id}:{X1.shape[0]}:({idx1}/{n_groups})")
            pbar.update()

            # defining key for storing results
            key = f"{group0_id}{sep}{group1_id}"

            # computing distances
            results_dict[key] = distance_fn(X0, X1, **kwargs).item()
    return results_dict
