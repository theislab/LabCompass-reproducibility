from sklearn.metrics import pairwise_distances

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
