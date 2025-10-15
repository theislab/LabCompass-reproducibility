import numpy as np
import torch

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.data.container import DataMixin


def predict_on_ood_data(
    N,
    flow_matching,
    ood_data,
):
    perturbation_data = ood_data.perturbation_data
    assert perturbation_data is not None
    if ood_data.seen_combinations is not None:
        perturbation_data = DataMixin(ood_data.perturbation_data)
        comb2pred = {}
        # iterating over unique combinations
        for comb in ood_data.seen_combinations:
            comb_idxs = np.all((ood_data.adata.obs[flow_matching.data_manager.perturbations] == comb).values, axis=-1)
            comb_adata = ood_data.adata[comb_idxs]
            comb_data = flow_matching.data_manager.get_data(comb_adata)
            comb_perts = DataMixin(comb_data.perturbation_data)
            comb_perts = comb_perts.apply(lambda x: np.unique(x, axis=0))
            comb_perts = comb_perts.apply(lambda x: torch.from_numpy(x).float().to(flow_matching.device).repeat(N, 1))
            preds = flow_matching.predict(
                {
                    DataFields.PERTURBATION_DATA: comb_perts,
                },
            ).detach().cpu().numpy()
            comb2pred[comb] = {
                PredictionFields.PREDICTION_DATA: preds,
                DataFields.TARGET_STATE: comb_data.state_data
            }

    # paired setting TODO
    else:
        comb2pred = {}
        perturbation_data = DataMixin(ood_data.perturbation_data)
        unique_perts = perturbation_data.apply(lambda x: np.unique(x, axis=0))
        print("unique_perts ", next(iter(unique_perts.values())).shape)
        for idx in range(next(iter(unique_perts.values())).shape[0]):
            comb_perts = unique_perts.apply(lambda x: x[idx, :])
            print("ood_data", next(iter(perturbation_data.values())).shape)
            print("comb_perts", next(iter(comb_perts.values())).shape)
            comb_idxs = np.all(next(iter(perturbation_data.values())) == next(iter(comb_perts.values())), axis=-1)
            print("comb_idxs", comb_idxs.shape)

            comb_adata = ood_data.adata[comb_idxs]
            comb_data = flow_matching.data_manager.get_data(comb_adata)
            comb_perts = comb_perts.apply(lambda x: torch.from_numpy(x[None]).float().to(flow_matching.device).repeat(N, 1))
            print(comb_perts)
            preds = flow_matching.predict(
                {
                    DataFields.PERTURBATION_DATA: comb_perts,
                },
            ).detach().cpu().numpy()
            comb2pred[idx] = {
                PredictionFields.PREDICTION_DATA: preds,
                DataFields.TARGET_STATE: comb_data.state_data
            }

    return comb2pred

def validate_on_ood_data(
    N,
    flow_matching,
    ood_data,
    metrics_callback,
):
    metrics_input = predict_on_ood_data(
        N, flow_matching, ood_data
    )
    return metrics_callback.run_on_valid_step(
        metrics_input
    )