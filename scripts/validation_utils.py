import numpy as np
import torch

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.data.container import DataMixin

def predict_on_ood_data(
    N,
    flow_matching,
    ood_data,
):
    control_data = ood_data.control_data
    perturbation_data = ood_data.perturbation_data

    if perturbation_data is not None:
        unique_perts = DataMixin(perturbation_data.apply(lambda x: np.unique(x, axis=0)))
        perts_data = unique_perts.apply(lambda x: torch.from_numpy(x).float().to(flow_matching.device))
        if not flow_matching.generate_from_noise:
            perts_data = perts_data.apply(lambda x: x.unsqueeze(0).repeat(N, 1, 1))
    
    if control_data is not None:
        if not flow_matching.generate_from_noise:
            control_data = control_data.apply(lambda x: x.unsqueeze(0).repeat(N, 1, 1))

    return flow_matching.predict(
        {
            DataFields.SOURCE_STATE: control_data,
            DataFields.PERTURBATION_DATA: perts_data,
        },
        num_samples=N,
    ).detach().cpu().numpy()

def validate_on_ood_data(
    N,
    flow_matching,
    ood_data,
    metrics_callback,
    sep="+"
):
    x1_hat = predict_on_ood_data(
        N, flow_matching, ood_data
    )
    preds_dict = DataMixin({
        sep.join(comb_id): x1_hat[:, idx, :] for idx, comb_id in enumerate(ood_data.seen_combinations)
    })
    preds_dict[DataFields.CONDITION_VALUES] = np.concat(list(preds_dict.values()))

    state_data = ood_data.state_data
    treatment_idxs_per_condition = ood_data.treatment_idxs_per_condition
    treatment_idxs_per_condition = DataMixin({
        k if isinstance(k, str) else sep.join(k): v for k, v in  treatment_idxs_per_condition.items()
    })
    target_data = treatment_idxs_per_condition.apply(lambda idx: state_data[idx])

    metrics_input = {
        k: {
            PredictionFields.PREDICTION_DATA: preds_dict[k],
            DataFields.TARGET_STATE: target_data[k]
        } for k in target_data.keys()
    }

    return metrics_callback.run_on_valid_step(
        metrics_input
    )
