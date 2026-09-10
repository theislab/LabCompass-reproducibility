import os
import logging
import sys
import traceback

import scanpy as sc
import numpy as np
import wandb

import hydra
from omegaconf import DictConfig
import torch

from sc_exp_design.config import NeuralVelocityFieldConfig
from sc_exp_design.models import FlowMatching, TargetPredictionModel
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.training.callbacks import WandBLogger, MetricsCallBack, TrainingCallBacks
from tqdm import tqdm
from sc_exp_design.constants import DataFields

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DO_VALIDATION = True
time_samplers = {}
noise_distributions = {}
activation_functions = {}
optimizers = {}
schedulers = {}
state_transforms = {}


@hydra.main(
    config_path="/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC/forward/conditional_model/flow_matching/config",
    config_name="train_cfm",
)
def main(config: DictConfig):

    # import modules
    sys.path.insert(0, "/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC/shared_utils")
    from train_utils import (
        parse_mlp_config_dictionary,
        parse_nested_mlp_config_dictionary,
        resolve_omegaconf_to_dictionary
    )

    # Data 0. loading adata
    logger.info("Loading data...")
    train_adata = sc.read_h5ad(config.paths.train_h5ad_path)
    val_adata = sc.read_h5ad(config.paths.val_h5ad_path)
    logger.info(f"Train data loaded! {train_adata}")
    logger.info(f"Validation data loaded! {val_adata}")

    set_reproducibility(config.reproducibility.seed)

    # Model 1. initialize flow matching model
    logger.info("Initializing model...")
    flow_matching =  FlowMatching(
        flow_type=config.flow_matching.flow_type,
        flow_kwargs=resolve_omegaconf_to_dictionary(config.flow_matching.flow_kwargs),
        coupling_type=config.flow_matching.coupling_type,
        coupling_kwargs=resolve_omegaconf_to_dictionary(config.flow_matching.coupling_kwargs),
        time_sampler=time_samplers.get(config.flow_matching.time_sampler, torch.rand),
        device_id=config.flow_matching.device_id,
        generate_from_noise=config.flow_matching.generate_from_noise,
        noise_distribution=noise_distributions.get(config.flow_matching.noise_distribution, torch.randn),
    )
    logger.info("Model initalized!")

    # Model 2. prepare data
    logger.info("Preparing training data...")
    flow_matching.prepare_train_data(
        train_adata,
        sample_rep=config.data.sample_rep,
        control_key=config.data.control_key,
        perturbations=config.data.perturbations,
        perturbations_in_obsm=config.data.perturbations_in_obsm,
        perturbation_covariates=config.data.perturbation_covariates,
        perturbation_reps=config.data.perturbation_reps,
        load_target_covariates=config.data.load_target_covariates,
        target_covariates=config.data.target_covariates,
        target_covariates_in_obsm=config.data.target_covariates_in_obsm,
        target_covariates_kwargs=resolve_omegaconf_to_dictionary(config.data.target_covariates_kwargs),
    )
    logger.info("Train data ready!")
    logger.info("Preparing validation data...")
    flow_matching.prepare_validation_data("RA_4", val_adata)
    logger.info("Validation data ready!")
    print(flow_matching.train_data.perturbation_data.keys())

    # Model 3. initialize velocity field configurations and prepare additional arguments
    logger.info("Initializing neural configurations...")
    resnet_normalization = config.vf.resnet_normalization
    if resnet_normalization == "none":
        resnet_normalization = None
    cvf_config = NeuralVelocityFieldConfig(
        train_adata.obsm[config.data.sample_rep].shape[1], # dimension of the flow, features
        encode_state=config.vf.encode_state,
        state_encoder_output_dim=config.vf.state_encoder_output_dim,
        state_encoder_mlp_kwargs=parse_mlp_config_dictionary(activation_functions, config.vf.state_encoder_mlp_kwargs),
        encode_time=config.vf.encode_time,
        use_sinusoidal_time_features=config.vf.use_sinusoidal_time_features,
        time_features_num_freqs=config.vf.time_features_num_freqs,
        time_features_max_periods=config.vf.time_features_max_periods,
        time_encoder_output_dim=config.vf.time_encoder_output_dim,
        time_encoder_mlp_kwargs=parse_mlp_config_dictionary(activation_functions, config.vf.time_encoder_mlp_kwargs),
        use_guidance=config.vf.use_guidance,
        encode_conditions=config.vf.encode_conditions,
        perturbation_encoder_output_dim=config.vf.perturbation_encoder_output_dim,
        perturbation_layers_before_pooling=parse_nested_mlp_config_dictionary(activation_functions, config.vf.perturbation_layers_before_pooling),
        perturbation_covariates_not_pooled=config.vf.perturbation_covariates_not_pooled,
        perturbation_pooling=config.vf.perturbation_pooling,
        perturbation_pooling_kwargs=config.vf.perturbation_pooling_kwargs,
        perturbation_layers_after_pooling=parse_mlp_config_dictionary(activation_functions, config.vf.perturbation_layers_after_pooling),
        decoder_mlp_kwargs=parse_mlp_config_dictionary(activation_functions, config.vf.decoder_mlp_kwargs),
        use_source_as_condition=config.vf.use_source_as_condition,
        encode_source=config.vf.encode_source, 
        source_encoder_output_dim=config.vf.source_encoder_output_dim,
        source_encoder_mlp_kwargs=parse_mlp_config_dictionary(activation_functions, config.vf.source_encoder_mlp_kwargs),
        conditioning_type=config.vf.conditioning_type,
        n_resnet_blocks=config.vf.n_resnet_blocks,
        resnet_dropout_prob=config.vf.resnet_dropout_prob,
        resnet_normalization=resnet_normalization,
        use_classifier_free_guidance=config.vf.use_classifier_free_guidance,
        cfg_null_condition_token=config.vf.cfg_null_condition_token,
    )
    logger.info("Configurations ready!")

    # model 4. initialize neural velocity field
    logger.info("Initializing velocity field...")
    flow_matching.prepare_model(
        cvf_config,
        optimizer_class=optimizers.get(config.model.optimizer_id, torch.optim.Adam),
        optimizer_kwargs=resolve_omegaconf_to_dictionary(config.model.optimizer_kwargs),
        lr_scheduler_class=schedulers.get(config.model.scheduler_id, None),
        lr_scheduler_kwargs=resolve_omegaconf_to_dictionary(config.model.lr_scheduler_kwargs),
        lr_scheduler_step=config.model.lr_scheduler_step,
        num_time_steps=config.model.num_time_steps,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.model.solver_kwargs),
    )
    logger.info(f"Velocity field initialized!\n{flow_matching.velocity_field}")

    # model 5. prepare training callbacks
    logger.info("Preparing training callbacks...")
    wandb_callback = WandBLogger(
        project_name=config.callbacks.wandb_project_name,
        log_dir=config.paths.log_dir,
        config=config,
        **resolve_omegaconf_to_dictionary(config.callbacks.wandb_kwargs),
    )
    metrics_callback = MetricsCallBack(
        config.callbacks.metrics,
        state_transforms=state_transforms.get(config.training.state_transforms, None),
    )
    callbacks = TrainingCallBacks(
        [
            wandb_callback,
            metrics_callback
        ]
    )
    logger.info("Callbacks ready!")

    # model 6. train model
    logger.info("Training the model...")
    flow_matching.train(
        num_training_steps=config.training.num_training_steps,
        valid_freq=config.training.valid_freq,
        train_batch_size=config.training.train_batch_size,
        validation_batch_size=config.training.validation_batch_size,
        state_transforms=state_transforms.get(config.training.state_transforms, None),
        callbacks=callbacks,
        grad_steps_log_interval=config.training.grad_steps_log_interval,
        num_treatments_to_load=config.training.num_treatments_to_load,
        num_samples_per_validation_step=config.training.num_samples_per_validation_step,
        cfg_prob_unconditional=config.training.cfg_prob_unconditional,
        validation_cfg_guidance_strength=config.training.validation_cfg_guidance_strength,
        num_grad_accumulation_steps=config.training.num_grad_accumulation_steps,
        close_wandb_connection=False,
        #sample_groups=config.training.sample_groups,
    )
    logger.info("Model trained!")

    # Generate samples from the trained model:
    # Predict the observations for the present perturbations in the validation set
    N = config.sampling.N
    M = config.sampling.M
    assert N % M == 0, f"N ({N}) must be divisible by M ({M})"
    B = N // M # samples per populations to estimate the standard deviation of the estimates
    regions = next(iter(config.data.target_covariates_kwargs.values()))[DataFields.TARGET_CATEGORIES]
    num_regions = len(regions)

    logger.info("Loading the trained classifier")
    clf = TargetPredictionModel.load(config.paths.classifier_path)
    logger.info(f"Classifier loaded! {clf}")

    logger.info(f"Generating {N} samples from the trained model")
    pred_mean_props = []
    pred_std_props = []
    val_treatments = val_adata.obs["condition"].unique().tolist()
    for treatment in tqdm(val_treatments): 
        pert_representation_val = torch.from_numpy(val_adata.uns["conditions"][treatment]).float().unsqueeze(0).expand(N, -1).to(flow_matching.device)
        batch_dict = {
            DataFields.PERTURBATION_DATA: {"repr_condition_conditions": pert_representation_val
                                                                }}
        # pushing forward the particles 
        X_pert_pred_val = flow_matching.predict(batch_dict, 
                                            no_grad=True,
                                            batch_size=(N,),
                                            num_time_steps = config.model.num_time_steps,
                                        )

        region_generated_logits = clf.predict(X_pert_pred_val)[next(iter(config.data.target_covariates_kwargs))] # could also write ["Region"]
        region_generated_logits = region_generated_logits.reshape(M, B, num_regions)
        region_generated_probs = torch.softmax(region_generated_logits, dim=-1)

        population_props = region_generated_probs.mean(dim=1)
        mean_props = population_props.mean(dim=0)
        std_props = population_props.std(dim=0)
        logger.info(f"Treatment: {treatment}, mean props: {mean_props}, std props: {std_props}")

        pred_mean_props.append(mean_props)
        pred_std_props.append(std_props)

    pred_mean_props = torch.stack(pred_mean_props) 
    pred_std_props = torch.stack(pred_std_props)

    # Evaluating the hyperparameters 
    logger.info("Evaluating hyperparameters...")
    true_props_val = (val_adata.obs
                      .groupby("condition", observed=True)[next(iter(config.data.target_covariates_kwargs))] 
                      .value_counts(normalize=True)
                      .unstack(fill_value=0)
                      .reindex(index=val_treatments, columns=regions, fill_value=0)
                      )
    
    true_props_tensor = torch.tensor(
        true_props_val.values,
        dtype=pred_mean_props.dtype,
        device=pred_mean_props.device,
        )
    
    eps = 1e-8 # for numerical stability
    pred_props_safe = pred_mean_props.clamp(min=eps)

    cross_entropy_per_treatment = -(true_props_tensor * torch.log(pred_props_safe)).sum(dim=1)
    mean_cross_entropy = cross_entropy_per_treatment.mean()
    logger.info(f"Cross entropy per treatment: {cross_entropy_per_treatment}")
    logger.info(f"Mean cross entropy: {mean_cross_entropy.item():.6f}")

    table = wandb.Table(
        columns=[
            "treatment",
            "region",
            "true_proportion",
            "predicted_proportion",
            "uncertainty",
            ]
    )

    for i, treatment in enumerate(val_treatments):
        for j, region in enumerate(regions):
            table.add_data(
                treatment,
                region,
                true_props_tensor[i, j].item(),
                pred_mean_props[i, j].item(),
                pred_std_props[i, j].item(),
            )

    wandb.log({
        "Validation/predictions": table,
        })

    wandb.log({
        "Validation/mean_cross_entropy": mean_cross_entropy.item(),
        "Validation/mean_uncertainty": pred_std_props.mean().item(),
        "Validation/max_uncertainty": pred_std_props.max().item(),
        })
    logger.info("Hyperparameters evaluated!")

    # model 7. save results (optional)
    if config.callbacks.save_model:
        # creating logging folder
        logger.info(f"Preparing output directories...")
        if not os.path.exists(config.paths.dump_dir):
            os.mkdir(config.paths.dump_dir)
            logger.info(f"Dump dir created at {config.paths.dump_dir}.")
        else:
            logger.warning(f"Dump dir {config.paths.dump_dir} alrerady exists.")

        # dumping model
        logger.info("Saving the trained model...")
        flow_matching.save(
            config.paths.dump_dir,
            model_prefix=wandb_callback.run_name,
        )
        logger.info("Model dumped and run finished!")
    return mean_cross_entropy.item()


if __name__ == "__main__":

    # running the experiment
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
