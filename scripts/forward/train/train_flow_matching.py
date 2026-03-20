import os
import logging
import sys
import traceback

import hydra
from omegaconf import DictConfig
import torch

from sc_exp_design.config import NeuralVelocityFieldConfig
from sc_exp_design.models import FlowMatching
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.training.callbacks import WandBLogger, MetricsCallBack, TrainingCallBacks


logger = logging.getLogger(__name__)

DO_VALIDATION = False
time_samplers = {}
noise_distributions = {}
activation_functions = {}
optimizers = {}
schedulers = {}
state_transforms = {}


@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/forward/conditional_model/flow_matching/config",
    config_name="train_cfm",
)
def main(config: DictConfig):

    # import modules
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
    from data_utils import get_adata_splits
    from train_utils import (
        parse_mlp_config_dictionary,
        parse_nested_mlp_config_dictionary,
        resolve_omegaconf_to_dictionary
    )

    # 0. retrieving adata and ensuring reproducibility
    train_adata, ood_adatas_dict = get_adata_splits(config)
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
    logger.info("Model inialized!")

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
    logger.info("Preparing OOD data...")
    for k, v in ood_adatas_dict.items():
        flow_matching.prepare_validation_data(k, v)
    logger.info("OOD data ready!")
    print(flow_matching.train_data.perturbation_data.keys())

    # Model 3. initialize velocity field configurations and prepare additional arguments
    logger.info("Initializing neural configurations...")
    cvf_config = NeuralVelocityFieldConfig(
        flow_matching.train_data.state_data.shape[-1],
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
        resnet_normalization=config.vf.resnet_normalization,
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
        sample_groups=config.training.sample_groups,
    )
    logger.info("Model trained!")

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
    return 0


if __name__ == "__main__":

    # running the experiment
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
