from collections.abc import Sequence
from functools import partial
import sys
import os
import traceback
import logging
from typing import Any

from anndata import AnnData
from omegaconf import DictConfig, OmegaConf
import hydra

from sc_exp_design.training.callbacks import (
    MetricsCallBack,
    WandBLogger,
    TrainingCallBacks,
)
from sc_exp_design.config import NeuralVelocityFieldConfig
from sc_exp_design.models import FlowMatching
from sc_exp_design.utils import set_reproducibility

from utils import (
    parse_mlp_config_dictionary,
    resolve_omegaconf_to_dictionary,
    TimeSamplers,
    Optimizers,
    LRSchedulers,
    StateTransforms,
    NoiseDistributions,
    ActivationFunctions,
)
from dispatcher import fetch_adata
from split_data import split_train_validation_adata

logger = logging.getLogger(__name__)


@hydra.main()
def main(config: DictConfig) -> int:
    """"""
    # optional seeding of random number generation
    if config.reproducibility.seed is not None:
        set_reproducibility(config.reproducibility.seed)

    # retrieving settings
    time_sampler = vars(TimeSamplers())[config.flow_matching.time_sampler]
    optimizer_class = vars(Optimizers())[config.model.optimizer]
    lr_scheduler_class = vars(LRSchedulers())[config.model.lr_scheduler]
    state_transforms = vars(StateTransforms())[config.training.state_transforms]
    noise_distribution = vars(NoiseDistributions())[config.flow_matching.noise_distribution]
    resnet_normalization = None if config.vf.resnet_normalization=="none" else config.vf.resnet_normalization

    # preparing validation split function
    validation_split_fn = None
    if config.dispatcher.split_data:
        validation_split_kwargs = {}
        if config.dispatcher.split_kwargs is not None:
            validation_split_kwargs = config.dispatcher.split_kwargs
        validation_split_fn = partial(split_train_validation_adata, **validation_split_kwargs)

    # retrieving data
    if config.callbacks.verbose:
        logger.info("Loading adata...")
    train_adata, val_adata = fetch_adata(
        config.dispatcher.dataset_name,
        config.paths.h5ad_data_path,
        config.dispatcher.h5ad_data_kwargs,
        validation_split_fn,
    )
    if config.callbacks.verbose:
        logger.info("Adata loaded!")
        logger.info(f"Train adata {train_adata}")
        logger.info(f"Validation adata {val_adata}")

    # retrieving dimensionality of the flow
    if config.data.sample_rep is not None:
        flow_dim = train_adata.obsm[config.data.sample_rep].shape[1]
    else:
        flow_dim = train_adata.X.shape[1]

    # initializing flow model
    if config.callbacks.verbose:
        logger.info("Initializing forward flow matching model")
    flow_model = FlowMatching(
        flow_type=config.flow_matching.flow_type,
        flow_kwargs=config.flow_matching.flow_kwargs,
        coupling_type=config.flow_matching.coupling_type,
        coupling_kwargs=config.flow_matching.coupling_kwargs,
        time_sampler=time_sampler,
        device_id=config.flow_matching.device_id,
        generate_from_noise=config.flow_matching.generate_from_noise,
        noise_distribution=noise_distribution,
    )
    if config.callbacks.verbose:
        logger.info("Forward flow matching model intialized!")

    # preparing train data
    if config.callbacks.verbose:
        logger.info("Preparing training data...")
    flow_model.prepare_train_data(
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
        target_covariates_kwargs=config.data.target_covariates_kwargs,
    )
    if config.callbacks.verbose:
        logger.info("Training data ready!")
    # preparing optional validation data
    if val_adata is not None:
        if config.callbacks.verbose:
            logger.info("Preparing validation data...")
        flow_model.prepare_validation_data(val_adata)
        if config.callbacks.verbose:
            logger.info("Validation data ready!")

    # parsing mlp configurations
    if config.callbacks.verbose:
        logger.info("Parsing configurations...")
    state_encoder_mlp_kwargs = {}
    if config.vf.state_encoder_mlp_kwargs is not None:
        state_encoder_mlp_kwargs = parse_mlp_config_dictionary(config.vf.state_encoder_mlp_kwargs)
    
    time_encoder_mlp_kwargs = {}
    if config.vf.time_encoder_mlp_kwargs is not None:
        time_encoder_mlp_kwargs = parse_mlp_config_dictionary(config.vf.time_encoder_mlp_kwargs)

    decoder_mlp_kwargs = {}
    if config.vf.decoder_mlp_kwargs is not None:
        decoder_mlp_kwargs = parse_mlp_config_dictionary(config.vf.decoder_mlp_kwargs)

    perturbation_layers_before_pooling = None
    if config.vf.perturbation_layers_before_pooling is not None:
        perturbation_layers_before_pooling = {}
        for perturbation, perturbation_kwargs in config.vf.perturbation_layers_before_pooling.items():
            perturbation_kwargs = parse_mlp_config_dictionary(perturbation_kwargs)
            perturbation_layers_before_pooling[perturbation] = perturbation_kwargs

    perturbation_layers_after_pooling = None
    if config.vf.perturbation_layers_after_pooling is not None:
        perturbation_layers_after_pooling = parse_mlp_config_dictionary(config.vf.perturbation_layers_after_pooling)

    source_encoder_mlp_kwargs = {}
    if config.vf.source_encoder_mlp_kwargs is not None:
        source_encoder_mlp_kwargs = parse_mlp_config_dictionary(config.vf.source_encoder_mlp_kwargs)
    if config.callbacks.verbose:
        logger.info("Configuration parsed!")

    # retrieving velocity field configurations
    if config.callbacks.verbose:
        logger.info("Inizializing velocitity field...")
    cvf_config = NeuralVelocityFieldConfig(
        flow_dim,
        encode_state=config.vf.encode_state,
        state_encoder_output_dim=config.vf.state_encoder_output_dim,
        state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
        encode_time=config.vf.encode_time,
        use_sinusoidal_time_features=config.vf.use_sinusoidal_time_features,
        time_features_num_freqs=config.vf.time_features_num_freqs,
        time_features_max_periods=config.vf.time_features_max_periods,
        time_encoder_output_dim=config.vf.time_encoder_output_dim,
        time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
        use_guidance=config.vf.use_guidance,
        encode_conditions=config.vf.encode_conditions,
        perturbation_latent_dim=config.vf.perturbation_latent_dim,
        perturbation_layers_before_pooling=perturbation_layers_before_pooling,
        perturbation_covariates_not_pooled=config.vf.perturbation_covariates_not_pooled,
        perturbation_pooling=config.vf.perturbation_pooling,
        perturbation_pooling_kwargs=config.vf.perturbation_pooling_kwargs,
        perturbation_layers_after_pooling=perturbation_layers_after_pooling,
        decoder_mlp_kwargs=decoder_mlp_kwargs,
        use_source_as_condition=config.vf.use_source_as_condition,
        encode_source=config.vf.encode_source, 
        source_latent_dim=config.vf.source_latent_dim,
        source_encoder_mlp_kwargs=source_encoder_mlp_kwargs,
        use_resnet_blocks=config.vf.use_resnet_blocks,
        n_resnet_blocks=config.vf.n_resnet_blocks,
        resnet_dropout_prob=config.vf.resnet_dropout_prob,
        resnet_normalization=resnet_normalization,
    )

    # resolving the dictionary settings from omegaconf
    solver_kwargs = resolve_omegaconf_to_dictionary(config.model.solver_kwargs)
    lr_scheduler_kwargs = resolve_omegaconf_to_dictionary(config.model.lr_scheduler_kwargs)  

    # preparing the flow model
    flow_model.prepare_model(
        cvf_config,
        optimizer_class=optimizer_class,
        optimizer_kwargs=config.model.optimizer_kwargs,
        lr_scheduler_class=lr_scheduler_class,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        lr_scheduler_step=config.model.lr_scheduler_step,
        num_time_steps=config.model.num_time_steps,
        solver_kwargs=solver_kwargs,
    )
    if config.callbacks.verbose:
        logger.info("Velocity field initialized!")

    # prepare logging kwargs
    if config.callbacks.verbose:
        logger.info("Preparing to train the forward model...")
    wandb_kwargs = config.callbacks.wandb_kwargs
    if wandb_kwargs is None:
        wandb_kwargs = {}

    # prepare WandB logging callback
    wandb_callback = WandBLogger(
        project_name=config.callbacks.wandb_project_name,
        log_dir=config.paths.log_dir,
        config=config,
        **wandb_kwargs,
    )

    # prepare metrics computation callback
    metrics_callback = MetricsCallBack(
        config.callbacks.metrics,
        state_transforms=state_transforms,
    )

    # initialize callbacks for training
    callbacks = TrainingCallBacks([wandb_callback, metrics_callback])

    # training the flow model
    if config.callbacks.verbose:
        logger.info("Starting training!")
    flow_model.train(
        num_training_steps=config.training.num_training_steps,
        valid_freq=config.training.valid_freq,
        train_batch_size=config.training.train_batch_size,
        validation_batch_size=config.training.validation_batch_size,
        state_transforms=state_transforms,
        callbacks=callbacks,
        grad_steps_log_interval=config.training.grad_steps_log_interval,
    )
    if config.callbacks.verbose:
        logger.info("Training finished!")

    # saving the model
    if config.callbacks.save_model:
        # creating logging folder
        if not os.path.exists(config.paths.dump_dir):
            os.mkdir(config.paths.dump_dir)
            if config.callbacks.verbose:
                logger.info(f"Dump dir created at {config.paths.dump_dir}.")
        else:
            if config.callbacks.verbose:
                logger.warning(f"Dump dir {config.paths.dump_dir} alrerady exists.")
        
        # dumping model
        if config.callbacks.verbose:
            logger.info("Saving the trained model...")
        flow_model.save(
            config.paths.dump_dir,
            model_prefix=wandb_callback.run_name,       
        )
        if config.callbacks.verbose:
            logger.info("Model dumped and run finished!")

    return 0

if __name__ == "__main__":

    # running the experiment
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)