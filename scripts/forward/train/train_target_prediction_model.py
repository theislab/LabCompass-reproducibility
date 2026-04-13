import logging
import os
import sys
import traceback


import hydra
import numpy as np
from omegaconf import DictConfig
import scanpy as sc
from sklearn.utils.class_weight import compute_class_weight
import torch

from sc_exp_design.models import TargetPredictionModel
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.training.callbacks import MetricsCallBack, TrainingCallBacks, WandBLogger

logger = logging.getLogger(__name__)

split_fns = {}
activation_functions = {}
optimizers = {}
schedulers = {}
state_transforms = {}


@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/forward/cell_type_classification/config",
    config_name="train_classifier",
)
def main(config: DictConfig):
    # import modules
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
    from data_utils import apply_shared_transformations
    from ood_utils import shuffle_split
    from train_utils import (
        parse_mlp_config_dictionary,
        parse_nested_mlp_config_dictionary,
        resolve_omegaconf_to_dictionary
    )

    # Data 0. reading adata
    logger.info("Reading data...")
    adata = sc.read_h5ad(config.paths.h5ad_path)
    logger.info(f"Data read! {adata}")

    # Data 1. splitting data
    logger.info(f"Splitting data...\n\tPerforming validation split {config.splits.mode}")
    split_fn = split_fns.get(config.splits.mode, shuffle_split)
    train_adata, ood_adata = split_fn(
        adata,
        **resolve_omegaconf_to_dictionary(config.splits.kwargs)
    )
    set_reproducibility(config.reproducibility.seed)
    # logger.info(f"Data split performed!\n \tTrain data of shape {train_adata.shape}\tValidation data of shape {ood_adata.shape}")

    # Data 2. apply shared transformations
    logger.info("Computing tranformation params on train data and applying to both train and ood data...")
    train_adata, ood_adatas_dict = apply_shared_transformations(
        train_adata,
        # {"test0": ood_adata},
        ood_adata,
        config.transforms.scatter_columns,
        compute_channel_pcs=config.transforms.compute_channel_pcs,
    )
    ood_adata = next(iter(ood_adatas_dict.values()))
    logger.info("Shared tranformations applied!")

    # Model 0. initializing model
    logger.info("Initializing model...")
    classifier = TargetPredictionModel(
        config.model.device_id
    )
    logger.info("Model inialized!")

    # Model 1. preparing data
    logger.info("Preparing training data...")
    classifier.prepare_train_data(
        train_adata,
        sample_rep=config.data.sample_rep,
        target_covariates=config.data.target_covariates,
        target_covariates_in_obsm=config.data.target_covariates_in_obsm,
        target_covariates_kwargs=config.data.target_covariates_kwargs,
    )
    logger.info("Train data ready!")
    logger.info("Preparing OOD data...")
    classifier.prepare_validation_data(ood_adata)
    logger.info("OOD data ready!")

    # Model 2. preparing model
    logger.info("Initializing classifier...")
    classifier.prepare_model(
        config.model.target_covariates,
        resolve_omegaconf_to_dictionary(config.model.target_covariates_dims),
        target_covariates_noise_models=resolve_omegaconf_to_dictionary(config.model.target_covariates_noise_models),
        target_covariates_predictor_kwargs=parse_nested_mlp_config_dictionary(activation_functions, config.model.target_covariates_predictor_kwargs),
        target_covariates_use_shared_representation=config.model.target_covariates_use_shared_representation,
        target_covariates_latent_dim=config.model.target_covariates_latent_dim,
        target_covariates_encoder_mlp_kwargs=parse_mlp_config_dictionary(activation_functions, config.model.target_covariates_encoder_mlp_kwargs),
        optimizer_class=optimizers.get(config.model.optimizer_id, torch.optim.Adam),
        optimizer_kwargs=resolve_omegaconf_to_dictionary(config.model.optimizer_kwargs),
        lr_scheduler_class=schedulers.get(config.model.scheduler_id, None),
        lr_scheduler_kwargs=resolve_omegaconf_to_dictionary(config.model.lr_scheduler_kwargs),
        lr_scheduler_step=config.model.lr_scheduler_step,
    )
    logger.info(f"Classifier initialized!\n{classifier.target_prediction_model}")

    # model 3. prepare training callbacks
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

    # model 4. optionally using class weights
    if config.training.use_class_weights:
        logger.info("Computing class weights...")
        values = train_adata.obs[config.training.class_weights_obs_col].values
        classes = np.unique(values)
        class_weights = torch.from_numpy(
            compute_class_weight(
                config.training.class_weight,
                classes=classes,
                y=values,
                **resolve_omegaconf_to_dictionary(config.training.class_weights_kwargs),
            )
        ).float().to(config.model.device_id)
        logger.info(f"Class weights computed: {class_weights}")
        loss_fn_kwargs = {"weight": class_weights}
    else:
        loss_fn_kwargs = None

    # model 4. training model
    logger.info("Training the model...")
    classifier.train(
        num_training_steps=config.training.num_training_steps,
        valid_freq=config.training.valid_freq,
        train_batch_size=config.training.train_batch_size,
        validation_batch_size=config.training.validation_batch_size,
        state_transforms=state_transforms.get(config.training.state_transforms, None),
        callbacks=callbacks,
        grad_steps_log_interval=config.training.grad_steps_log_interval,
        loss_fn_kwargs=loss_fn_kwargs,
    )
    logger.info("Model trained!")

    # model 5. save results (optional)
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
        classifier.save(
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
