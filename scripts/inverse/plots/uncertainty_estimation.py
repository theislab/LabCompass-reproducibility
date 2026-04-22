import pandas as pd
from tqdm import tqdm 
import scanpy as sc
import sys
import os 
import logging 
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from pathlib import Path 
import torch.nn.functional as F

from sc_exp_design.constants import DataFields

BASE_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
CONFIG_PATH = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/loss_guidance/config"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config: DictConfig):
    # Load forward model and shared utils
    sys.path.insert(0, os.path.join(BASE_DIR, "shared_utils"))
    logger.info("Loading forward model")    
    from experiment_utils import (
        get_forward_model,
        get_target_dict, 
        get_loss_fn, 
        compute_exploitation_score, 
        compute_exploration_score, 
        compute_weighted_uncertainty_score
    )
    
    # Sanity check
    assert config.n_noise_samples % config.n_populations == 0, "The number of noise samples must be divisible by the number of populations."
    
    with initialize(config_path=config.base_config_path, version_base=None):
        base_cfg = compose(
            config_name=config.base_config_name,
            overrides=[f"paths={config.paths}"]
        )
    
    print("Using checkpoints", base_cfg.paths.perturbation_prediction_path)
        
    forward_model, (_, target_prediction_model) = get_forward_model(base_cfg, logger=logger)
    logger.info("Read model")
    
    # Unique concentration adata 
    adata_unique_concentrations = sc.read_h5ad(config.true_concentration_path)
    protocol_columns = base_cfg["annotation"]["protocol_columns"]
    adata_unique_concentrations =  adata_unique_concentrations[:, protocol_columns]
    X_real_concentrations = adata_unique_concentrations.X
    protocol_columns = [col + ":rescaled" for col in protocol_columns]
    
    # Prepare label encoder
    logger.info("Preparing labels")   
    ct_le = LabelEncoder()
    ct_values = target_prediction_model.train_data.adata.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()  # List of cell types in alphabetical order 
    
    # Optimize over a cell type list 
    cell_type_list = config.target_cell_types.split("__")

    for cell_type_id in tqdm(cell_type_list):  # Iterate over cell types 
        logger.info(f"Evaluating cell type {cell_type_id}")
        
        # Take cell type index 
        cell_type_index = classes.index(cell_type_id)
        ct_safe_string = cell_type_id.replace("/", ":") # cell type dir
        
        # Initialize cell type folder 
        target_ct_dir = os.path.join(config.result_dir, config.experiment_type)
        target_ct_dir = os.path.join(target_ct_dir, ct_safe_string)
        for run_dir in tqdm(os.listdir(target_ct_dir)):  # Loop across experiments
            # Create folder for uncertainty annotation 
            uncertainty_annotation_folder = Path(os.path.join(target_ct_dir, run_dir, "uncertainty_annotation"))
            uncertainty_annotation_folder.mkdir(parents=True, exist_ok=True)
            
            # Read the csv file 
            result_csv_path = os.path.join(target_ct_dir, run_dir, "candidates.csv")    
            result_csv = pd.read_csv(result_csv_path)
            
            # Read yaml config 
            logger.info("Initialize configuration and protocol columns")
            config_run = OmegaConf.load(os.path.join(target_ct_dir, run_dir, "config.yaml"))

            # Subset to protocol columns and move to numpy
            X_candidates = result_csv.loc[:, protocol_columns]
            X_candidates = torch.from_numpy(
                X_candidates.values
                ).float().to(
                    forward_model.forward_model.device
                    )
                
            # Expand candidates 
            logger.info("Perform predictions")
            X_candidates_add = X_candidates.unsqueeze(1).repeat(1, config.n_noise_samples, 1)  # no_candidates x n_noise_sample x candidate_dim
            condition_repr = next(iter(forward_model.forward_model.train_data.data.perturbation_covariates))
            batch_dict = {DataFields.PERTURBATION_DATA: {condition_repr: X_candidates_add}}
            forward_out = forward_model.predict(
                batch_dict,
                no_grad=True,
                fix_noise=False,
                num_time_steps=30
            )
            # Collect predictions 
            y_pred = forward_out["target_prediction_data"]["cell_type"].view(X_candidates_add.shape[0],
                                                                             config.n_populations,
                                                                             -1,
                                                                             len(classes))  
            y_pred_softmax = F.softmax(y_pred, dim=-1)
            
            # Take the mean proportions 
            y_pred_mean = y_pred.mean(2)  # no_candidates x no_populations x no_cell_types 
            y_pred_softmax = y_pred_softmax.mean(2)  # no_candidates x no_populations x no_cell_types 

            # Collect cell type of interest 
            y_pred_ct_std = y_pred_softmax[..., cell_type_index].std(1).detach().cpu().numpy()
            # Calculate loss
            y_target_ct = get_target_dict(config_run, classes, "cuda")["cell_type"]  # 1 x no_cell_type
            y_target_ct = y_target_ct.unsqueeze(0)  # 1 x 1 x no_cell_types
            
            logger.info("Compute loss function")
            loss_fn = get_loss_fn(config_run)["cell_type"]
            with torch.no_grad():
                loss = loss_fn(y_pred_mean, y_target_ct)  # no_candidates x no_populations
            loss_std = loss.std(1).detach().cpu().numpy()
            loss_mean = loss.mean(1).detach().cpu().numpy()
            
            result_csv[f"{cell_type_id}_prop_std"] = y_pred_ct_std
            result_csv["target_ct_loss_mean"] = loss_mean
            result_csv["target_ct_loss_std"] = loss_std
            
            y_pred_var = y_pred_softmax.var(1).detach().cpu().numpy()  # no_candidates x no_cell_types
            y_pred_total_var = y_pred_var.sum(1)  # no_candidates 
            result_csv["ct_prop_total_variance"] = y_pred_total_var
            for i, cell_type in enumerate(classes):
                if cell_type == cell_type_id:
                    continue
                result_csv[f"{cell_type}_prop_std"] = np.sqrt(y_pred_var[:, i])
                
            # Exploration & Exploitation scores 
            exploitation_score = compute_exploitation_score(loss_mean)
            # Leggi i dati veri e calcola le distanxe
            X_candidates = X_candidates.detach().cpu().numpy()
            Dts = cdist(X_candidates, X_real_concentrations)
            exploration_score = compute_exploration_score(Dts)
            weights = np.linspace(0, 1, 100)[None, :].repeat(exploitation_score.shape[0], axis=0)
            one_minus_weights = 1. - weights
            # Calcola interpolazioni e media 
            metric_response_surface = (weights * exploitation_score[:, None].repeat(100, axis=1) + one_minus_weights * exploration_score[:, None].repeat(100, axis=1))
            metric_response_surface = metric_response_surface.mean(1)
            
            weighted_uncertainty_score = compute_weighted_uncertainty_score(loss_mean, X_candidates, X_real_concentrations)
            
            result_csv["exploitation_score"] = exploitation_score
            result_csv["exploration_score"] = exploration_score
            result_csv["weighted_uncertainty_score"] = weighted_uncertainty_score
            result_csv["metric_response_surface"] = metric_response_surface
            
            # Save updated results 
            result_csv.to_csv(uncertainty_annotation_folder / "candidates_with_uncertainties.csv")
        
def parse_args():
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_cell_types", default=None)
    parser.add_argument("--n_noise_samples", type=int, default=2000)
    parser.add_argument("--n_populations", type=int, default=20)
    parser.add_argument("--base_config_path", required=True, default=...)
    parser.add_argument("--base_config_name", required=True, default=...)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--experiment_type", required=True)
    parser.add_argument("--true_concentration_path", required=True)
    parser.add_argument("--paths", required=True, default="default")
    return parser.parse_args()

def run():
    args = parse_args()
    main(args)

if __name__ == "__main__":
    run()
    