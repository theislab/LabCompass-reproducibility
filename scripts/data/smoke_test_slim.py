"""End-to-end check that slimmed checkpoints work in the real pipeline.

Points the repo's own `get_forward_model` at a slimmed loop and runs a forward
prediction, which is the first thing any downstream user or notebook does.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "shared_utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    from slim_checkpoints import CHECKPOINT_KEYS, LOOP_SETS, resolve

    release = REPO_ROOT / "project_folder/release/checkpoints" / args.loop
    config_dir = str(REPO_ROOT / "inverse/loss_guidance/config")

    # loops from 3 on were trained on the expanded 15-axis protocol space
    annotation = args.annotation or ("bloodplus_loop3" if args.loop in ("loop3", "loop4") else "default")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="run_inverse",
            overrides=[f"paths={LOOP_SETS[args.loop]}", f"annotation={annotation}"],
        )
    print(f"  annotation={annotation} ({len(config.annotation.protocol_columns)} protocol axes)", flush=True)

    # repoint every checkpoint at the slimmed copy
    OmegaConf.set_struct(config, False)
    for key in CHECKPOINT_KEYS:
        if config.paths.get(key):
            slim = release / resolve(config.paths[key]).name
            assert slim.exists(), f"missing slim checkpoint: {slim}"
            config.paths[key] = str(slim)
            print(f"  using {key}: {slim.name}", flush=True)

    from experiment_utils import get_forward_model

    print("\n>>> get_forward_model() on slimmed checkpoints", flush=True)
    forward_model, (flow_matching, target_prediction_model) = get_forward_model(config, logger=logger)
    print(">>> loaded OK", flush=True)

    # exercises train_data.data.perturbation_covariates and adata.obs, both of
    # which the pipeline reads out of the checkpoint
    from labcompass.constants import DataFields

    n_cond = len(config.annotation.protocol_columns)
    cell_type_column = config.loss.cell_type_column
    labels = target_prediction_model.train_data.adata.obs[cell_type_column].values
    print(f"\n>>> obs['{cell_type_column}']: {len(labels)} rows, "
          f"{len(set(map(str, labels[:100000])))} distinct in first 100k", flush=True)

    perturbation_reps = next(iter(forward_model.forward_model.train_data.data.perturbation_covariates))
    print(f">>> perturbation_covariates key: {perturbation_reps}", flush=True)

    print(f"\n>>> forward prediction, {n_cond} protocol axes", flush=True)
    torch.manual_seed(0)
    conditions = torch.zeros(2, n_cond).to(forward_model.forward_model.device)
    batch_dict = {DataFields.PERTURBATION_DATA: {perturbation_reps: conditions}}
    out = forward_model.predict(batch_dict, return_trajectory=False,
                                num_samples=args.n_samples, no_grad=True)
    tensor = out if torch.is_tensor(out) else next(
        v for v in (out.values() if isinstance(out, dict) else out) if torch.is_tensor(v))
    print(f">>> output shape={tuple(tensor.shape)} finite={bool(torch.isfinite(tensor).all())}", flush=True)

    print("\nSMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop", default="loop0")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--annotation", default=None)
    main(p.parse_args())
