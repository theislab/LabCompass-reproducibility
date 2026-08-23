import logging
import sys

import scanpy as sc

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import os
os.environ["REPO_ROOT"] = str(REPO_ROOT)

# initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    # import modules
    sys.path.insert(0, str(REPO_ROOT / "shared_utils"))
    from data_utils import drop_duplicates, ensure_type_safety

    # read anndatas
    logger.info(f"Reading full data from {config.full_base_data_path}...")
    full_base_adata = sc.read_h5ad(config.full_base_data_path)
    logger.info(f"{full_base_adata=}")

    logger.info(f"Reading annotated subset of data from {config.annot_base_data_path}...")
    annot_base_adata = sc.read_h5ad(config.annot_base_data_path)
    logger.info(f"{annot_base_adata=}")

    logger.info(f"Reading additional experiment data from {config.annot_base_data_path}...")
    additional_adata = sc.read_h5ad(config.additional_base_data_path)
    logger.info(f"{additional_adata=}")

    # concatenating data
    logger.info(f"Concatenating new measurements with full data...")
    concat_full = sc.concat((full_base_adata, additional_adata), uns_merge="same")
    concat_full = drop_duplicates(concat_full, logger=logger)
    concat_full = ensure_type_safety(concat_full)
    logger.info(f"{concat_full=}")


    logger.info(f"Concatenating new measurements with annotated subset of data...")
    concat_annot = sc.concat((annot_base_adata, additional_adata), uns_merge="same")
    concat_annot = drop_duplicates(concat_annot, logger=logger)
    concat_annot = ensure_type_safety(concat_annot)
    logger.info(f"{concat_annot=}")

    # saving to disk
    logger.info(f"Writing the full concatentaed data to {config.full_concat_data_path}...")
    concat_full.write_h5ad(config.full_concat_data_path)
    logger.info("Data written to disk!")
    logger.info(f"Writing the full concatentaed data to {config.annot_concat_data_path}...")
    concat_annot.write_h5ad(config.annot_concat_data_path)
    logger.info("Data written to disk!")
    logger.info("Run finished with exit code 0, goodbye!")


def parse_args():
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_base_data_path", required=False, default=str(REPO_ROOT / "project_folder" / "data/gdrive/HID01_fcs_concatenated_logicle.h5ad"))
    parser.add_argument("--annot_base_data_path", required=False, default=str(REPO_ROOT / "project_folder" / "data/gdrive/ds_HID01_logicle_100k_UMAP_ann.h5ad"))
    parser.add_argument("--additional_base_data_path", required=False, default=str(REPO_ROOT / "project_folder" / "data/gdrive/LPHO012/projected_LPHO12_all_exp.h5ad"))
    parser.add_argument("--full_concat_data_path", required=False, default=str(REPO_ROOT / "project_folder" / "output/miscellaneous/HID01_fcs_concatenated_logicle_with_LPHO12.h5ad"))
    parser.add_argument("--annot_concat_data_path", required=False, default=str(REPO_ROOT / "project_folder" / "output/miscellaneous/ds_HID01_logicle_100k_UMAP_ann_with_LPHO12.h5ad"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
