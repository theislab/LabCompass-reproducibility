import logging

import scanpy as sc

# initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    # read anndatas
    logger.info(f"Reading full data from {config.full_base_data_path}...")
    full_base_adata = sc.read_h5ad(config.full_base_data_path)
    logger.info(f"{config.full_base_adata=}")
    logger.info(f"Reading annotated subset of data from {config.annot_base_data_path}...")
    annot_base_adata = sc.read_h5ad(config.annot_base_data_path)
    logger.info(f"{config.annot_base_adata=}")
    logger.info(f"Reading additional experiment data from {config.annot_base_data_path}...")
    additional_adata = sc.read_h5ad(config.additional_base_data_path)
    logger.info(f"{additional_adata=}")

    # concatenating data
    logger.info(f"Concatenating new measurements with full data...")
    concat_full = sc.pp.concat((full_base_adata, additional_adata), uns_merge="same")
    logger.info(f"{concat_full=}")
    logger.info(f"Concatenating new measurements with annotated subset of data...")
    concat_annot = sc.pp.concat((annot_base_adata, additional_adata), uns_merge="same")
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
    parser.add_argument("--full_base_data_path", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/HID01_fcs_concatenated_logicle.h5ad")
    parser.add_argument("--annot_base_data_path", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/ds_HID01_logicle_100k_UMAP_ann.h5ad")
    parser.add_argument("--additional_base_data_path", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/LPHO012/projected_LPHO12_all_exp.h5ad")
    parser.add_argument("--full_concat_data_path", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/miscellaneous/HID01_fcs_concatenated_logicle_with_LPHO12.h5ad")
    parser.add_argument("--annot_concat_data_path", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/miscellaneous/ds_HID01_logicle_100k_UMAP_ann_with_LPHO12.h5ad")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
