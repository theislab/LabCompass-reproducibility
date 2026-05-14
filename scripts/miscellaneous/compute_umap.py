import scanpy as sc


input_path = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/loop2/BloodPlus_Loop2_logicle_500k_combined.h5ad"
output_path = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/miscellaneous_new/BloodPlus_Loop2_logicle_500k_combined_with_umap.h5ad"

if __name__ == "__main__":
    adata = sc.read_h5ad(input_path)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata.write_h5ad(output_path)
