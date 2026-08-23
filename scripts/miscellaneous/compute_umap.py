import scanpy as sc

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import os
os.environ["REPO_ROOT"] = str(REPO_ROOT)


input_path = str(REPO_ROOT / "project_folder" / "data/gdrive/loop2/BloodPlus_Loop2_logicle_500k_combined.h5ad")
output_path = str(REPO_ROOT / "project_folder" / "output/miscellaneous_new/BloodPlus_Loop2_logicle_500k_combined_with_umap.h5ad")

if __name__ == "__main__":
    adata = sc.read_h5ad(input_path)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata.write_h5ad(output_path)
