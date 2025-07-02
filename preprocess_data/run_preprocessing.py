import numpy as np
import scanpy as sc
import rapids_singlecell as rsc


# paths
READ_PATH = "../../theis/HID01_fcs_concatenated.h5ad"
DUMP_PATH = "../../theis/HID01_fcs_concatenated_pp.h5ad"

# cofactors
COFACTORS = [
    1, 5, 10, 50, 100, 200, 300, 500, 700, 1_000, 2_500, 5_000, 7_500, 10_000
]


if __name__ == "__main__":

    adata = sc.read_h5ad(READ_PATH)
    for cofactor in COFACTORS:
        key_arcsin = f"X_arcsinh_cof{cofactor}"
        key_logabs = f"X_logabs_cof{cofactor}"
        print(f"Applying transformation for {cofactor=}")
        adata.layers[key_arcsin] = np.arcsinh(adata.X/cofactor)
        adata.layers[key_logabs] = np.sign(adata.X)*np.log(np.abs(adata.X / cofactor))

    # iterating over the representations
    for data_id in adata.layers.keys():
        print(data_id)

        # handling keys
        pca_key = f"{data_id}_pca"
        neighbors_key = f"{data_id}_neighbors"
        umap_key = f"{data_id}_umap"
        leiden_key = f"{data_id}_leiden"
        louvain_key = f"{data_id}_louvain"

        # computing pca, neighbors and umap
        rsc.pp.pca(adata, layer=data_id, key_added=pca_key)
        rsc.pp.neighbors(adata, use_rep=pca_key, key_added=neighbors_key)
        rsc.tl.umap(adata, neighbors_key=neighbors_key, key_added=umap_key)
        rsc.tl.louvain(adata, neighbors_key=neighbors_key, key_added=louvain_key)
        rsc.tl.leiden(adata, neighbors_key=neighbors_key, key_added=leiden_key)
    
    adata.write_h5ad(DUMP_PATH)
