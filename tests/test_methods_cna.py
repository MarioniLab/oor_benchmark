import numpy as np

from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_cna


def test_method_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata.obs["OOR_state"] = np.where(adata.obs["louvain"] == "B cells", 1, 0)
    remove_cells = adata.obs_names[(adata.obs["OOR_state"] == 1) & (adata.obs["dataset_group"] != "query")]
    adata = adata[~adata.obs_names.isin(remove_cells)].copy()
    adata = scArches_cna.scArches_cna(adata, embedding_reference="atlas", diff_reference="ctrl")
    assert check_method(adata)
