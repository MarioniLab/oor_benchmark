import numpy as np
import pytest

from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_meld


@pytest.mark.skip(reason="Core dumped running this on github actions")
def test_method_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata.obs["OOR_state"] = np.where(adata.obs["louvain"] == "B cells", 1, 0)
    remove_cells = adata.obs_names[(adata.obs["OOR_state"] == 1) & (adata.obs["dataset_group"] != "query")]
    adata = adata[~adata.obs_names.isin(remove_cells)].copy()
    assert check_method(scArches_meld.scArches_atlas_meld_ctrl(adata))
    assert check_method(scArches_meld.scArches_atlas_meld_atlas(adata))
    assert check_method(scArches_meld.scArches_ctrl_meld_ctrl(adata))
