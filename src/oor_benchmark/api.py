import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csc_matrix


def check_dataset(adata: AnnData):
    """Check that dataset output fits expected API."""
    assert "dataset_group" in adata.obs
    assert "OOR_state" in adata.obs
    assert "sample_id" in adata.obs
    assert "cell_annotation" in adata.obs
    assert all(adata.obs.loc[adata.obs["OOR_state"] == 1, "dataset_group"] == "query")
    assert not _check_nonegative_integers_X(adata)
    return True


def check_method(adata: AnnData):
    """Check that method output fits expected API."""
    assert check_dataset(adata)
    assert "OOR_state" in adata.obs
    assert "sample_adata" in adata.uns
    assert "OOR_score" in adata.uns["sample_adata"].var
    assert "OOR_signif" in adata.uns["sample_adata"].var
    assert all(adata.uns["sample_adata"].var["OOR_signif"].isin([0, 1]))
    assert "groups" in adata.uns["sample_adata"].varm
    assert isinstance(adata.uns["sample_adata"].varm["groups"], csc_matrix)
    return True


def _check_nonegative_integers_X(adata):
    """Check that adata.X contains counts."""
    data_check = adata.X.data[0:100]
    negative = any(data_check < 0)
    non_integers = any(data_check % 1 != 0)
    return negative | non_integers


def sample_dataset():
    """Create a simple dataset to use for testing methods in this task."""
    np.random.seed(3456)
    adata = sc.datasets.pbmc3k_processed()
    adata_raw = sc.datasets.pbmc3k()
    adata.X = adata_raw[adata.obs_names][:, adata.var_names].X.copy()
    adata.obs["cell_annotation"] = adata.obs["louvain"].copy()
    # Split in samples and dataset group
    adata.obs["sample_id"] = np.random.choice([f"S{n}" for n in range(16)], size=adata.n_obs)
    adata.obs["dataset_group"] = np.nan
    adata.obs.loc[adata.obs["sample_id"].isin([f"S{n}" for n in range(8)]), "dataset_group"] = "atlas"
    adata.obs.loc[adata.obs["sample_id"].isin([f"S{n}" for n in range(8, 12)]), "dataset_group"] = "ctrl"
    adata.obs.loc[adata.obs["sample_id"].isin([f"S{n}" for n in range(12, 16)]), "dataset_group"] = "query"
    # # Make out-of-reference cell state
    # adata.obs["OOR_state"] = np.where(adata.obs["louvain"] == "B cells", 1, 0)
    # remove_cells = adata.obs_names[(adata.obs["OOR_state"] == 1) & (adata.obs["dataset_group"] != "query")]
    # adata = adata[~adata.obs_names.isin(remove_cells)].copy()
    return adata
