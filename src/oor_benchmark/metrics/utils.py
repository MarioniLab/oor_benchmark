import numpy as np
from anndata import AnnData


def make_OOR_per_group(adata: AnnData, frac_perc: int = 20):
    """Calculate OOR ground-truth per group.

    Parameters:
    -----------
    adata: AnnData
        AnnData object after running method
    frac_perc: int
        Percentile of maximum fraction of OOR cells in group to consider as OOR state
        (default: 20%)

    Returns:
    --------
    None, modifies adata in place
    """
    groups_mat = _get_sample_adata(adata).varm["groups"].copy()
    n_OOR_cells = groups_mat[:, adata.obs["OOR_state"] == 1].toarray().sum(1)
    frac_OOR_cells = n_OOR_cells / np.array(groups_mat.sum(1)).ravel()
    adata.uns["sample_adata"].var["n_OOR_cells"] = n_OOR_cells
    adata.uns["sample_adata"].var["frac_OOR_cells"] = frac_OOR_cells
    # Define threshold as 20% of max fraction
    OOR_thresh = frac_perc * (frac_OOR_cells.max() / 100)
    adata.uns["sample_adata"].var["OOR_state_group"] = (frac_OOR_cells > OOR_thresh).astype("int")


def _get_sample_adata(adata: AnnData):
    return adata.uns["sample_adata"]
