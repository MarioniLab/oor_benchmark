import meld
import numpy as np
import pandas as pd
from anndata import AnnData


def run_meld(
    adata_design: AnnData, query_group: str, reference_group: str, sample_col: str = "sample_id", n_neighbors=10
):
    """
    Run MELD to compute probability estimate per condition.

    Parameters:
    ------------
    adata_design : AnnData
        AnnData object of disease and reference cells to compare
    query_group : str
        Name of query group in adata_design.obs['dataset_group']
    reference_group : str
        Name of reference group in adata_design.obs['dataset_group']
    sample_col : str
        Name of column in adata_design.obs to use as sample ID
    n_neighbors : int
        Number of neighbors to use for MELD KNN graph (default: 10)
    """
    adata_design.obs["is_query"] = adata_design.obs["dataset_group"] == query_group
    adata_design.uns["n_conditions"] = 2

    # Complete the result in-place
    meld_op = meld.MELD(knn=n_neighbors, verbose=True)
    adata_design.obsm["sample_densities"] = meld_op.fit_transform(
        adata_design.obsm["X_scVI"], sample_labels=adata_design.obs[sample_col]
    ).set_index(adata_design.obs_names)

    # Normalize the probability estimates for each condition per replicate
    adata_design.obsm["probability_estimate"] = pd.DataFrame(
        np.zeros(shape=(adata_design.n_obs, adata_design.uns["n_conditions"])),
        index=adata_design.obs_names,
        columns=["query", "reference"],
    )

    query_samples = adata_design.obs["sample_id"][adata_design.obs["dataset_group"] == query_group].unique().tolist()
    reference_samples = (
        adata_design.obs["sample_id"][adata_design.obs["dataset_group"] == reference_group].unique().tolist()
    )

    adata_design.obsm["probability_estimate"]["query"] = adata_design.obsm["sample_densities"][query_samples].mean(
        axis=1
    )
    adata_design.obsm["probability_estimate"]["reference"] = adata_design.obsm["sample_densities"][
        reference_samples
    ].mean(axis=1)
    adata_design.obsm["probability_estimate"] = meld.utils.normalize_densities(
        adata_design.obsm["probability_estimate"]
    )

    return None
