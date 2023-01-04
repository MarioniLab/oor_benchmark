from anndata import AnnData


def prep_dataset(
    adata: AnnData,
    sample_id_col: str = "sample_id",
    annotation_col: str = "leiden",
):
    """Harmonize metadata in anndata object for OOR benchmark."""

    # assert "dataset_group" in adata.obs
    # assert "OOR_state" in adata.obs
    # assert "sample_id" in adata.obs
    # assert "cell_annotation" in adata.obs
    # assert all(adata.obs.loc[adata.obs["OOR_state"] == 1, "dataset_group"] == "query")
